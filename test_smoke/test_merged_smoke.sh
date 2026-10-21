#!/bin/bash
# =============================================================================
# test_merged_smoke.sh
# ms-swift-merged 冒烟测试：单节点 8 GPU，2 数据集，~20 iters。
#
# 验证合并包的三个功能（以及组合）：
#   1. 任务感知路由  --moe_use_task_condition / --num_tasks / --task_router_mode bias
#   2. 任务 loss     --enable_channel_loss true（tensorboard 应出现 loss_<ds>）
#   3. 专家 PCGrad   --enable_expert_pcgrad true（每步 T+1 次前后向，可观察 step 耗时）
#
# 用法：
#   MODE=all      （默认）组合测试：三功能全开 —— 合并的核心价值
#   MODE=task     仅任务路由
#   MODE=channel  仅任务 loss
#   MODE=pcgrad   仅 PCGrad
#   MODE=baseline 全关回归（行为应与 base commit 完全一致）
#   ITERS=N       训练步数（默认 20）
#
# 判定（看 LOG_FILE / tensorboard）：
#   - [task_ids in collator. dataset_name -> task_id mapping: {...}]  → task_ids 已生成
#   - [task_router] Saved task router state ... task_router_state.pt   → 保存成功（需 save 触发）
#   - tensorboard 出现 loss_task_a / loss_task_b（即 loss_<数据集名>）
#   - enable_expert_pcgrad=True 时每步耗时约为 baseline 的 (T+1)/1 倍
# =============================================================================
source /opt/rh/devtoolset-8/enable

set -eo pipefail

MODE=${MODE:-all}
ITERS=${ITERS:-20}

# ---- 环境配置 ----
ROOT=$(cd "$(dirname "$0")/.." && pwd)
env_path=${SWIFT_CONDA_ENV:-/path/to/conda/env}   # 填入你的 conda 环境路径（或 export SWIFT_CONDA_ENV）
export PATH="${env_path}/bin:$PATH"

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${env_path}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH

export HF_DATASETS_CACHE="/workdir/data_cache_TMP"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${ROOT}/test_smoke/.cache/torch_extensions}

# ---- 关键：指向合并包与 patched Megatron-Core ----
export MEGATRON_LM_PATH=${ROOT}/Megatron-LM-core_v0.15.0
SWIFT_DIR=${ROOT}/ms-swift
export PYTHONPATH="${SWIFT_DIR}:${MEGATRON_LM_PATH}:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-30B-A3B-Instruct-2507-mcore}   # Megatron 权重格式的 MoE 模型

cd $SWIFT_DIR

# ---- 分布式配置（单节点 8 GPU）----
if [ -n "${AFO_ENV_CLUSTER_SPEC}" ]; then
    master_addr=$(echo "${AFO_ENV_CLUSTER_SPEC}" | python3 -c "import sys,json; print(json.load(sys.stdin)['worker'][0])")
    node_rank=$(echo "${AFO_ENV_CLUSTER_SPEC}" | python3 -c "import sys,json; print(json.load(sys.stdin)['index'])")
    head_hostname=$(echo "$master_addr" | cut -d':' -f1)
    port=$(echo "$master_addr" | cut -d':' -f2)
else
    head_hostname=${MASTER_ADDR:-localhost}
    port=${MASTER_PORT:-29500}
    node_rank=${NODE_RANK:-0}
fi

# ---- 实验配置 ----
DS1=task_a
DS2=task_b

exp_name="test_merged_smoke_${MODE}"
OUTPUT_DIR=${ROOT}/test_smoke/${exp_name}/model
TENSORBOARD_DIR=${ROOT}/test_smoke/${exp_name}/tensorboard
mkdir -p $OUTPUT_DIR $TENSORBOARD_DIR

LOG_FILE="${OUTPUT_DIR}/../run_${node_rank}_$(date +%Y%m%d_%H%M%S).log"
echo "[${exp_name}] logging to ${LOG_FILE}"

# ---- 按 MODE 组装功能 flag ----
feature_flags=""
case "${MODE}" in
    all)
        feature_flags="\
    --moe_use_task_condition true \
    --num_tasks 2 \
    --task_router_mode bias \
    --enable_channel_loss true \
    --enable_expert_pcgrad true \
    --moe_task_dataset_names ${DS1} ${DS2}"
        ;;
    task)
        feature_flags="\
    --moe_use_task_condition true \
    --num_tasks 2 \
    --task_router_mode bias \
    --moe_task_dataset_names ${DS1} ${DS2}"
        ;;
    channel)
        feature_flags="--enable_channel_loss true"
        ;;
    pcgrad)
        feature_flags="\
    --enable_expert_pcgrad true \
    --moe_task_dataset_names ${DS1} ${DS2}"
        ;;
    baseline)
        feature_flags=""
        ;;
    *)
        echo "Unknown MODE=${MODE} (use all|task|channel|pcgrad|baseline)"; exit 1
        ;;
esac

# ---- 启动训练 ----
NNODES=1 \
NODE_RANK=$node_rank \
NPROC_PER_NODE=8 \
MASTER_ADDR=$head_hostname \
MASTER_PORT=$port \
megatron sft \
    --load ${MODEL_PATH} \
    --dataset ${DS1} ${DS2} \
    --split_dataset_ratio 0 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff 0 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-6 \
    --lr_warmup_iters 2 \
    --min_lr 1e-7 \
    --train_iters ${ITERS} \
    --max_length 8192 \
    --save ${OUTPUT_DIR} \
    --tensorboard_dir ${TENSORBOARD_DIR} \
    --save_interval 100000 \
    --log_interval 1 \
    --num_workers 8 \
    --save_safetensors false \
    --ckpt_format torch_dist \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8 \
    --model_author demo \
    --model_name task-router-smoke \
    --use_flash_attn true \
    ${feature_flags} \
    2>&1 | tee "${LOG_FILE}"
