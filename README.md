# MoE 多任务训练套件 (moe-multitask-suite)

ms-swift + 定制 Megatron-Core 的自包含合并包，提供三个相互独立、可任意组合的 MoE 多任务训练功能：

| 功能 | 开关 | 说明 |
|---|---|---|
| **任务感知路由** | `--moe_use_task_condition true` | 每层 MoE router 根据 task_id 注入可学习的路由修正（task_bias / task_emb+task_proj），训练与推理行为一致 |
| **按任务打印 loss** | `--enable_channel_loss true` | 每个数据集（=任务）的 loss 单独进 tensorboard（`loss_<数据集名>`） |
| **专家粒度 PCGrad** | `--enable_expert_pcgrad true` | 每 step 做 T+1 次前后向：T 个任务分别前向取专家梯度，PCGrad 投影消解任务间梯度冲突后再合并更新 |

## 目录结构

```
moe-multitask-suite/
├── README.md                          # 本文件
├── ms-swift/                          # ms-swift 合并包（base commit d2644e010 + 三功能，无 git 历史）
│   └── swift/...
├── Megatron-LM-core_v0.15.0/          # patched Megatron-Core（任务路由的全部模型侧实现在这里）
│   └── megatron/core/...
├── patches/
│   └── megatron-core_task-router.patch  # mcore 相对 pip 原版 megatron_core-0.15.0 的完整 diff（溯源用）
└── test_smoke/                        # 冒烟测试（hope + sh，单节点 8 GPU）
    ├── test_merged_smoke.sh
    └── test_merged_smoke.hope
```

## 环境要求

- conda 环境：需含 `megatron` CLI 入口与全部依赖（Python 3.10 + CUDA 12.2）
- GPU 镜像：任意含 CUDA 12.2 的训练镜像即可

## 快速开始

**关键约束：用 `PYTHONPATH` 指向本套件，不要 `pip install -e`**（共享 env 的 easy-install.pth 指向别的工作副本，安装会破坏其他实验脚本）。

```bash
ROOT=/path/to/moe-multitask-suite   # 本仓库根目录
export PYTHONPATH="${ROOT}/ms-swift:${ROOT}/Megatron-LM-core_v0.15.0"
export MEGATRON_LM_PATH="${ROOT}/Megatron-LM-core_v0.15.0"   # 必须用本套件内的 mcore

megatron sft \
    --load <模型路径> \
    --dataset dsA dsB dsC ... \
    --moe_use_task_condition true \
    --num_tasks <任务数> \
    --task_router_mode bias \
    --enable_channel_loss true \
    --enable_expert_pcgrad true \
    --moe_task_dataset_names "dsA,dsB,dsC" \
    ...其余训练参数...
```

`--moe_task_dataset_names`：数据集名 → task_id 的映射表，**顺序即编号**（第 i 个 → task_id=i）；逗号串与空格分隔两种写法都支持。开了任务路由或 PCGrad 时必填。

### 冒烟测试

```bash
cd ${ROOT}/test_smoke
hope run test_merged_smoke.hope        # 默认 MODE=all：三功能全开，20 iters
```

切换模式：改 `test_merged_smoke.sh` 里 `MODE=${MODE:-all}` 的默认值（或提交前 `export MODE=...`）：

| MODE | 验证点 |
|---|---|
| `all` | 三功能组合（合并的核心价值：channel 回退 + 共享 dataset_name 注入路径） |
| `task` | 日志出现 `dataset_name -> task_id mapping: {...}` |
| `channel` | tensorboard 出现 `loss_task_a` / `loss_task_b` 两条 |
| `pcgrad` | 每步耗时 ~3× baseline（T+1=3 次前后向） |
| `baseline` | 全关，行为应与上游 ms-swift 完全一致（回归对照） |

## 训练 → 导出 → 推理链路（任务感知路由）

1. **训练**：`task_ids` 全链路由 ms-swift 侧生成并送达模型——
   `loader.py` 注入 `dataset_name` 列 → `_DatasetNameInjectWrapper` 带入编码样本 →
   task_ids collator 生成 `[bsz, seq]`（packing 模式为 `[1, total_tokens]`）→
   CP 按序列维切分 → `_prepare_batch` 展平为 token-major 1D →
   `GPTModel.forward(task_ids=...)` → 各 TransformerLayer → MoE router。
2. **导出**：`gpt_bridge._save_task_router_state` 在 HF 导出目录额外写
   `task_router_state.pt`（每层的 task_bias / task_emb / task_proj + 元信息）。
   注意 task_bias 等参数在 HF safetensors 转换中会被**静默跳过**（无标准 key），
   所以走纯 HF 路径加载会丢失任务条件——必须配合 step 3。
3. **推理**：由外部 vLLM wrapper 消费 `task_router_state.pt`
   （评测脚本传 `--task_bias_path <dir>/task_router_state.pt` 与
   `--task_id_domain_map "taskA:0,taskB:1,..."` 做按请求动态切换 task_id）。
   vLLM wrapper 不在本套件内。

断点续训无需特殊处理：task_bias / task_emb / task_proj 是 TopKRouter 内的标准
nn.Module 参数，走 Megatron torch_dist 分布式 checkpoint 自动保存与恢复。

## 三个功能的参数速查

| 参数 | 默认 | 作用 |
|---|---|---|
| `--moe_use_task_condition` | false | 开任务路由（含旧别名 `moe_task_bias_enabled`） |
| `--num_tasks` | 0 | 任务数（含旧别名 `moe_task_bias_num_tasks`），开路由时必填 >0 |
| `--task_router_mode` | bias | `bias` / `bias_plus_proj` / `emb_proj_only`（`film` 参数保留但当前 mcore 不支持，会在校验时告警） |
| `--task_router_emb_dim` | 64 | emb 维度（`bias_plus_proj` / `emb_proj_only` 模式） |
| `--moe_task_dataset_names` | [] | 数据集名列表（顺序=task_id），路由与 PCGrad 必填 |
| `--enable_channel_loss` | false | 按数据集打 loss 到 tensorboard |
| `--enable_expert_pcgrad` | false | 专家粒度 PCGrad（要求 MoE 模型 + 上面 names 必填） |
| `--expert_pcgrad_eps` | 1e-8 | PCGrad 投影分母 eps |
| `--task_homogeneous_batch` | false | 每个 global batch 只含单任务样本（配套 `--task_homogeneous_field`，默认 `dataset_name`） |

## 架构决策备忘（合并时两份源代码的冲突取舍）

- **task_ids 张量流水线**采用 margin 方案：collator 产出 `[bsz, seq]`，`_prepare_batch`
  展平为 token-major 1D。mcore router 的 `_align_task_ids_to_tokens` 两种布局都接受；
  若同时叠加 projects 的 utils.py 转置方案会双重转置导致 per-token id 错乱。
- **task_ids collator 的 default_id**：任务路由时为 `0`（mcore 的 `F.embedding` 不接受
  负索引），PCGrad 时为 `-1`（未映射样本从每个 per-task pass 中掩蔽，仅经合并 pass 贡献）。
  两功能同开时取 `0`。
- **与 channel loss 的组合**：channel collator 在最外层把 `dataset_name` 改名为
  `channel`，task_ids collator 保留了 `channel` 回退行，因此二者可同 batch 共存。
- **`streaming=true` 不支持**（dataset_name 注入被跳过，参数校验会拦）；建议
  `--split_dataset_ratio 0`（val split 无 dataset_name）。

## 已知限制

- `task_router_mode film`：参数与保存逻辑保留，但当前 mcore 只实现了三种模式
  （`transformer_config` 会 assert）；复活 film 需改 mcore 侧。
- PCGrad：要求 main_grad（分布式 optimizer 路径，否则回退 `.grad`）；每步 T+1 次
  前后向的开销约为 baseline 的 (T+1)/1；与 GKD / PP / VP 的组合未验证。
- mcore patch 内含休眠的 overlap / OT-sep aux loss 代码（config 系数控制，默认 0，
  ms-swift 侧已无对应参数，无法启用），见 `patches/` 中的 diff。

## 来源与版本

| 组件 | 来源 | 版本基准 |
|---|---|---|
| ms-swift | `projects/ms-swift`（任务路由）+ `margin/ms-swift`（channel loss、PCGrad）手工合并 | 上游 commit `d2644e010`（[bugfix] fix megatron kto pp + sp (#7882)） |
| Megatron-Core | `projects/Megatron-LM-core_v0.15.0`（已在用的工作副本，未做任何新修改） | pip `megatron_core-0.15.0` + 任务路由补丁（`patches/megatron-core_task-router.patch`） |

开发用 git worktree 在 `projects/ms-swift-merged`（分支 `merged-features`，可追踪
合并过程的每一步）；本套件是其快照 + mcore 副本的自包含分发形态。**后续若在
worktree 里改代码，记得重新同步到本套件**（复制 8 个改动文件 + test_smoke 即可）。

ms-swift 侧合并共改动 8 个文件（+1156 行）：
`swift/llm/dataset/loader.py`、`swift/megatron/argument/megatron_args.py`、
`swift/megatron/init.py`、`swift/megatron/model/gpt_bridge.py`、
`swift/megatron/train/sft.py`、`swift/megatron/trainers/base.py`、
`swift/megatron/trainers/gkd_trainer.py`、`swift/megatron/trainers/utils.py`。
