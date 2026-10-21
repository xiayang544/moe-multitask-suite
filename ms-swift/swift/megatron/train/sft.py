# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers.utils import is_torch_npu_available

from swift.llm import TEMPLATE_MAPPING
from swift.llm.dataset import load_dataset
from swift.llm.train import SwiftSft
from swift.utils import get_logger, is_last_rank, plot_images
from ..argument import MegatronTrainArguments
from ..trainers import MegatronTrainer
from ..utils import get_padding_to
from .utils import build_streaming_dataloader

if is_torch_npu_available():
    # Enable Megatron on Ascend NPU
    from mindspeed.megatron_adaptor import repatch
else:
    repatch = None

logger = get_logger()


class _DatasetNameInjectWrapper(Dataset):
    """Wraps a LazyLLMDataset so that each encoded sample carries the
    ``dataset_name`` field from the raw (pre-encode) row.

    This is needed by the task-conditioned MoE router, expert-wise PCGrad and
    per-dataset (channel) loss tracking: every sample must be labelled with
    its source dataset so the collator can produce a ``task_ids`` tensor that
    maps tokens to task indices, and a ``channel`` field for per-task metrics.

    The wrapper preserves the full ``Dataset`` interface expected by the
    Megatron data loading pipeline.
    """

    def __init__(self, dataset: Dataset, dataset_name_field: str = 'dataset_name') -> None:
        self._dataset = dataset
        self._dataset_name_field = dataset_name_field
        # Proxy attribute access so that callers that inspect the wrapped
        # dataset (e.g. to read ``dataset_type``, ``split``, …) still work.
        self.__dict__['_dataset'] = dataset

    # ------------------------------------------------------------------ #
    # Forward attribute reads to the inner dataset so that external code
    # that reads e.g. ``dataset.split`` continues to work.
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str):
        # __getattr__ is only called when the normal lookup fails, so this
        # does not create infinite recursion.
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx):
        # ``idx`` can be an int (normal indexing) or a string (used by some
        # Megatron samplers to inspect dataset columns such as 'length').
        if isinstance(idx, str):
            return self._dataset[idx]

        # ---------------------------------------------------------------- #
        # Retrieve the raw row from the *inner* dataset before encoding so
        # that we can read dataset_name.  The inner dataset is a
        # LazyLLMDataset which wraps a HuggingFace dataset; calling
        # ``self._dataset.dataset[idx]`` gives us the raw row.
        # ---------------------------------------------------------------- #
        raw_row: Dict[str, Any] = {}
        inner = getattr(self._dataset, 'dataset', None)
        if inner is not None:
            try:
                raw_row = inner[idx]
            except Exception:
                pass

        encoded = self._dataset[idx]

        # Inject dataset_name from raw row into the encoded sample.
        dataset_name = raw_row.get(self._dataset_name_field)
        if dataset_name is not None and isinstance(encoded, dict):
            encoded['dataset_name'] = dataset_name

        return encoded


def _build_task_ids_collator(
    base_collator: Callable,
    dataset_name_to_id: Dict[str, int],
    cp_pad_length_fn: Optional[Callable[[int], int]] = None,
    default_id: int = -1,
) -> Callable:
    """Return a collator that calls *base_collator* and then appends a
    ``task_ids`` tensor to the resulting batch.

    ``task_ids`` is an int32 tensor of shape ``[batch_size, seq_len]``.
    Each row holds the task-id of the corresponding sample, broadcast
    across its token dimension.  This shape mirrors ``input_ids`` so that
    the existing CP/TP split helpers can handle it without modification.

    **Packing / padding-free mode**:
    When ``padding_free=True`` the base collator packs multiple samples into
    a single sequence of length ``total_tokens``.  In that case the output
    shape is ``[1, total_tokens]`` and each token's task-id must come from
    the sample it originally belonged to.  We detect this case by checking
    whether the output ``batch_size`` is 1 while ``len(sample_names) > 1``
    and reconstruct the per-token mapping from each sample's ``input_ids``
    length (obtained before the collator merges them).

    Args:
        base_collator: The underlying collate function to wrap.
        dataset_name_to_id: Mapping from dataset name to integer task id.
        cp_pad_length_fn: Optional function ``(raw_len) -> padded_len`` that
            accounts for the CP-alignment padding that the template adds to
            each sample before packing (see ``Template._handle_megatron_cp``).
            When provided it is applied to each sample's raw length so that
            the per-token task-id assignment matches the actual packed layout.
        default_id: Task-id used for samples without a resolvable
            ``dataset_name``.  ``-1`` (PCGrad masking: the sample is masked
            out of every per-task pass but still contributes via the combined
            pass) or ``0`` (task-conditioned routing: mcore's task embedding
            lookup rejects negative indices).

    If a sample does not have a ``dataset_name`` key or the name is not in
    ``dataset_name_to_id``, its task-id defaults to ``default_id``.
    """

    def collator_with_task_ids(batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Packing datasets return a list-of-lists; flatten to a flat list of
        # sample dicts so we can iterate over individual samples.
        if batch and isinstance(batch[0], list):
            flat_batch: List[Dict[str, Any]] = sum(batch, start=[])
        else:
            flat_batch = batch

        # Extract dataset_name *before* passing to the base collator because
        # base collator may raise on unknown keys.
        # When enable_channel_loss is also active the outer collator may have
        # already renamed dataset_name -> channel before we run; in that case
        # fall back to reading the channel field.
        sample_names = [
            b.pop('dataset_name', None) or b.get('channel', None)
            for b in flat_batch
        ]
        ids_per_sample = [dataset_name_to_id.get(name, default_id) for name in sample_names]

        # Record per-sample raw token counts (before packing merges them).
        # We look for 'input_ids' first then 'labels' as a fallback.
        # These lengths are in *raw* form (before CP-alignment padding).
        per_sample_lengths: Optional[List[int]] = None
        if len(flat_batch) > 0:
            lengths = []
            for b in flat_batch:
                src = b.get('input_ids') or b.get('labels')
                if src is not None:
                    lengths.append(len(src) if isinstance(src, list) else src.shape[0])
                else:
                    lengths = None
                    break
            per_sample_lengths = lengths

        res = base_collator(batch, **kwargs)

        # Determine sequence length and batch size from the batch output.
        out_batch_size: Optional[int] = None
        seq_len: Optional[int] = None
        for key in ('input_ids', 'labels', 'loss_scale', 'position_ids'):
            t = res.get(key)
            if t is not None and isinstance(t, torch.Tensor) and t.ndim >= 2:
                # After collation: [batch_size, seq_len] (or [1, seq_len] for
                # padding-free packing where batch_size collapses to 1).
                out_batch_size = t.shape[0]
                seq_len = t.shape[-1]
                break

        if seq_len is None or out_batch_size is None:
            # Cannot determine shape; skip task_ids.
            return res

        num_orig_samples = len(ids_per_sample)
        is_packing_mode = (out_batch_size == 1 and num_orig_samples > 1)

        if is_packing_mode and per_sample_lengths is not None:
            # Padding-free / packing: reconstruct per-token task-ids by
            # repeating each sample's id for its (CP-padded) token count.
            token_ids: List[int] = []
            for tid, raw_len in zip(ids_per_sample, per_sample_lengths):
                # Apply CP-alignment padding if a padding function is given.
                eff_len = cp_pad_length_fn(raw_len) if cp_pad_length_fn is not None else raw_len
                token_ids.extend([tid] * eff_len)
            # Safety guard: pad or truncate to the actual collated seq_len.
            if len(token_ids) < seq_len:
                token_ids.extend([default_id] * (seq_len - len(token_ids)))
            else:
                token_ids = token_ids[:seq_len]
            task_ids = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0)  # [1, seq_len]
        else:
            # Standard padded mode: broadcast per-sample id across seq_len.
            ids_tensor = torch.tensor(ids_per_sample, dtype=torch.int32)  # [B]
            task_ids = ids_tensor.unsqueeze(1).expand(out_batch_size, seq_len).contiguous()

        res['task_ids'] = task_ids  # [batch_size, seq_len]
        return res

    return collator_with_task_ids


def _build_channel_from_dataset_name_collator(base_collator: Callable) -> Callable:
    """Wrap *base_collator* to rename the ``dataset_name`` field injected by
    ``_DatasetNameInjectWrapper`` into the ``channel`` field expected by
    ``enable_channel_loss`` loss tracking.

    This allows per-dataset loss logging without any modification to the
    dataset files: the dataset name is injected at load time and then exposed
    as a ``channel`` string that the Megatron trainer's ``loss_func`` uses to
    accumulate per-channel metrics.
    """

    def collator_with_channel(batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Packing datasets return a list-of-lists; flatten first.
        if batch and isinstance(batch[0], list):
            flat_batch: List[Dict[str, Any]] = sum(batch, start=[])
        else:
            flat_batch = batch

        # Rename dataset_name -> channel before passing to the base collator
        # so that the template's collator stores it as the channel field.
        for sample in flat_batch:
            if 'dataset_name' in sample and 'channel' not in sample:
                sample['channel'] = sample.pop('dataset_name')
            else:
                # Remove stray dataset_name so the base collator doesn't choke.
                sample.pop('dataset_name', None)

        return base_collator(batch, **kwargs)

    return collator_with_channel


class MegatronSft(SwiftSft):
    args_class = MegatronTrainArguments
    args: args_class

    def prepare_trainer(self):
        return MegatronTrainer(self.args, self.template)

    def __init__(self, args: Optional[Union[List[str], MegatronTrainArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        if repatch is not None:
            if args.attention_backend != 'local':
                # MindSpeed requires passing `use_flash_attn` to Megatron
                # to enable flash attention on Ascend NPU.
                args.use_flash_attn = True
            megatron_args = asdict(self.args)
            repatch(megatron_args)
        template_cls = TEMPLATE_MAPPING[args.template].template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs, download_model=args.load is None)
        self._prepare_template()
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
        args.save_args(args.save)
        self.template.use_megatron = True
        self.trainer = self.prepare_trainer()

    # ------------------------------------------------------------------ #
    # Override _get_dataset to inject ``dataset_name`` column into each
    # sub-dataset before they are concatenated, when the task-conditioned
    # router, task-homogeneous batching, expert PCGrad, or
    # enable_channel_loss is enabled.
    # ------------------------------------------------------------------ #
    def _get_dataset(self):
        args = self.args
        need_inject = (
            args.moe_use_task_condition
            or args.moe_task_bias_enabled
            or args.enable_expert_pcgrad
            or args.enable_channel_loss
            or args.task_homogeneous_batch
        )
        if not need_inject:
            return super()._get_dataset()
        # Replicate SwiftSft._get_dataset logic but pass inject_dataset_name=True.
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                inject_dataset_name=True,
                **dataset_kwargs)
        if len(args.val_dataset) > 0:
            dataset_kwargs.pop('interleave_prob', None)
            _, val_dataset = load_dataset(
                args.val_dataset,
                split_dataset_ratio=1.0,
                shuffle=args.val_dataset_shuffle,
                inject_dataset_name=True,
                **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        return train_dataset, val_dataset

    # ------------------------------------------------------------------ #
    # Post-process datasets: optionally inject dataset_name into samples.
    # ------------------------------------------------------------------ #
    def _post_process_datasets(self, datasets):
        """Wrap datasets with ``_DatasetNameInjectWrapper`` when the
        task-conditioned router, enable_channel_loss, or enable_expert_pcgrad
        is enabled, then delegate to the parent.
        """
        datasets = super()._post_process_datasets(datasets)
        args = self.args
        need_inject = (
            args.moe_use_task_condition
            or args.moe_task_bias_enabled
            or args.enable_channel_loss
            or args.enable_expert_pcgrad
        )
        if need_inject:
            if args.moe_use_task_condition or args.moe_task_bias_enabled:
                logger.info(
                    'moe_use_task_condition=True: wrapping datasets to inject '
                    'dataset_name into encoded samples for task_ids generation.')
            if args.enable_channel_loss:
                logger.info(
                    'enable_channel_loss=True: wrapping datasets to inject '
                    'dataset_name into encoded samples for per-dataset loss tracking.')
            if args.enable_expert_pcgrad:
                logger.info(
                    'enable_expert_pcgrad=True: wrapping datasets to inject '
                    'dataset_name into encoded samples for per-task expert gradient collection.')
            datasets = [
                _DatasetNameInjectWrapper(ds) if ds is not None else None for ds in datasets
            ]
        return datasets

    # ------------------------------------------------------------------ #
    # Stat override: unwrap ``_DatasetNameInjectWrapper`` before delegating
    # to the parent implementation.  Without this, ``SwiftSft._stat_dataset``
    # falls through both ``isinstance(..., LazyLLMDataset)`` and
    # ``isinstance(..., HfDataset)`` checks and ends up accessing
    # ``packed_length`` via ``__getattr__``, which the underlying
    # ``LazyLLMDataset`` does not provide -> ``AttributeError``.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stat_dataset(dataset):
        if isinstance(dataset, _DatasetNameInjectWrapper):
            dataset = dataset._dataset
        return SwiftSft._stat_dataset(dataset)

    def _get_data_collator(self):
        data_collator = self.template.data_collator
        padding_to = get_padding_to(self.args)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)

        args = self.args
        # Build task_ids collator when the task-conditioned router or
        # expert PCGrad is enabled.  Both features require per-token task
        # identity to be available in the batch.
        task_condition_enabled = args.moe_use_task_condition or args.moe_task_bias_enabled
        need_task_ids = task_condition_enabled or args.enable_expert_pcgrad
        if need_task_ids:
            dataset_names = list(args.moe_task_dataset_names)
            dataset_name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(dataset_names)}
            if task_condition_enabled:
                logger.info(
                    f'moe_use_task_condition=True: building task_ids in collator. '
                    f'dataset_name -> task_id mapping: {dataset_name_to_id}')
            if args.enable_expert_pcgrad:
                logger.info(
                    f'enable_expert_pcgrad=True: building task_ids in collator. '
                    f'dataset_name -> task_id mapping: {dataset_name_to_id}')
            # Warn about datasets that will fall back to the default task id.
            for ds_name in args.dataset:
                if ds_name not in dataset_name_to_id:
                    logger.warning(
                        f'[task-tag] Dataset "{ds_name}" is not in moe_task_dataset_names '
                        f'({dataset_names}). Its task_id will default to '
                        f'{"0 (task routing)" if task_condition_enabled else "-1 (masked out of per-task PCGrad passes)"}.')
            # task-conditioned routing: mcore's task embedding lookup
            # (F.embedding) rejects negative indices -> default to task 0;
            # PCGrad masking: -1 keeps unmapped samples out of every
            # per-task pass (they still contribute via the combined pass).
            default_id = 0 if task_condition_enabled else -1
            # Pass the CP-alignment padding function so that packing mode can
            # correctly account for the per-sample padding added by the template
            # before sequences are concatenated (see Template._handle_megatron_cp).
            cp_pad_fn = None
            if self.template.padding_free and self.template.use_megatron:
                cp_pad_fn = self.template._get_megatron_cp_length
            data_collator = _build_task_ids_collator(
                data_collator, dataset_name_to_id, cp_pad_fn, default_id=default_id)

        if args.enable_channel_loss:
            logger.info(
                'enable_channel_loss=True: wrapping collator to map '
                'dataset_name -> channel for per-dataset loss tracking.')
            data_collator = _build_channel_from_dataset_name_collator(data_collator)

        return data_collator

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        data_collator = self._get_data_collator()

        if args.streaming:
            train_dataset = build_streaming_dataloader(args, train_dataset, data_collator)
            if val_dataset is not None:
                val_dataset = build_streaming_dataloader(args, val_dataset, data_collator)

        try:
            self.trainer.train(train_dataset, val_dataset, data_collator)
        finally:
            # Visualization
            if is_last_rank():
                images_dir = os.path.join(args.save, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Optional[Union[List[str], MegatronTrainArguments]] = None):
    return MegatronSft(args).main()
