# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def _preprocess_expert_vecs_for_cost(self, expert_vecs: torch.Tensor) -> torch.Tensor:
        """Center and project expert vectors before cosine-cost construction.

        Raw fc1 weights are very high-dimensional, which makes pairwise cosine
        similarities numerically collapse toward a near-constant off-diagonal
        matrix. To recover a more informative expert-expert geometry, we first
        subtract the across-expert mean vector, then project onto the leading
        principal directions of the expert-expert Gram matrix, and only then
        apply the final L2 normalisation used by the router-side cosine cost.
        """
        import torch.nn.functional as F

        vecs = expert_vecs.float()
        if vecs.ndim != 2 or vecs.size(0) <= 1:
            return F.normalize(vecs, p=2, dim=1)

        centered = vecs - vecs.mean(dim=0, keepdim=True)
        processed = centered

        default_proj_dim = 16
        proj_dim = int(getattr(self.config, 'moe_task_ot_sep_expert_proj_dim', default_proj_dim))
        proj_dim = max(1, min(proj_dim, centered.size(0) - 1))

        if proj_dim > 0:
            gram = torch.mm(centered, centered.t())
            gram = 0.5 * (gram + gram.t())
            eigvals, eigvecs = torch.linalg.eigh(gram)
            eigvals = eigvals.clamp_min(0.0)
            eigval_tol = eigvals.max().detach() * 1e-6 + eigvals.new_tensor(1e-8)
            top_eigvals = eigvals[-proj_dim:]
            top_eigvecs = eigvecs[:, -proj_dim:]
            positive = top_eigvals > eigval_tol
            if positive.any():
                processed = top_eigvecs[:, positive] * torch.sqrt(top_eigvals[positive]).unsqueeze(0)

        # Rare degenerate case: a projected row can be all zeros if all experts
        # collapse to the same mean. Fall back to centered/raw vectors per row.
        row_norms = processed.norm(dim=1, keepdim=True)
        if (row_norms <= 1e-12).any():
            centered_norms = centered.norm(dim=1, keepdim=True)
            fallback = torch.where(centered_norms > 1e-12, centered, vecs)
            processed = torch.where(row_norms > 1e-12, processed, fallback)

        return F.normalize(processed, p=2, dim=1)

    def _inject_expert_vecs(self) -> None:
        """Compute and inject per-expert parameter vectors into the router.

        Called once per forward pass when ``moe_task_ot_sep_c_type == 'expert'``
        is active.  Extracts the local experts' fc1 parameters (gate+up projection),
        all-gathers across the EP group so that every router rank holds the full
        [E, param_dim] matrix, centers and projects the expert vectors to a low-rank
        subspace, then normalises each row to unit length and stores the result in
        ``self.router._expert_vecs``.

        Supported expert implementations:
            - GroupedMLP: ``weight1`` with shape
              [hidden_size, num_local_experts * ffn_hidden_per_tp]
            - TEGroupedMLP: ``linear_fc1.weight{i}`` per local expert
            - SequentialMLP: ``local_experts[i].linear_fc1.weight``

        GroupedMLP ``weight1`` is reshaped as:
            [hidden_size, num_local_experts * ffn_hidden_per_tp]
            -> [hidden_size, num_local_experts, ffn_hidden_per_tp]
            -> [num_local_experts, hidden_size, ffn_hidden_per_tp]
            -> [num_local_experts, hidden_size * ffn_hidden_per_tp]

        After all-gather over the EP group:
            [num_experts, hidden_size * ffn_hidden_per_tp]

        The result is stored on the same device as the router weight and does
        NOT require gradients (detach), keeping the cost matrix fixed during
        the Sinkhorn iterations.
        """
        import torch.nn.functional as F

        # Only inject when the expert-cost mode is active.
        _c_type = getattr(self.config, 'moe_task_ot_sep_c_type', 'router')
        _ot_coeff = getattr(self.config, 'moe_task_ot_sep_coeff', 0.0)
        if _c_type != 'expert' or _ot_coeff <= 0.0:
            return

        # Retrieve the local experts' fc1 weights.
        experts = self.experts
        if hasattr(experts, 'weight1'):
            # Legacy GroupedMLP: weight1 shape [H, L * F].
            w1 = experts.weight1.detach().float()  # [H, L*F]
            H = self.config.hidden_size
            L = self.num_local_experts
            F_dim = w1.shape[1] // L
            local_vecs = w1.reshape(H, L, F_dim).permute(1, 0, 2).reshape(L, -1)  # [L, H*F]
        elif hasattr(experts, 'linear_fc1'):
            # TEGroupedMLP: one parameter per local expert, typically named weight0..weight{L-1}.
            fc1 = experts.linear_fc1
            vecs = []
            for i in range(self.num_local_experts):
                weight = getattr(fc1, f'weight{i}', None)
                if weight is None:
                    break
                vecs.append(weight.detach().float().reshape(-1))

            if len(vecs) != self.num_local_experts:
                named_params = dict(fc1.named_parameters())
                vecs = []
                for i in range(self.num_local_experts):
                    weight = named_params.get(f'weight{i}')
                    if weight is None:
                        vecs = []
                        break
                    vecs.append(weight.detach().float().reshape(-1))

            if not vecs:
                # Fallback: cannot determine TEGroupedMLP expert parameters, skip injection.
                return
            local_vecs = torch.stack(vecs, dim=0)
        elif hasattr(experts, 'local_experts'):
            # SequentialMLP: list of individual MLP modules, each with linear_fc1.weight.
            vecs = []
            for exp in experts.local_experts:
                w = exp.linear_fc1.weight.detach().float()  # [ffn_out, H]
                vecs.append(w.reshape(-1))
            local_vecs = torch.stack(vecs, dim=0)
        else:
            # Fallback: cannot determine expert parameters, skip injection.
            return

        # All-gather across EP group to get full [E, param_dim] matrix.
        ep_group = self.ep_group
        ep_size = utils.get_pg_size(ep_group)
        if ep_size > 1:
            # Each rank contributes local_vecs [L, D]; gather along EP to [E, D].
            # torch.distributed.all_gather requires a list of same-shape tensors.
            gathered = [torch.empty_like(local_vecs) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered, local_vecs, group=ep_group)
            # Concat in EP rank order → [E, D]
            all_vecs = torch.cat(gathered, dim=0)  # [E, D]
        else:
            all_vecs = local_vecs  # [E, D]

        # Center and project before the final cosine-normalisation so that
        # expert-mode OT uses a more informative geometry than raw fc1 weights.
        all_vecs = self._preprocess_expert_vecs_for_cost(all_vecs)

        # Store in router (detached; no gradient through cost matrix).
        self.router._expert_vecs = all_vecs.to(device=self.router.weight.device)

    def router_and_preprocess(
        self,
        hidden_states: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.

        Args:
            hidden_states: Input tensor to the MoE layer.
            task_ids: Optional long tensor of per-token task IDs with shape
                [seq, bsz] or [T].  Forwarded to the router when
                ``moe_use_task_condition`` is True.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states, task_ids=task_ids)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def routed_experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.
            task_ids (torch.Tensor, optional): Long tensor of per-token task IDs with shape
                [seq, bsz] or [T].  When provided and ``moe_use_task_condition`` is True,
                the task condition is injected into the router logits.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        # task_ids is captured by closure; it is a tensor (or None) and does NOT
        # need to be passed as a positional arg to checkpoint() – which only
        # saves / replays positional args.  The closure capture is safe because
        # task_ids is read-only during the forward pass and recompute.
        def custom_forward(hidden_states):
            # Inject expert parameter vectors into the router before routing,
            # so that the OT cost matrix C[e,f]=1-cos(θ_e,θ_f) is up-to-date.
            self._inject_expert_vecs()
            shared_expert_output = self.shared_experts_compute(hidden_states)
            hidden_states, probs, residual = self.router_and_preprocess(
                hidden_states, task_ids=task_ids
            )
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
