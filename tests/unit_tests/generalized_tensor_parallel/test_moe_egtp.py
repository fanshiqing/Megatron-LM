# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for EGTP_remat + MoE correctness.

Test groups
-----------
TestMoEEGTPCorrectness  - EGTP_remat MoE loss trajectory matches baseline (no-EGTP_remat) over 10
                          training steps using MXFP8 and Nemotron3-Super MoE hyperparameters.
TestMoEEGTPPrecisionOverride
                        - EGTP_remat grouped experts that a te-precision-config override keeps
                          off the op-fuser path must still complete backward.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _assert_loss_trajectories_match,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

# ---------------------------------------------------------------------------
# MoE EGTP_remat correctness: per-step loss trajectory EP=4 baseline vs EP=2+EGTP_remat=2
# ---------------------------------------------------------------------------


def _worker_moe_egtp_correctness(rank, world_size, port):
    """Verify EP=2+EGTP_remat=2 MoE matches per-step loss of EP=4 no-EGTP_remat baseline.

    Phase 1 — EP=4, EGTP_remat=1:
        All 4 ranks form one EP group; each rank holds 2 full expert weights (8 total).
        All ranks receive the same MoE-layer input; alltoall dispatch routes each token
        to its assigned expert rank, so each rank computes a different token subset.
        Gradients are local to each expert's rank.  Weight update:
            param.data -= lr * param.grad

    Phase 2 — EP=2, EGTP_remat=2:
        Two EP groups of 2 ranks, each EGTP_remat-sharded over 2 ranks.  Expert weights
        are sharded along dim 0 within each EGTP_remat group (shard = full_dim0 / egtp_remat_size).
        After backward, wgrad reduce-scatter sums each shard's identical wgrad:
            main_grad[rank_i] = egtp_remat_size * dW[shard_i]
        The optimizer divides by egtp_remat_size:
            param.data -= (lr / egtp_remat_size) * param.main_grad

    Weight sharing (test-only):
        To ensure both phases start from identical expert weights, an all-gather
        collects the full 8-expert table from the EP=4 group (where each rank holds
        only 2 experts) onto every rank.  Phase 2 then slices each rank's local
        experts and EGTP_remat shard from that global table.

    Nemotron3-Super Proxy MoE hyperparameters (scaled for unit-test speed):
        hidden=4096, ffn_hidden_size=2688, num_experts=8, topk=2
    MXFP8 alignment with EGTP_remat=2:
        2688/2=1344, 1344%16=0 (fc1 shard); 4096/2=2048, 2048%16=0 (fc2 shard)
    """
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Nemotron3-Super MoE hyperparameters (num_experts scaled from 512 to 8 for test speed).
    HIDDEN = 4096
    FFN_HIDDEN = 2688
    NUM_EXPERTS = 8
    TOPK = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16

    def make_config():
        return TransformerConfig(
            num_attention_heads=32,
            num_layers=1,
            hidden_size=HIDDEN,
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_ffn_hidden_size=FFN_HIDDEN,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_moe_layer(config, pg_collection):
        moe_spec = get_moe_module_spec(use_te=True, num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
        return moe_spec(config, layer_number=1, pg_collection=pg_collection)

    def run_step(layer, x):
        with fp8_autocast(enabled=False):
            output, _ = layer(x)
        return output.mean()

    # -------------------------------------------------------------------------
    # Phase 1: Baseline — EP=4, EGTP_remat=1 (DP=1)
    # -------------------------------------------------------------------------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_gtp_remat_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['ep'])
    ep_group = pg_collection.ep
    num_local_experts_baseline = NUM_EXPERTS // 4  # = 2

    config = make_config()
    layer = make_moe_layer(config, None)  # MoELayer uses get_default_pg_collection()
    layer.cuda()

    # Verify baseline has no GTP_remat sharding (EGTP_remat=1 should leave plain parameters).
    assert not any(
        isinstance(p, GTPShardedParam) for p in layer.parameters()
    ), "Baseline EP=4 layer should have no GTPShardedParam (EGTP_remat=1)"

    # Synchronize non-expert weights from rank 0; expert weights are rank-local.
    for name, p in layer.named_parameters():
        if 'linear_fc1.weight' not in name and 'linear_fc2.weight' not in name:
            dist.broadcast(p.data, src=0)

    # Collect the full expert weight table so Phase 2 can restore identical init weights.
    # EP=4: each rank holds 2 experts; all-gather gives every rank the complete [8, dim, ...] table.
    local_fc1 = torch.stack(
        [
            dict(layer.named_parameters())[f'experts.linear_fc1.weight{i}'].data
            for i in range(num_local_experts_baseline)
        ]
    )  # [2, FFN_HIDDEN, HIDDEN]
    global_fc1 = torch.zeros(NUM_EXPERTS, FFN_HIDDEN, HIDDEN, dtype=dtype, device='cuda')
    dist.all_gather_into_tensor(global_fc1, local_fc1, group=ep_group)

    local_fc2 = torch.stack(
        [
            dict(layer.named_parameters())[f'experts.linear_fc2.weight{i}'].data
            for i in range(num_local_experts_baseline)
        ]
    )  # [2, HIDDEN, FFN_HIDDEN]
    global_fc2 = torch.zeros(NUM_EXPERTS, HIDDEN, FFN_HIDDEN, dtype=dtype, device='cuda')
    dist.all_gather_into_tensor(global_fc2, local_fc2, group=ep_group)

    # Save non-expert param values (router, norms, etc.) from rank 0.
    non_expert_weights = {}
    for name, p in layer.named_parameters():
        if 'linear_fc1.weight' not in name and 'linear_fc2.weight' not in name:
            non_expert_weights[name] = p.data.clone()

    baseline_losses = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layer, x)
        if rank == 0:
            baseline_losses.append(loss.item())

        loss.backward()
        with torch.no_grad():
            for p in layer.parameters():
                if p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    FP8GlobalStateManager.reset()

    # -------------------------------------------------------------------------
    # Phase 2: EP=2, EGTP_remat=2 (DP=1 effective)
    # -------------------------------------------------------------------------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_gtp_remat_size=2,
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['expt_gtp_remat'])
    egtp_remat_group = pg_collection.expt_gtp_remat
    egtp_remat_size = egtp_remat_group.size()
    egtp_rank = egtp_remat_group.rank()
    ep_rank_egtp = dist.get_rank(ps.get_expert_model_parallel_group())
    num_local_experts_egtp = NUM_EXPERTS // 2  # = 4

    config = make_config()
    # Build full pg_collection for MoELayer: default groups + expt_gtp for EGTP_remat sharding.
    moe_pg = get_default_pg_collection()
    moe_pg.expt_gtp_remat = egtp_remat_group
    layer_egtp = make_moe_layer(config, moe_pg)
    layer_egtp.cuda()

    # Verify EGTP_remat is truly active: expert weight params must be GTPShardedParam instances.
    egtp_params = [p for p in layer_egtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(egtp_params) > 0, "EGTP_remat inactive: no GTPShardedParam in EP=2+EGTP_remat=2"

    # Restore weights from saved global tables.
    # Expert local index j → global expert id = ep_rank_egtp * num_local_experts_egtp + j.
    fc1_shard = FFN_HIDDEN // egtp_remat_size  # 2688/2 = 1344
    fc2_shard = HIDDEN // egtp_remat_size  # 4096/2 = 2048
    for name, p in layer_egtp.named_parameters():
        if 'linear_fc1.weight' in name:
            j = int(name.rsplit('weight', 1)[1])
            gid = ep_rank_egtp * num_local_experts_egtp + j
            p.data.copy_(global_fc1[gid, egtp_rank * fc1_shard : (egtp_rank + 1) * fc1_shard])
        elif 'linear_fc2.weight' in name:
            j = int(name.rsplit('weight', 1)[1])
            gid = ep_rank_egtp * num_local_experts_egtp + j
            p.data.copy_(global_fc2[gid, egtp_rank * fc2_shard : (egtp_rank + 1) * fc2_shard])
        elif name in non_expert_weights:
            p.data.copy_(non_expert_weights[name])

    # Pre-allocate main_grad for EGTP_remat params (required before the first backward).
    for p in layer_egtp.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    egtp_losses = []
    for step in range(STEPS):
        for p in layer_egtp.parameters():
            if isinstance(p, GTPShardedParam):
                p.main_grad.zero_()

        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layer_egtp, x)
        if rank == 0:
            egtp_losses.append(loss.item())

        loss.backward()

        # After RS, main_grad = egtp_remat_size * dW_shard. Divide by egtp_remat_size for baseline.
        with torch.no_grad():
            for p in layer_egtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / egtp_remat_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}

    # -------------------------------------------------------------------------
    # Compare per-step loss trajectories on rank 0
    # -------------------------------------------------------------------------
    if rank == 0:
        _assert_loss_trajectories_match(baseline_losses, egtp_losses, STEPS, label="egtp_remat")


def _worker_expert_bias_gtp_inclusive(rank, world_size, port):
    """Router expert-bias must stay identical across gtp_remat peers.

    The aux-loss-free balancer (``get_updated_expert_bias``) all-reduces per-expert token counts
    over the router's tp_dp_cp group, then sign-updates the replicated expert_bias. gtp_remat peers
    hold DISTINCT tokens but share the replicated router, so the reduction must span gtp_remat.
    ``get_tensor_and_data_parallel_group`` therefore spans gtp_remat (like dp); over a gtp-EXCLUDED
    group the peers reduce different token sums and the bias diverges -> routing instability
    (loss spikes). world=4 -> tp1 * gtp_remat2 * dp2.
    """
    import torch.distributed as dist

    from megatron.core import parallel_state as ps
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
    from megatron.core.utils import get_pg_size

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    gtp_group = ps.get_gtp_weight_remat_group()
    tp_dp_cp = ps.get_tensor_and_data_parallel_group(with_context_parallel=True)  # spans gtp_remat
    # gtp-EXCLUDED replicate group (explicit, since get_data_parallel_group now defaults inclusive).
    replicate = ps.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=False)

    num_experts = 8
    torch.manual_seed(1000 + rank)  # distinct per-rank tokens (distinct data on each rank)
    base = torch.randint(0, 100, (num_experts,), device="cuda").float()

    def _max_bias_diff_across_gtp(group):
        bias = torch.zeros(num_experts, device="cuda")  # identical start on every rank
        updated = get_updated_expert_bias(base.clone(), bias, 0.01, tp_dp_cp_group=group)
        buf = [torch.zeros_like(updated) for _ in range(gtp_group.size())]
        dist.all_gather(buf, updated, group=gtp_group)
        return max((buf[i] - buf[0]).abs().max().item() for i in range(gtp_group.size()))

    # tp_dp_cp must span gtp_remat (size = replicate_dp_cp x gtp_remat at tp=1).
    spans_gtp = get_pg_size(tp_dp_cp) == get_pg_size(replicate) * get_pg_size(gtp_group)
    diff_excluded = _max_bias_diff_across_gtp(replicate)
    diff_included = _max_bias_diff_across_gtp(tp_dp_cp)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()

    if rank == 0:
        assert spans_gtp, "tp_dp_cp group must span the gtp_remat axis (like dp)"
        # gtp-excluded reduction reproduces the bug; tp_dp_cp (spans gtp_remat) keeps peers in sync.
        assert (
            diff_excluded > 0
        ), f"gtp-excluded group should diverge across gtp_remat peers, got {diff_excluded}"
        assert (
            diff_included == 0
        ), f"tp_dp_cp must keep expert_bias identical across gtp_remat peers, got {diff_included}"


# ---------------------------------------------------------------------------
# EGTP_remat grouped experts kept unfused by a te-precision-config override
# ---------------------------------------------------------------------------

# The production matcher from te_quant_mxfp8.cfg / bf16_overrides.yaml, verbatim.
MTP_BF16_PATTERN = "*mtp.layers.*"
# A module path the matcher hits, so the override is resolved exactly as it is in production
# (by module path) without building a full MTP stack.
MTP_LAYER_NAME = "mtp.layers.0.mlp"


def _bf16_override_recipe():
    """The `mtp_bf16` matcher: force anything under `mtp.layers.` to high precision."""
    from megatron.core.quantization.quant_config import RecipeConfig

    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                }
            },
            "matchers": {
                "mtp_bf16": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": MTP_BF16_PATTERN,
                    "config": "bf16",
                }
            },
        }
    )


def _worker_egtp_precision_override_backward(rank, world_size, port, egtp_remat_size):
    """EGTP_remat-sharded grouped experts that a precision override keeps unfused.

    A ``--te-precision-config-file`` matcher that forces a grouped-expert layer to high
    precision (in production: ``pattern: "*mtp.layers.*"`` against an MXFP8 run) drives
    ``TEGroupedMLP._with_fused_impl`` to False, so those experts run on the stock
    ``te.pytorch.GroupedLinear`` module instead of the op fuser. That module is still
    constructed with ``use_grouped_tensor=True``, because ``use_transformer_engine_op_fuser``
    force-enables ``moe_use_grouped_tensor`` for the whole config.

    TE's grouped-tensor forward materializes a DistributedWeight into a plain gathered
    tensor before it builds the wgrad-accumulation closures, and -- unlike its
    split-quantize sibling, which branches on ``is_dist_weight`` and reads
    ``origin_weights[i].grad_buffer`` -- it closes over the materialized tensor:

        ctx.main_grad_funcs = [lambda j=i: weights[j].main_grad for i in range(len(weights))]

    so backward raises ``AttributeError: 'Tensor' object has no attribute 'main_grad'``.

    Both ingredients are required. At ``egtp_remat_size=1`` the expert weights are ordinary
    Parameters, nothing is materialized, and the same closure reads a real ``main_grad`` --
    which is why the benchmark scripts, which default to EGTP_remat=1, never hit this.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN_OVERRIDE = 512
    FFN_HIDDEN_OVERRIDE = 256
    NUM_EXPERTS_OVERRIDE = 4
    SEQ_OVERRIDE = 16
    BATCH_OVERRIDE = 1
    dtype = torch.bfloat16

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_gtp_remat_size=egtp_remat_size,
    )
    model_parallel_cuda_manual_seed(42)

    config = TransformerConfig(
        num_attention_heads=8,
        num_layers=1,
        hidden_size=HIDDEN_OVERRIDE,
        num_moe_experts=NUM_EXPERTS_OVERRIDE,
        moe_router_topk=2,
        moe_ffn_hidden_size=FFN_HIDDEN_OVERRIDE,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        params_dtype=dtype,
        bf16=True,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # Force-enables moe_use_grouped_tensor in TransformerConfig.__post_init__, which is
        # what leaves the *unfused* module on TE's grouped-tensor path.
        use_transformer_engine_op_fuser=True,
        # Required for TE to build main_grad_funcs at all; DDP normally sets this.
        gradient_accumulation_fusion=True,
        quant_recipe=_bf16_override_recipe(),
    )

    # MoELayer's default collection has no expt_gtp_remat entry; EGTP_remat sharding keys off it.
    egtp_remat_group = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['expt_gtp_remat']
    ).expt_gtp_remat
    assert egtp_remat_group.size() == egtp_remat_size
    moe_pg = get_default_pg_collection()
    moe_pg.expt_gtp_remat = egtp_remat_group

    moe_spec = get_moe_module_spec(
        use_te=True, num_experts=NUM_EXPERTS_OVERRIDE, moe_grouped_gemm=True
    )
    layer = moe_spec(config, layer_number=1, pg_collection=moe_pg, name=MTP_LAYER_NAME)
    layer.cuda()

    try:
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)

        # Ingredient 1: the precision override took these experts off the op-fuser path (#7212).
        assert not experts._with_fused_impl, (
            "the bf16 override must keep these experts unfused; without that this test exercises "
            "the fused path instead and cannot reproduce the bug"
        )
        # The config-level force-enable is in play for both cases: use_transformer_engine_op_fuser
        # turns moe_use_grouped_tensor on for every layer, unfused ones included.
        assert config.moe_use_grouped_tensor

        # Ingredient 2: EGTP_remat > 1 makes the expert weights TE DistributedWeights.
        sharded = [name for name, p in experts.named_parameters() if isinstance(p, GTPShardedParam)]
        if egtp_remat_size > 1:
            assert sharded, "EGTP_remat=2 must shard the grouped expert weights"
            # Only now may the guard pull the module off the grouped-tensor path, where TE
            # cannot accumulate wgrad into a sharded weight.
            assert not experts.linear_fc1.use_grouped_tensor
            assert not experts.linear_fc2.use_grouped_tensor
            assert not experts._use_grouped_tensor  # also selects the m_splits form passed to TE
        else:
            assert not sharded, "EGTP_remat=1 must leave the grouped expert weights unsharded"
            # Unfused but UNSHARDED is not the bug, and the grouped-tensor path is both faster
            # and CUDA-graph-safe (device-side m_splits). The guard must leave it alone -- this
            # is what stops it from silently disabling --moe-use-grouped-tensor at large.
            assert experts.linear_fc1.use_grouped_tensor
            assert experts.linear_fc2.use_grouped_tensor
            assert experts._use_grouped_tensor

        # DDP would allocate these; the wgrad closures are what the bug corrupts, so they must
        # exist for the backward below to mean anything.
        for p in experts.parameters():
            if p.requires_grad and not hasattr(p, "main_grad"):
                p.main_grad = torch.zeros_like(p.data, dtype=torch.float32)

        x = torch.randn(
            (SEQ_OVERRIDE, BATCH_OVERRIDE, HIDDEN_OVERRIDE),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        output, _ = layer(x)
        # Pre-fix, egtp_remat_size=2 raises AttributeError: 'Tensor' object has no attribute
        # 'main_grad' from grouped_linear.py::_backward_grouped_tensor.
        output.mean().backward()

        accumulated = [
            name
            for name, p in experts.named_parameters()
            if getattr(p, "main_grad", None) is not None and p.main_grad.abs().sum() > 0
        ]
        assert accumulated, "no expert wgrad reached main_grad; wgrad accumulation silently no-op'd"
    finally:
        # Restore in finally: an assertion above would otherwise leave the EGTP_remat groups
        # installed and break every later test that calls initialize_model_parallel.
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestMoEEGTPCorrectness:
    def test_moe_egtp_loss_trajectory_matches_baseline(self):
        """EP=2+EGTP_remat=2 MoE per-step losses match EP=4 baseline: atol=rtol=1e-5; MXFP8"""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_moe_egtp_correctness, 4)

    def test_expert_bias_gtp_inclusive(self):
        """expert_bias stays synced across gtp_remat peers only with the gtp-inclusive group."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_expert_bias_gtp_inclusive, 4)


class TestMoEEGTPPrecisionOverride:
    @pytest.mark.parametrize("egtp_remat_size", (1, 2))
    def test_egtp_precision_override_backward(self, monkeypatch, egtp_remat_size):
        """bf16 MTP override on grouped experts must survive backward.

        ``egtp_remat_size=1`` is the control: it is the benchmark scripts' default and passes
        today. ``egtp_remat_size=2`` is the repro.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        # _is_fused_impl_supported() requires this for the GLU fused kernel, and the production
        # launch script exports it. Without it every expert is unfused for the wrong reason and
        # the `_with_fused_impl` assertion would no longer prove the override did anything.
        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
        # world=4 to match the rest of this file: EP=1 x EGTP_remat={1,2}, DP takes the remainder.
        _run_distributed(_worker_egtp_precision_override_backward, 4, egtp_remat_size)
