# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness with local partial CUDA graphs.

This is the CUDA-graph counterpart of ``test_gtp_loss_correctness.py``. It
compares a GTP=4 run using the local ``attn`` CUDA-graph scope against the same
GTP=1 eager baseline and verifies the complete per-step loss trajectory.
"""

import copy
import gc

import pytest
import torch
import torch.distributed as dist

from megatron.experimental.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

import megatron.experimental.gtp.generalized_tensor_parallelism as gtp_module
from megatron.experimental.gtp import (
    GTPChain,
    GTPShardedParam,
    classify_gtp_chains,
    reset_gtp_state,
    set_cuda_graph_modules,
    tag_gtp_params_with_names,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _worker_gtp_partial_cg_loss_correctness(rank, world_size, port):
    """Baseline (GTP=1, eager) vs GTP=4 with local ``attn`` CUDA graphs."""
    del world_size, port

    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.cuda_graphs import (
        _CudagraphGlobalRecord,
        create_cudagraphs,
        delete_cuda_graphs,
    )
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN = 4096
    NUM_HEADS = 32
    FFN_HIDDEN = 16384
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()

    def make_config(*, partial_cg=False):
        return TransformerConfig(
            num_attention_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            ffn_hidden_size=FFN_HIDDEN,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            fp8='e4m3',
            fp8_recipe='mxfp8',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_weight_remat_size=4 if partial_cg else 1,
            cuda_graph_impl="local" if partial_cg else "none",
            cuda_graph_modules=["attn"] if partial_cg else [],
            cuda_graph_warmup_steps=2,
        )

    def make_transformer_stack(config, pg_collection):
        # Partial ``attn`` CG captures attention layers. Remove the dense MLP,
        # matching the attention-only layer used by the hybrid model.
        spec = copy.deepcopy(get_gpt_layer_with_transformer_engine_spec())
        spec.submodules.pre_mlp_layernorm = IdentityOp
        spec.submodules.mlp = IdentityOp
        spec.submodules.mlp_bda = IdentityFuncOp
        return torch.nn.ModuleList(
            [
                spec.module(
                    config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection
                )
                for i in range(NUM_LAYERS)
            ]
        )

    def run_step(layers, x):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
        return x.mean()

    # ---- Phase 1: existing eager baseline, GTP=1 (DP=4) ----
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers = make_transformer_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()
    for param in layers.parameters():
        dist.broadcast(param.data, src=0)
    saved_weights = {name: param.data.clone() for name, param in layers.named_parameters()}

    baseline_losses = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)
        x.requires_grad_()
        loss = run_step(layers, x)
        baseline_losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for param in layers.parameters():
                if param.grad is not None:
                    param.data.sub_(LR * param.grad)
                    param.grad.zero_()

    ps.destroy_model_parallel()
    reset_gtp_state()
    FP8GlobalStateManager.reset()

    # ---- Phase 2: GTP=4 with local partial CUDA graphs ----
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=4,
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config(partial_cg=True)
    set_cuda_graph_modules({'attn'}, cuda_graph_impl=config.cuda_graph_impl)
    reset_gtp_state()
    layers_gtp = make_transformer_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()
    tag_gtp_params_with_names(layers_gtp)
    classify_gtp_chains(layers_gtp)

    gtp_group = ps.get_gtp_weight_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    assert gtp_size == 4, f"GTP shard group size should be 4, got {gtp_size}"
    gtp_params = [param for param in layers_gtp.parameters() if isinstance(param, GTPShardedParam)]
    assert gtp_params, "GTP not active: no GTPShardedParam found"
    assert all(param.chain_id == GTPChain.GRAPHED.value for param in gtp_params)

    for name, param in layers_gtp.named_parameters():
        full = saved_weights[name]
        if isinstance(param, GTPShardedParam):
            shard_size = param.shape[0]
            param.data.copy_(full[gtp_rank * shard_size : (gtp_rank + 1) * shard_size])
        else:
            param.data.copy_(full)

    for param in gtp_params:
        param.main_grad = torch.zeros(param.shape, dtype=dtype, device='cuda')

    original_cross_cg_overlap = gtp_module.GTP_CONFIG.cross_cg_overlap
    gtp_module.GTP_CONFIG.cross_cg_overlap = True
    gtp_losses = []
    try:
        for step in range(STEPS):
            for param in gtp_params:
                param.main_grad.zero_()
            torch.manual_seed(step)
            x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
            dist.broadcast(x, src=0)
            x.requires_grad_()
            loss = run_step(layers_gtp, x)
            gtp_losses.append(loss.item())
            loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()

            # The first eager pass records the per-layer runners. Capture them only
            # after the gradients are complete, matching the training schedule.
            if step == 0:
                with fp8_autocast(enabled=True, fp8_recipe=recipe):
                    create_cudagraphs()
                assert _CudagraphGlobalRecord.cudagraph_created
                assert all(
                    layer.cudagraph_manager.cudagraph_runners[0].gtp_remat
                    for layer in layers_gtp
                )

            with torch.no_grad():
                for param in layers_gtp.parameters():
                    if isinstance(param, GTPShardedParam):
                        param.data.sub_((LR / gtp_size) * param.main_grad)
                    elif param.grad is not None:
                        param.data.sub_(LR * param.grad)
                        param.grad.zero_()
        del loss, x
    finally:
        torch.cuda.synchronize()
        for layer in layers_gtp:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if runner.fwd_graph is not None:
                    runner.fwd_graph.reset()
                if runner.bwd_graph is not None:
                    runner.bwd_graph.reset()
        delete_cuda_graphs()
        for layer in layers_gtp:
            layer.cudagraph_manager.cudagraph_runners.clear()
        gc.collect()
        gtp_module.GTP_CONFIG.cross_cg_overlap = original_cross_cg_overlap
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        reset_gtp_state()

    assert len(baseline_losses) == STEPS and len(gtp_losses) == STEPS
    loss_error = torch.tensor(
        max(
            abs(gtp_loss - baseline_loss)
            for gtp_loss, baseline_loss in zip(gtp_losses, baseline_losses)
        ),
        device='cuda',
    )
    dist.all_reduce(loss_error, op=dist.ReduceOp.MAX)
    # MXFP8 graph capture/replay can introduce low-precision rounding drift.
    assert loss_error.item() <= 1e-4, f"Maximum loss error was {loss_error.item():.6g}"
    if rank == 0:
        for step, (baseline_loss, gtp_loss) in enumerate(zip(baseline_losses, gtp_losses)):
            print(
                f"Step {step:2d}: baseline={baseline_loss:.6f}  partial_cg={gtp_loss:.6f}",
                flush=True,
            )
        print(f"Maximum loss error across ranks: {loss_error.item():.6g}", flush=True)


class TestGTPPartialCGCorrectness:
    def test_gtp_partial_cg_loss_trajectory_matches_baseline(self):
        """GTP partial-CG losses must match the eager no-GTP baseline."""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_partial_cg_loss_correctness, 4)
