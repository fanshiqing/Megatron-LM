# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSV4 KV exclusion: construction and full-DP gradient correctness with outer DP > 1."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import reset_gtp_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_attention,
    _make_config,
    te_pytorch,
    te_recipe,
)


@pytest.fixture(autouse=True)
def parallel_groups():
    Utils.initialize_model_parallel(gtp_remat_size=2)
    if dist.get_world_size() != 4:
        Utils.destroy_model_parallel()
        pytest.skip("Run with four ranks: GTP2 x outer DP2")
    model_parallel_cuda_manual_seed(42)
    yield
    reset_gtp_state()
    Utils.destroy_model_parallel()


def make_attention(replicate, **kwargs):
    config = _make_config(
        tensor_parallel_num_weight_shards=2,
        gtp_remat_opt_out_modules=["dsv4_kv_proj"] if replicate else [],
        **kwargs,
    )
    config.use_cpu_initialization = False
    pg = ProcessGroupCollection.use_mpu_process_groups()
    attention = _build_attention(config, layer_number=1, pg_collection=pg).cuda()
    return config, pg, attention


@pytest.mark.parametrize("replicate", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_only_kv_projection_is_excluded(replicate, quantized):
    context = (
        te_pytorch.quantized_model_init(enabled=True, recipe=te_recipe.MXFP8BlockScaling())
        if quantized
        else nullcontext()
    )
    with context:
        _, pg, attention = make_attention(replicate)
    kv = attention.linear_kv_proj.weight
    assert bool(getattr(kv, "is_gtp_weight_remat", False)) is (not replicate)
    assert attention.linear_q_up_proj.weight.is_gtp_weight_remat
    assert attention.linear_proj.weight.is_gtp_weight_remat
    assert pg.gtp_remat.size() == 2
    if replicate:
        assert tuple(kv.shape) == (64, 256)
        assert getattr(kv, "qkv_layout", None) is None


@pytest.mark.parametrize("per_token", [False, True])
def test_replicated_kv_gradient_matches_full_dp(per_token):
    config, pg, attention = make_attention(True, calculate_per_token_loss=per_token)
    config.gradient_accumulation_fusion = True
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False, grad_reduce_in_fp32=True
        ),
        attention.linear_kv_proj,
        pg_collection=pg,
    )
    ddp.zero_grad_buffer()
    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(16, 1, 256, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(16, 1, 64, dtype=torch.bfloat16, device="cuda")
    y, _ = ddp(x)
    (y.float() * dy.float()).sum().backward()
    expected = dy.reshape(-1, 64).float().T @ x.reshape(-1, 256).float()
    dist.all_reduce(expected)
    # Per-token normalization counts all ranks' distinct tokens. Otherwise the
    # GTP and outer-DP means must together match a full-world mean.
    denominator = dist.get_world_size() * (16 if per_token else 1)
    expected /= denominator
    finalize_model_grads(
        [ddp],
        num_tokens=torch.tensor(16, dtype=torch.int64, device="cuda") if per_token else None,
        pg_collection=pg,
    )
    actual = ddp.module.weight.main_grad.float()
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)
    # A scale-sensitive SGD step catches errors that Adam can normalize away.
    before = ddp.module.weight.detach().float().clone()
    actual_update = before - 0.01 * actual
    expected_update = before - 0.01 * expected
    torch.testing.assert_close(actual_update, expected_update, rtol=0, atol=0.0003)


def test_unknown_exclusion_is_rejected():
    with pytest.raises(ValueError, match="gtp_remat_opt_out_modules"):
        _make_config(gtp_remat_opt_out_modules=["typo"])


def test_cli_exposes_opt_out(monkeypatch):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr("sys.argv", ["test", "--gtp-remat-opt-out-modules", "dsv4_kv_proj"])
    args = parse_args()
    assert args.gtp_remat_opt_out_modules == ["dsv4_kv_proj"]
