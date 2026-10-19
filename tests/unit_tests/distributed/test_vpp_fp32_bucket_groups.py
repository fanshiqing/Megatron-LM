# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("disable_bucketing", [False, True])
@pytest.mark.parametrize("fp32_accumulation", [False, True])
def test_vpp_mixed_dtype_buckets_reduce_all_gradients(disable_bucketing, fp32_accumulation):
    """A later VPP chunk must retain every async RS handle across dtype buffers."""
    Utils.initialize_model_parallel()
    try:
        module = torch.nn.Module()
        module.register_parameter(
            "bf16_weight", torch.nn.Parameter(torch.ones(256, device="cuda", dtype=torch.bfloat16))
        )
        module.register_parameter(
            "fp32_weight", torch.nn.Parameter(torch.ones(256, device="cuda", dtype=torch.float32))
        )
        config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            reduce_scatter_with_fp32_accumulation=fp32_accumulation,
        )
        group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        layout = DistributedOptimizer.compute_full_param_layout(
            list(module.parameters()), None, group.size(), config
        )
        ddp = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config=config,
            module=module,
            disable_bucketing=disable_bucketing,
            full_param_layout=layout,
        )
        assert len(ddp.buffers) == 2
        if fp32_accumulation or not disable_bucketing:
            assert all(len(bucket_group.buckets) == 1 for bucket_group in ddp.bucket_groups)
        else:
            assert len(ddp.bucket_groups) == 1
            assert len(ddp.bucket_groups[0].buckets) == 2
        for step in range(3):
            ddp.zero_grad_buffer()
            for buffer in ddp.buffers:
                buffer.grad_data.fill_(group.rank() + step + 1)
            ddp.start_grad_sync()
            ddp.finish_grad_sync()
            torch.cuda.synchronize()
            expected = step + (group.size() + 1) / 2
            for buffer in ddp.buffers:
                for bucket in buffer.buckets:
                    local = shard_buffer(bucket.grad_data, group.size())[group.rank()]
                    torch.testing.assert_close(
                        local, torch.full_like(local, expected), atol=0, rtol=0
                    )
    finally:
        Utils.destroy_model_parallel()
