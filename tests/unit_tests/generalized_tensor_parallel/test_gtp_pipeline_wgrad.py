# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A pipeline gradient-sync boundary must see the complete shared-head gradient."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.mark.parametrize('pipeline_size', [1, 2, 4])
def test_head_gradient_ready_at_pipeline_sync(pipeline_size, monkeypatch):
    group = torch.distributed.group.WORLD
    monkeypatch.setattr(gtp.GTP_CONFIG, 'async_reduction', False)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'weight_prefetch', False)
    model = torch.nn.Module()
    model.config = SimpleNamespace(pipeline_model_parallel_size=pipeline_size)
    model.output_layer = torch.nn.Linear(64, 64, bias=False, device='cuda', dtype=torch.bfloat16)
    gtp.wrap_module_params_gtp(model.output_layer, ['weight'], group)
    weight = model.output_layer.weight
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    gtp.classify_gtp_remat_chains(model)
    try:
        for _ in range(3):
            weight.main_grad.zero_()
            weight.grad = None
            for value in [1, 2]:
                x = torch.full(
                    (16, 1, 64), value, device='cuda', dtype=torch.bfloat16, requires_grad=True
                )
                y = linear_with_grad_accumulation_and_async_allreduce(
                    x,
                    weight,
                    None,
                    False,
                    False,
                    False,
                    tp_group=group,
                    gtp_remat_size=group.size(),
                )
                y.sum().backward()
            # The aligned pipeline schedule can dispatch DDP here, before the global fence.
            if pipeline_size > 1:
                torch.testing.assert_close(
                    weight.main_grad, torch.full_like(weight.main_grad, 48), rtol=0, atol=0
                )
            gtp.wait_for_gtp_grad_reduction_on_current_stream()
            torch.testing.assert_close(
                weight.main_grad, torch.full_like(weight.main_grad, 48), rtol=0, atol=0
            )
    finally:
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
