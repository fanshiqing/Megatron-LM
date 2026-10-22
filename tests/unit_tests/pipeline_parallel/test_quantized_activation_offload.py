# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Quantized activation offload must preserve storage, metadata, and graph replay."""

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    OffloadTensorPool,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("transformer_engine.pytorch")
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("use_pool", [False, True])
def test_mxfp8_activation_offload_roundtrip(columnwise, use_pool):
    """Keep exact bytes/scales and source ownership through eager and captured copies."""
    Utils.initialize_distributed()
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 quantization requires Blackwell or newer")
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True)
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
    source = quantizer(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def check(actual):
        assert isinstance(actual, MXFP8Tensor)
        assert actual.dtype == source.dtype and actual.shape == source.shape
        for name in (
            "_rowwise_data",
            "_columnwise_data",
            "_rowwise_scale_inv",
            "_columnwise_scale_inv",
        ):
            expected_buf, actual_buf = getattr(source, name), getattr(actual, name)
            if expected_buf is None:
                assert actual_buf is None
            else:
                assert actual_buf is not None
                torch.testing.assert_close(actual_buf, expected_buf, rtol=0, atol=0)
        torch.testing.assert_close(actual.dequantize(), source.dequantize(), rtol=0, atol=0)

    graph = None
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = handler.reload(handler.offload(source, use_cpu_pool=use_pool))
        torch.cuda.synchronize()
        check(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = handler.reload(handler.offload(source, use_cpu_pool=use_pool))
        for step in range(6):
            x.normal_().mul_(step + 1)
            quantizer.update_quantized(x, source)
            graph.replay()
            torch.cuda.synchronize()
            check(out)
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        handler.cpu_tensor_pool.clear()
