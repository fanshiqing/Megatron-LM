# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Storage lifetime regressions for eager/full-iteration GTP buffers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _torchrun_dist_init


@pytest.fixture(autouse=True)
def allocator_mode(monkeypatch):
    monkeypatch.setattr(gtp, "_FULL_ITERATION", True)
    monkeypatch.setattr(gtp, "_CUDA_GRAPH_MODULES", None)


def make_slot(buffer=None, *, reduce_scatter=False):
    param = SimpleNamespace(
        chain_id=gtp.GTPChain.UNGRAPHED.value,
        group=object(),
        _prefetch_handle=None,
        _already_ag_drained=False,
        _recompute_prefetch_handle=None,
        _recompute_already_drained=False,
        _wgrad_rs_handle=None,
        _ag_ticket_fwd=0,
        _ag_ticket_bwd=0,
        _ag_ticket_recompute=0,
    )
    param._weights = [param]
    slot = SimpleNamespace(
        param=param,
        buf=buffer,
        reduce_scatter=reduce_scatter,
        chain_id=param.chain_id,
        key=("test",),
    )
    cache = gtp.GTPWeightCache()
    cache._slots[0] = slot
    return cache, slot, param


@pytest.mark.parametrize("phase", ["fwd", "bwd", "recompute"])
def test_ag_release_waits_until_next_consumer(phase):
    buffer = torch.ones(256, device="cuda")
    cache, slot, param = make_slot(buffer)
    cache.release(0)
    assert slot.buf is buffer and buffer.untyped_storage().nbytes() > 0
    cache.end_ag_consume(param, phase)
    # A different phase must not release a live materialization.
    cache.begin_ag_consume(param, "bwd" if phase != "bwd" else "fwd")
    assert slot.buf is buffer
    consumed = buffer.sum()
    cache.begin_ag_consume(param, phase)
    assert slot.buf is None and buffer.untyped_storage().nbytes() == 0
    torch.testing.assert_close(consumed, torch.tensor(256.0, device="cuda"))


@pytest.mark.parametrize("new_materialization", [False, True])
def test_ag_release_preserves_pending_prefetch(new_materialization):
    buffer = torch.ones(32, device="cuda")
    cache, slot, param = make_slot(buffer)
    cache.end_ag_consume(param, "fwd")
    if new_materialization:
        slot.buf = torch.zeros_like(buffer)
    else:
        param._prefetch_handle = object()
    current = slot.buf
    cache.begin_ag_consume(param, "fwd")
    assert slot.buf is current and current.untyped_storage().nbytes() > 0


def test_partial_graphs_keep_existing_buffer_policy(monkeypatch):
    monkeypatch.setattr(gtp, "_FULL_ITERATION", False)
    monkeypatch.setattr(gtp, "_CUDA_GRAPH_MODULES", {"attn"})
    buffer = torch.ones(32, device="cuda")
    cache, slot, param = make_slot(buffer)
    cache.end_ag_consume(param, "fwd")
    cache.begin_ag_consume(param, "fwd")
    cache.release(0)
    assert slot.buf is buffer and cache._pool[slot.key][0] is buffer
    assert not cache._last_ag_consumer


@pytest.mark.parametrize("captured", [False, True])
def test_rs_release_preserves_queued_accumulation(captured):
    result = torch.zeros((), device="cuda")
    cache, slot, _ = make_slot(reduce_scatter=True)

    def accumulate():
        buffer = torch.full((1024,), 3.0, device="cuda")
        slot.buf = buffer
        result.copy_(buffer.sum())
        cache.release(0)
        assert slot.buf is None and buffer.untyped_storage().nbytes() == 0
        torch.empty_like(buffer).fill_(-100)

    accumulate()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            accumulate()
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.tensor(3072.0, device="cuda"))


@pytest.mark.parametrize(
    "pending",
    [
        "_wgrad_rs_handle",
        "_prefetch_handle",
        "_already_ag_drained",
        "_recompute_prefetch_handle",
        "_recompute_already_drained",
    ],
)
def test_capture_reset_rejects_pending_work_atomically(pending):
    cache, first, _ = make_slot(object())
    _, second, param = make_slot(object())
    cache._slots[1] = second
    before = [first.buf, second.buf]
    setattr(param, pending, True)
    cache._pool["sentinel"] = [object()]
    with pytest.raises(AssertionError):
        cache.prepare_full_graph_capture()
    assert first.buf is before[0] and second.buf is before[1]
    assert cache._pool["sentinel"]


def test_capture_reset_clears_completed_references(monkeypatch):
    cache, slot, _ = make_slot(object())
    cache._pool["sentinel"] = [object()]
    cache._last_ag_consumer["sentinel"] = object()
    monkeypatch.setattr(gtp, "_FULL_ITERATION", False)
    with pytest.raises(RuntimeError):
        cache.prepare_full_graph_capture()
    assert slot.buf is not None
    monkeypatch.setattr(gtp, "_FULL_ITERATION", True)
    cache.prepare_full_graph_capture()
    assert slot.buf is None and not cache._pool and not cache._last_ag_consumer
