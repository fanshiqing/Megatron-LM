# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP symmetric-memory pools (Layer 1: the shared registration primitive).

One ``ncclMemAlloc``-backed ``torch.cuda.MemPool`` per GTP process group,
registered once with ``backend.register_mem_pool(pool, symm=True)``. PyTorch's
ProcessGroupNCCL segment hook then auto-registers every subsequent allocation
made under ``gtp_mem_pool_ctx(group)`` via
``ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)``. With both ends of a
collective in such a pool on the same group's comm, NCCL selects its symmetric /
NVLS kernels.

This module owns *only* the pool + registration concern. Higher layers ride on
``gtp_mem_pool_ctx``:
  - persistent ticket cache (``GTPWeightCache``) -> AG output buffers,
  - a transient LIFO cache -> RS send buffers.

A single gate per param class (dense vs expert) controls symmetric memory; it is
not split by collective (AG and RS share the gate):
  - ``ENABLE_GTP_SYMM``  (default 1): dense (non-expert) GTP params.
  - ``ENABLE_EGTP_SYMM`` (default 0): expert / routed-expert (EGTP) params.

NCCL env this needs (launcher concern, not set here):
  NCCL_NVLS_ENABLE=1, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0
(both before init_process_group), plus ``--use-nccl-ub`` so the param-buffer
(AG input) side is registered too.
"""

import logging
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# Per-class symmetric-memory gates (one gate each; AG and RS share it).
ENABLE_GTP_SYMM = os.environ.get("ENABLE_GTP_SYMM", "1") == "1"
ENABLE_EGTP_SYMM = os.environ.get("ENABLE_EGTP_SYMM", "0") == "1"


def gtp_symm_eligible(is_expert: bool) -> bool:
    """Whether a param/buffer of this class should use the GTP symm-mem pool."""
    return ENABLE_EGTP_SYMM if is_expert else ENABLE_GTP_SYMM


# group.group_name -> per-group MemPool (one pool per group, registered once).
_pools: "dict[str, torch.cuda.MemPool]" = {}
# group.group_name of groups whose pool has been registered.
_registered: "set[str]" = set()


def get_gtp_pool(group) -> "torch.cuda.MemPool":
    """Return the per-group ``ncclMemAlloc``-backed MemPool, creating it once."""
    name = group.group_name
    pool = _pools.get(name)
    if pool is None:
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        _pools[name] = pool
    return pool


def register_gtp_pool(group) -> "torch.cuda.MemPool":
    """Create (if needed) and register the per-group pool on ``group``. Idempotent.

    Call once at model-construction time (before any CUDA-graph capture or
    forward), because it issues a collective (comm warmup). After this, the
    segment hook auto-registers future segments, so ``gtp_mem_pool_ctx`` itself
    is capture-safe and collective-free.
    """
    pool = get_gtp_pool(group)
    if group.group_name in _registered:
        return pool
    # Warm the group's comm so register_mem_pool sees an initialized
    # communicator (NCCL comms are created lazily on first collective).
    warmup = torch.zeros(1, device=torch.cuda.current_device())
    dist.all_reduce(warmup, group=group)
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)
    _registered.add(group.group_name)
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][GTP] Registered GTP cache pool on group {group.group_name} "
        f"(size={group.size()})",
    )
    return pool


def gtp_mem_pool_ctx(group):
    """Context manager: allocations inside land in ``group``'s registered pool.

    Pure ``use_mem_pool`` -- no collective -- so it is safe inside CUDA-graph
    capture. The pool must already be registered via ``register_gtp_pool`` for
    new segments to be window-registered by the segment hook.
    """
    return torch.cuda.use_mem_pool(get_gtp_pool(group))


class RegisteredLifoPool:
    """Group-aware LIFO cache of buffers in the per-group symmetric pool.

    Layer 2b of the symm stack: a transient pool for reduce-scatter *send*
    buffers (the full-shape wgrad the bwd GEMM writes, then scatters). Fresh
    allocations go through ``gtp_mem_pool_ctx`` so they are window-registered;
    freed buffers are recycled (keyed by numel + dtype + group), which keeps
    memory flat instead of one persistent buffer per weight.

    **No ``max_live`` constant.** The steady-state RS concurrency is reached
    organically during the eager warmup iterations (which run the same RS
    overlap as the captured steps): each ``alloc`` beyond the free-list does a
    fresh symm-pool allocation, ``free`` returns it, so by capture time the
    free-list already holds exactly the peak number of registered buffers.
    During capture ``alloc`` therefore only ever pops. The single failure mode
    -- a fresh allocation *during* capture (which would be illegal and would
    re-register the pool mid-graph) -- is turned into a clear error via
    ``torch.cuda.is_current_stream_capturing()``, with no magic number to tune.

    Storage is 1-D so one key serves any shape with that numel; ``alloc``
    returns a view tagged with ``_gtp_symm_group`` for tag-based recycling.
    """

    def __init__(self):
        # (numel, dtype, group_name) -> list of free 1-D buffers.
        self._free: "dict[tuple, list]" = defaultdict(list)

    def alloc(self, shape, dtype, device, group) -> "torch.Tensor":
        numel = int(math.prod(shape))
        bucket = self._free[(numel, dtype, group.group_name)]
        if bucket:
            flat = bucket.pop()
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "[GTP] RegisteredLifoPool exhausted during CUDA-graph capture "
                    f"(group={group.group_name}, numel={numel}, dtype={dtype}). The "
                    "eager warmup did not pre-populate enough RS send buffers for "
                    "the reduce-scatter overlap depth -- run more warmup iters, or "
                    "the RS concurrency changed between warmup and capture."
                )
            with gtp_mem_pool_ctx(group):
                flat = torch.empty(numel, dtype=dtype, device=device)
        out = flat.view(shape)
        out._gtp_symm_group = group  # tag so callers can recycle via tag dispatch
        return out

    def free(self, buf: "torch.Tensor") -> None:
        group = getattr(buf, "_gtp_symm_group", None)
        if group is None:
            return
        self._free[(buf.numel(), buf.dtype, group.group_name)].append(buf.reshape(-1))

    def clear(self) -> None:
        self._free.clear()
