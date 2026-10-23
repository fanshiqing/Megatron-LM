# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility entry point for the retained development DSV4 attention implementation.

The benchmark branch keeps its CSA/mHC layout and constructor APIs. Route the
upstream module-spec entry point through the matching development builder until
that architecture is migrated as a separately validated change.
"""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsv4_hybrid_module_spec_for_backend,
)
