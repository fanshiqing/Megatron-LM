# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the GTP VMM allocator's runtime extension build."""

from pathlib import Path
from unittest.mock import Mock

import pytest

import megatron.core.allocator.vmm_symm_allocator as vmm_alloc


@pytest.fixture(autouse=True)
def reset_build_state(monkeypatch):
    """Isolate the allocator build cache."""
    monkeypatch.setattr(vmm_alloc, "_allocator", None)
    monkeypatch.setattr(vmm_alloc, "_build_error", None)


def test_build_uses_packaged_source(monkeypatch, tmp_path):
    """Avoid source writes before PyTorch's build lock and cache the allocator."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    allocator = object()
    extension = Mock(get_vmm_allocator=Mock(return_value=allocator))
    load = Mock(return_value=extension)
    monkeypatch.setattr(vmm_alloc.torch.utils.cpp_extension, "load", load)
    monkeypatch.setattr(
        vmm_alloc.torch.utils.cpp_extension,
        "load_inline",
        lambda **_kwargs: pytest.fail("load_inline writes main.cpp before taking its build lock"),
    )

    vmm_alloc._build_vmm_allocator()
    vmm_alloc._build_vmm_allocator()

    assert vmm_alloc._allocator is allocator
    load.assert_called_once()
    extension.get_vmm_allocator.assert_called_once_with()
    kwargs = load.call_args.kwargs
    assert kwargs["build_directory"] == str(tmp_path / "vmm_symm_allocator")
    source_path = Path(vmm_alloc.__file__).parent / "csrc" / "vmm_symm_allocator.cpp"
    assert kwargs["sources"] == [str(source_path)]
    assert source_path.is_file()
    assert not source_path.is_relative_to(Path(kwargs["build_directory"]))
    assert kwargs["with_cuda"] is True
    assert kwargs["extra_ldflags"] == ["-lcuda"]
    assert kwargs["is_python_module"] is True
