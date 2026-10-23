# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-head dtype compatibility for retained GPT and Hybrid model APIs."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_model(kind, logit_dtype, **config_kwargs):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **config_kwargs,
    )
    kwargs = dict(
        config=config,
        vocab_size=128,
        max_sequence_length=8,
        logit_dtype=logit_dtype,
        position_embedding_type='none',
    )
    if kind == 'gpt':
        return GPTModel(
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(), **kwargs
        ).cuda()
    return HybridModel(
        hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern='*-', **kwargs
    ).cuda()


@pytest.mark.parametrize('kind', ['gpt', 'hybrid'])
@pytest.mark.parametrize('logit_dtype', [None, torch.float32], ids=['default', 'fp32'])
def test_model_logit_dtype_cross_entropy_backward(kind, logit_dtype):
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        model = _build_model(kind, logit_dtype)
        hidden = torch.randn(8, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        labels = torch.arange(8, device='cuda').reshape(1, 8)
        logits, _ = model.output_layer(hidden)
        assert logits.dtype == (logit_dtype or hidden.dtype)
        reference_logits = F.linear(
            hidden.detach().float(), model.output_layer.weight.detach().float()
        )
        torch.testing.assert_close(logits.float(), reference_logits, atol=0.005, rtol=0.005)
        reference_loss = F.cross_entropy(
            logits.detach().float().reshape(-1, 128), labels.reshape(-1), reduction='none'
        ).reshape(1, 8)
        loss = model.compute_language_model_loss(labels, logits)
        torch.testing.assert_close(loss, reference_loss, atol=1e-5, rtol=1e-5)
        loss.mean().backward()
        assert hidden.grad.dtype == torch.bfloat16
        assert torch.isfinite(hidden.grad).all()
        assert model.output_layer.weight.grad.dtype == torch.bfloat16
        assert torch.isfinite(model.output_layer.weight.grad).all()
    finally:
        Utils.destroy_model_parallel()


def test_gpt_logit_dtype_rejects_fused_linear_cross_entropy():
    Utils.initialize_model_parallel(1, 1)
    try:
        with pytest.raises(ValueError, match='logit_dtype is not supported with fused linear'):
            _build_model(
                'gpt',
                torch.float32,
                cross_entropy_loss_fusion=True,
                cross_entropy_fusion_impl='linear',
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('numerical', [False, True], ids=['constructor', 'numerical'])
def test_gpt_fused_linear_cross_entropy_preserves_default_dtype(numerical):
    if numerical:
        cute = pytest.importorskip('cutlass.cute')
        if not hasattr(cute, 'make_fragment'):
            pytest.skip('Retained fused-linear CE kernels require the CuTe make_fragment API')
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        model = _build_model(
            'gpt', None, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='linear'
        )
        assert model.fuse_linear_cross_entropy
        assert model.output_layer.output_dtype is None
        if not numerical:
            return
        hidden = torch.randn(8, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        labels = torch.arange(8, device='cuda').reshape(1, 8)
        reference_logits = F.linear(
            hidden.detach().float(), model.output_layer.weight.detach().float()
        )
        reference_loss = F.cross_entropy(
            reference_logits.reshape(-1, 128), labels.reshape(-1), reduction='none'
        ).reshape(1, 8)
        loss = model.output_layer(hidden, output_cross_entropy_loss=True, labels=labels)
        torch.testing.assert_close(loss, reference_loss, atol=0.005, rtol=0.005)
        loss.mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert model.output_layer.weight.grad is not None
        assert torch.isfinite(model.output_layer.weight.grad).all()
    finally:
        Utils.destroy_model_parallel()
