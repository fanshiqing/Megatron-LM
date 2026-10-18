# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSV4-recipe hybrid Muon parity on four GPUs, using a small MXFP8 MoE model."""

import pytest
import torch

from tests.unit_tests.optimizer.test_layer_sharded_e2e_parity import _LayerShardedHarness
from tests.unit_tests.test_utilities import Utils


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("padded_layout", [False, True])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_hybrid_muon_mxfp8_matches_duplicated_experts(overlap, padded_layout, ep_size):
    """Changing only expert NS distribution preserves the NS10 loss trajectory.

    EP1 verifies semantic expert dispatch even when expert gradients use all-reduce.
    EP2 verifies expert/dense process-group separation. Both compact/padded layouts
    and synchronous/overlapped parameter synchronization exercise FP32-master staging.
    Dense weights use auto in both runs to isolate expert distribution from changes
    in dense NS collective arithmetic.
    """
    if Utils.world_size != 4:
        pytest.skip("Requires exactly four ranks")
    harness = _LayerShardedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    common = dict(
        tp_size=1,
        recipe="mxfp8",
        fp8_param_gather=True,
        num_steps=6,
        num_layers=2,
        padded_vocab_size=512,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        global_batch_size=4,
        optimizer="muon",
        muon_scalar_optimizer="adam",
        muon_momentum=0.95,
        muon_scale_mode="spectral",
        muon_num_ns_steps=10,
        muon_coefficient_type="deepseekv4",
        muon_tp_mode="auto",
        lr=2e-4,
        clip_grad=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_parallel_num_weight_shards=4,
        untie_embeddings_and_output_weights=True,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
        use_layer_wise_param_layout=padded_layout,
        num_experts=4,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_ffn_hidden_size=256,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        expert_tensor_parallel_num_weight_shards=4 // ep_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        swiglu=True,
    )
    try:
        reference = harness._run_test_helper(muon_expert_tp_mode="duplicated", **common)
        assert harness.layer_sharded_instances == 0
        actual = harness._run_test_helper(muon_expert_tp_mode="layer_sharded", **common)
        assert harness.layer_sharded_instances > 0
        assert harness.wired_groups > 0
        assert reference.numel() == actual.numel() == common["num_steps"]
        pairs = torch.stack((reference, actual)).cuda()
        gathered = [torch.empty_like(pairs) for _ in range(4)]
        torch.distributed.all_gather(gathered, pairs)
        for rank, pair in enumerate(gathered):
            assert torch.isfinite(pair).all(), f"Non-finite loss on rank {rank}"
            torch.testing.assert_close(pair[0], pair[1], atol=1e-4, rtol=1e-3)
        if torch.distributed.get_rank() == 0:
            print(
                f"Hybrid NS10 MXFP8 parity: EP{ep_size}, padded={padded_layout}, "
                f"overlap={overlap}; losses={actual.tolist()}"
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)
