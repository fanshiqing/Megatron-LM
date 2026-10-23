# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility of main runtime consumers with the dev args-owned training loop."""

from types import SimpleNamespace

import pytest

from megatron.training.global_vars import _LegacyTrainState


def test_progress_is_live_in_both_directions_without_changing_resume_baseline():
    args = SimpleNamespace(
        iteration=20, consumed_train_samples=5120, num_floating_point_operations_so_far=1000
    )
    state = _LegacyTrainState(args)
    args.consumed_train_samples += 256
    assert state.consumed_train_samples == 5376
    state.consumed_train_samples += 256
    assert args.consumed_train_samples == 5632
    state.iteration += 1
    state.num_floating_point_operations_so_far += 100
    assert args._current_train_iteration == 21
    assert args._current_floating_point_operations == 1100
    assert state.resume_iteration == args.iteration == 20
    assert state.resume_num_floating_point_operations == 1000


def test_state_checkpoint_roundtrip_updates_live_arguments():
    original = _LegacyTrainState(SimpleNamespace(iteration=4, consumed_train_samples=1024))
    original.iteration = 7
    original.num_floating_point_operations_so_far = 10**20
    args = SimpleNamespace(iteration=0)
    restored = _LegacyTrainState(args)
    restored.load_state_dict(original.state_dict())
    assert args._current_train_iteration == 7
    assert args.consumed_train_samples == 1024
    assert restored.num_floating_point_operations_so_far == 10**20
    restored.skipped_train_samples = 256
    assert args.skipped_train_samples == 256


def test_parse_only_registration_initializes_runtime_once_and_resets(monkeypatch):
    """Registering CLI inputs must not suppress timer/tokenizer/microbatch setup."""
    from unittest.mock import Mock

    from megatron.training import global_vars, initialize

    for name, value in vars(global_vars).copy().items():
        if name.startswith('_GLOBAL_'):
            monkeypatch.setattr(global_vars, name, value)
    monkeypatch.setattr(global_vars, '_GLOBAL_ARGS', None)
    monkeypatch.setattr(global_vars, '_GLOBAL_RUNTIME_INITIALIZED', False)
    monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', None)
    monkeypatch.setattr(global_vars, '_GLOBAL_TRAIN_STATE', None)
    builders = {}
    for name in (
        'init_num_microbatches_calculator',
        'unset_num_microbatches_calculator',
        '_build_tokenizer',
        '_set_tensorboard_writer',
        '_set_wandb_writer',
        '_set_one_logger',
        '_set_adlr_autoresume',
        '_set_timers',
        '_set_energy_monitor',
    ):
        builders[name] = Mock()
        monkeypatch.setattr(global_vars, name, builders[name])
    monkeypatch.setattr(initialize, 'setup_logging', Mock())
    monkeypatch.setattr(initialize, 'initialize_rerun_state_machine', Mock())
    monkeypatch.setattr(initialize, 'RerunMode', lambda value: value)
    monkeypatch.setattr(initialize, 'RerunDiagnostic', lambda value: value)
    monkeypatch.setattr(initialize, 'RerunErrorInjector', lambda **kwargs: None)
    args = SimpleNamespace(
        rank=0,
        global_batch_size=64,
        micro_batch_size=2,
        data_parallel_size=4,
        gtp_weight_remat_size=2,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=4096,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
        async_save=False,
        rerun_mode='disabled',
        error_injection_rate=0,
        error_injection_type='none',
        result_rejected_tracker_filename=None,
        batch_invariant_mode=False,
    )
    global_vars.set_args(args)
    cfg = SimpleNamespace(model=None, tokenizer=SimpleNamespace())
    global_vars.set_run_config(cfg)
    state = global_vars.get_train_state()
    assert not global_vars.is_global_variables_initialized()

    initialize.initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    initialize.initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    assert global_vars.is_global_variables_initialized()
    assert global_vars.get_run_config() is cfg
    assert global_vars.get_train_state() is state
    for name, builder in builders.items():
        if name != 'unset_num_microbatches_calculator':
            builder.assert_called_once()
    assert builders['init_num_microbatches_calculator'].call_args.kwargs['data_parallel_size'] == 8
    with pytest.raises(AssertionError, match='already initialized'):
        global_vars.set_global_variables(args)

    global_vars.unset_global_variables()
    assert not global_vars.is_global_variables_initialized()
    assert global_vars.get_train_state() is None
    builders['unset_num_microbatches_calculator'].assert_called_once()


@pytest.mark.parametrize('model_kind', ['gpt', 'hybrid'])
@pytest.mark.parametrize('explicit_vocab', [None, 32001])
def test_runtime_vocab_binding_preserves_explicit_model_config(
    monkeypatch, model_kind, explicit_vocab
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import global_vars
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    model_cls = GPTModelConfig if model_kind == 'gpt' else HybridModelConfig
    model = model_cls(
        transformer=TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4),
        vocab_size=explicit_vocab,
        should_pad_vocab=True,
    )
    cfg = SimpleNamespace(model=model, tokenizer=SimpleNamespace())
    monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', cfg)
    global_vars._bind_tokenizer_vocab_to_run_config(SimpleNamespace(padded_vocab_size=32256))
    assert cfg.tokenizer.padded_vocab_size == 32256
    assert model.vocab_size == (32256 if explicit_vocab is None else explicit_vocab)
    assert model.should_pad_vocab is (explicit_vocab is not None)


@pytest.mark.parametrize('sample_based', [False, True])
@pytest.mark.parametrize('fractional_warmup', [False, True])
def test_runtime_finalization_preserves_scheduler_units_and_batch_defaults(
    monkeypatch, sample_based, fractional_warmup
):
    _check_runtime_scheduler(monkeypatch, sample_based, fractional_warmup, batch_schedule=None)


def test_runtime_finalization_supports_sample_based_batch_schedule(monkeypatch):
    _check_runtime_scheduler(monkeypatch, True, False, batch_schedule='0:16 64:32')


def _check_runtime_scheduler(monkeypatch, sample_based, fractional_warmup, batch_schedule):
    from unittest.mock import Mock

    from megatron.core import num_microbatches_calculator as calculators
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training import global_vars, training
    from megatron.training.config import (
        CheckpointConfig,
        LoggerConfig,
        PretrainConfigContainer,
        SchedulerConfig,
        TrainingConfig,
    )

    cfg = PretrainConfigContainer(
        model=None,
        train=TrainingConfig(
            micro_batch_size=2,
            train_iters=None if sample_based else 10,
            train_samples=160 if sample_based else None,
            step_batch_size_schedule=batch_schedule,
        ),
        optimizer=OptimizerConfig(lr=0.001, weight_decay=0.1),
        scheduler=SchedulerConfig(
            lr_decay_style='WSD',
            lr_wsd_decay_style='minus_sqrt',
            lr_wsd_decay_iters=None if sample_based else 3,
            lr_wsd_decay_samples=48 if sample_based else None,
            lr_warmup_fraction=0.25 if fractional_warmup else None,
            lr_warmup_iters=0 if sample_based or fractional_warmup else 2,
            lr_warmup_samples=32 if sample_based and not fractional_warmup else 0,
        ),
        logger=LoggerConfig(),
        checkpoint=CheckpointConfig(),
    )
    args = SimpleNamespace(
        batch_parallel_size=8,
        data_parallel_size=4,
        gtp_weight_remat_size=4,
        gtp_remat_num_sequence_shards=2,
        micro_batch_size=2,
        train_iters=cfg.train.train_iters,
        train_samples=cfg.train.train_samples,
        step_batch_size_schedule=batch_schedule,
        lr=0.001,
        min_lr=0.00001,
        iteration=5,
        _current_train_iteration=17,
        consumed_train_samples=80,
        consumed_valid_samples=4,
        skipped_train_samples=2,
    )
    monkeypatch.setattr(global_vars, '_GLOBAL_ARGS', args)
    monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', cfg)
    monkeypatch.setattr(calculators, '_GLOBAL_NUM_MICROBATCHES_CALCULATOR', None)
    global_vars._resolve_runtime_batch_defaults(args)
    assert args.global_batch_size == args.eval_global_batch_size == 16
    assert args.eval_micro_batch_size == 2
    calculators.init_num_microbatches_calculator(
        rank=0,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=args.batch_parallel_size,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=batch_schedule,
    )
    global_vars._finalize_runtime_config(args)
    assert cfg._finalized
    assert args.start_weight_decay == args.end_weight_decay == 0.1
    assert args.train_iters == (7 if batch_schedule else 10)
    assert args.lr_decay_steps == args.wd_incr_steps == 160
    assert args.lr_warmup_steps == (40 if fractional_warmup else 32)
    assert args.wsd_decay_steps == 48
    assert (args.iteration, args._current_train_iteration) == (5, 17)
    assert (
        args.consumed_train_samples,
        args.consumed_valid_samples,
        args.skipped_train_samples,
    ) == (80, 4, 2)

    scheduler = Mock()
    monkeypatch.setattr(training, 'OptimizerParamScheduler', scheduler)
    training.get_optimizer_param_scheduler(object())
    kwargs = scheduler.call_args.kwargs
    assert kwargs['lr_decay_steps'] == kwargs['wd_incr_steps'] == 160
    assert kwargs['lr_warmup_steps'] == (40 if fractional_warmup else 32)
    assert kwargs['wsd_decay_steps'] == 48
    assert kwargs['lr_wsd_decay_style'] == 'minus_sqrt'
    assert kwargs['start_wd'] == kwargs['end_wd'] == 0.1
    # Re-entry must not revalidate a now-derived train_iters against train_samples.
    global_vars._finalize_runtime_config(args)
    assert args._current_train_iteration == 17
