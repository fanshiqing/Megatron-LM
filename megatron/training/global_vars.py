# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import signal
import sys
from datetime import timedelta

import torch

from megatron.core import Timers
from megatron.core.config import set_experimental_flag
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.jit import disable_jit_fuser
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.gtp_parallel_layout import get_batch_parallel_size
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.dist_signal_handler import DistributedSignalHandler
from megatron.training.state import TrainState

_GLOBAL_ARGS = None
_GLOBAL_RUNTIME_INITIALIZED = False
_GLOBAL_RUN_CONFIG = None
_GLOBAL_TRAIN_STATE = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_ENERGY_MONITOR = None
_GLOBAL_SIGNAL_HANDLER = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


class _LegacyTrainState(TrainState):
    """Live state view while the dev training loop still owns progress through args.

    Do not snapshot values on access: main utilities may mutate this object. All fields
    write through to the same argument namespace; resume FLOPs stays separate from the
    cumulative counter used by the running loop.
    """

    _arg_fields = {
        "iteration": "_current_train_iteration",
        "resume_iteration": "iteration",
        "num_floating_point_operations_so_far": "_current_floating_point_operations",
        "resume_num_floating_point_operations": "num_floating_point_operations_so_far",
    }

    def __init__(self, args):
        object.__setattr__(self, "_args", args)
        if not hasattr(args, "_current_train_iteration"):
            args._current_train_iteration = getattr(args, "iteration", 0)
        if not hasattr(args, "_current_floating_point_operations"):
            args._current_floating_point_operations = getattr(
                args, "num_floating_point_operations_so_far", 0
            )

    def __getattribute__(self, name):
        fields = TrainState.__dataclass_fields__
        if name in fields:
            args = object.__getattribute__(self, "_args")
            key = type(self)._arg_fields.get(name, name)
            return getattr(args, key, fields[name].default)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in TrainState.__dataclass_fields__:
            setattr(self._args, self._arg_fields.get(name, name), value)
        else:
            object.__setattr__(self, name, value)


def get_run_config():
    """Return the stored config, adapting legacy initialization when necessary."""
    global _GLOBAL_RUN_CONFIG
    if _GLOBAL_RUN_CONFIG is None:
        from megatron.training.argument_utils import pretrain_cfg_container_from_args

        _GLOBAL_RUN_CONFIG = pretrain_cfg_container_from_args(get_args())
    return _GLOBAL_RUN_CONFIG


def is_run_config_initialized():
    return _GLOBAL_RUN_CONFIG is not None


def set_run_config(cfg_container):
    global _GLOBAL_RUN_CONFIG
    _GLOBAL_RUN_CONFIG = cfg_container


def get_train_state():
    global _GLOBAL_TRAIN_STATE
    if _GLOBAL_TRAIN_STATE is None and _GLOBAL_ARGS is not None:
        _GLOBAL_TRAIN_STATE = _LegacyTrainState(_GLOBAL_ARGS)
    return _GLOBAL_TRAIN_STATE


def get_telemetry():
    """Telemetry services are not initialized by the retained dev runtime."""
    return None


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def get_energy_monitor():
    """Return energy monitor."""
    _ensure_var_is_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    return _GLOBAL_ENERGY_MONITOR


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler(exit_signal):

    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = DistributedSignalHandler(exit_signal).__enter__()


def _graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    from megatron.training.utils import print_rank_0

    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # synchronize all ranks before exiting
            try:
                # avoid deadlock if ranks don't all reach here
                torch.distributed.barrier()
            except Exception:
                pass

            torch.distributed.destroy_process_group()
    except Exception:
        pass

    sys.exit(0)


def is_global_variables_initialized():
    """Whether runtime services have been constructed, independently of CLI registration."""
    return _GLOBAL_RUNTIME_INITIALIZED


def set_global_variables(args, cfg_container=None, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    if isinstance(cfg_container, bool):
        build_tokenizer, cfg_container = cfg_container, None
    if cfg_container is not None:
        set_run_config(cfg_container)
    assert args is not None

    global _GLOBAL_RUNTIME_INITIALIZED
    assert not _GLOBAL_RUNTIME_INITIALIZED, 'global runtime services are already initialized'
    if _GLOBAL_ARGS is None:
        set_args(args)
    else:
        assert _GLOBAL_ARGS is args, 'runtime args must match the registered argument namespace'

    _resolve_runtime_batch_defaults(args)

    if args.step_batch_size_schedule is not None:
        print(f'> using step batch size schedule: {args.step_batch_size_schedule}')

    init_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        # Sequence-sharding GTP ranks do not introduce independent batch partitions.
        data_parallel_size=get_batch_parallel_size(args),
        decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
        step_batch_size_schedule=args.step_batch_size_schedule,
        seq_length=args.seq_length,
    )
    _finalize_runtime_config(args)
    if build_tokenizer:
        _ = _build_tokenizer(args)
        _bind_tokenizer_vocab_to_run_config(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_one_logger(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_energy_monitor(args)

    if args.enable_experimental:
        set_experimental_flag(True)

    if args.exit_signal_handler:
        _set_signal_handler(args.exit_signal)

    if args.exit_signal_handler_for_training:
        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    if args.disable_jit_fuser:
        disable_jit_fuser()

    _GLOBAL_RUNTIME_INITIALIZED = True


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_RUN_CONFIG, _GLOBAL_TRAIN_STATE, _GLOBAL_RUNTIME_INITIALIZED
    global _GLOBAL_ARGS
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_ENERGY_MONITOR
    global _GLOBAL_SIGNAL_HANDLER

    _GLOBAL_RUN_CONFIG = None
    _GLOBAL_TRAIN_STATE = None
    _GLOBAL_RUNTIME_INITIALIZED = False
    _GLOBAL_ARGS = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_ENERGY_MONITOR = None
    _GLOBAL_SIGNAL_HANDLER = None

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    global _GLOBAL_TRAIN_STATE
    _GLOBAL_TRAIN_STATE = None


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def _resolve_runtime_batch_defaults(args):
    """Resolve batch defaults before constructing the legacy microbatch calculator."""
    from megatron.training.config import PretrainConfigContainer

    cfg = get_run_config()
    if not isinstance(cfg, PretrainConfigContainer):
        return
    batch_sizes = cfg._resolve_batch_sizes(get_batch_parallel_size(args))
    (
        cfg.train.global_batch_size,
        cfg.validation.eval_global_batch_size,
        cfg.validation.eval_micro_batch_size,
    ) = batch_sizes
    args.global_batch_size, args.eval_global_batch_size, args.eval_micro_batch_size = batch_sizes


def _finalize_runtime_config(args):
    """Finalize config policy and bridge derived inputs to the args-owned runtime.

    Sample-based batch schedules require the microbatch calculator, so this runs after
    its creation. Only run configuration is copied back; live progress, resume counters
    and consumed samples remain owned by args/TrainState.
    """
    from dataclasses import fields

    from megatron.training.config import PretrainConfigContainer
    from megatron.training.config.container import ConfigContainerBase

    cfg = _GLOBAL_RUN_CONFIG
    if not isinstance(cfg, ConfigContainerBase):
        return
    cfg.finalize()
    if isinstance(cfg, PretrainConfigContainer):
        args.train_iters = cfg.train.train_iters
        args.global_batch_size = cfg.train.global_batch_size
        args.eval_global_batch_size = cfg.validation.eval_global_batch_size
        args.eval_micro_batch_size = cfg.validation.eval_micro_batch_size
        for field in fields(cfg.scheduler):
            setattr(args, field.name, getattr(cfg.scheduler, field.name))
    for field in fields(cfg.rng):
        setattr(args, field.name, getattr(cfg.rng, field.name))


def _bind_tokenizer_vocab_to_run_config(args):
    """Resolve deferred config vocabulary after the retained tokenizer lifecycle."""
    if _GLOBAL_RUN_CONFIG is None:
        return
    from megatron.training.argument_utils import resolve_tokenizer_vocab_size

    resolve_tokenizer_vocab_size(_GLOBAL_RUN_CONFIG, getattr(args, 'padded_vocab_size', None))


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, 'tensorboard writer')

    if (
        hasattr(args, 'tensorboard_dir')
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            print(
                'WARNING: TensorBoard writing requested but is not '
                'available (are you using PyTorch 1.1.0 or later?), '
                'no TensorBoard logs will be written.',
                flush=True,
            )


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, 'wandb writer')
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        wandb_config = vars(args)
        if (
            'kitchen_config_file' in wandb_config
            and wandb_config['kitchen_config_file'] is not None
        ):
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config['kitchen_config_file'], "r") as f:
                wandb_config['kitchen_config_file_contents'] = f.read()
        wandb_kwargs = {
            'dir': save_dir,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': wandb_config,
        }
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger and args.rank == (args.world_size - 1):
        if args.one_logger_async or getattr(args, 'wandb_project', ''):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger

            config = {
                'project': args.one_logger_project,
                'name': args.one_logger_run_name,
                'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print(
                'WARNING: one_logger package is required to enable e2e metrics '
                'tracking. please go to '
                'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                ' for details to install it'
            )


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        from megatron.training.utils import print_rank_0

        print_rank_0('enabling autoresume ...')
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except ImportError:
            print_rank_0('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def _set_energy_monitor(args):
    """Initialize energy monitor."""
    global _GLOBAL_ENERGY_MONITOR
    _ensure_var_is_not_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    _GLOBAL_ENERGY_MONITOR = EnergyMonitor()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_ENERGY_MONITOR
    _GLOBAL_ENERGY_MONITOR = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None
