"""Training entrypoint for SimToolReal on Isaac Lab (no isaacgym imports).

Launch training with:
    docker compose run --rm isaaclab python simtoolreal_lab/train_lab.py \\
        task.env.numEnvs=1024 experiment=my_experiment

This replaces isaacgymenvs/train.py for the Isaac Lab migration.
All rl_games configuration (network, PPO hyperparameters) is loaded from
simtoolreal_lab/cfg/train/SimToolRealLabPPO.yaml via Hydra.

Key differences from train.py:
  - No isaacgym / gymapi / gymtorch imports
  - Env created via SimToolRealEnv (DirectRLEnv) + RLGPUEnvLab wrapper
  - Hydra config root: isaaclab/cfg/  (separate from isaacgymenvs/cfg/)
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import types

import hydra
from omegaconf import DictConfig, OmegaConf
from rl_games.common.algo_observer import AlgoObserver

from simtoolreal_lab.utils.rlgames_utils_lab import RLGPUEnvLab, SimToolRealLabAlgoObserver

# Disable TorchDynamo in this container to avoid known torch/isaac extension import incompatibilities.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Remove Isaac prebundle torch path to avoid mixed torch/torchvision submodule imports.
_PREBUNDLE_MARKER = "omni.isaac.ml_archive/pip_prebundle"
sys.path = [p for p in sys.path if _PREBUNDLE_MARKER not in p]

# ---------------------------------------------------------------------------
# rl_games setup
# ---------------------------------------------------------------------------


# Track the most recent env-step and epoch-completion signals so silent stalls become diagnosable.
class TrainingProgressWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        now = time.monotonic()
        self._start_time = now
        self._last_env_step_time = now
        self._last_env_reset_time = now
        self._last_epoch_time = now
        self._env_step_count = 0
        self._env_reset_count = 0
        self._last_env_batch = 0
        self._last_epoch_num = -1
        self._last_frame = -1
        self._last_total_time = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # Record one environment reset so watchdog snapshots show whether resets are still flowing.
    def note_env_reset(self, env_batch_size: int) -> None:
        with self._lock:
            self._last_env_reset_time = time.monotonic()
            self._env_reset_count += 1
            self._last_env_batch = int(env_batch_size)

    # Record one environment step so watchdog snapshots show whether simulation is still advancing.
    def note_env_step(self, env_batch_size: int) -> None:
        with self._lock:
            self._last_env_step_time = time.monotonic()
            self._env_step_count += 1
            self._last_env_batch = int(env_batch_size)

    # Record one completed training epoch once rl_games has printed stats for it.
    def note_epoch_end(self, *, frame: int, epoch_num: int, total_time: float) -> None:
        with self._lock:
            self._last_epoch_time = time.monotonic()
            self._last_epoch_num = int(epoch_num)
            self._last_frame = int(frame)
            self._last_total_time = float(total_time)

    # Return one stable watchdog snapshot for logging and unit tests.
    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            now = time.monotonic()
            return {
                "uptime_sec": now - self._start_time,
                "seconds_since_env_step": now - self._last_env_step_time,
                "seconds_since_env_reset": now - self._last_env_reset_time,
                "seconds_since_epoch": now - self._last_epoch_time,
                "env_step_count": self._env_step_count,
                "env_reset_count": self._env_reset_count,
                "last_env_batch": self._last_env_batch,
                "last_epoch_num": self._last_epoch_num,
                "last_frame": self._last_frame,
                "last_total_time": self._last_total_time,
            }

    # Format one compact watchdog line so stalled runs show the last known env and epoch progress.
    def format_status_line(self) -> str:
        snapshot = self.snapshot()
        return (
            "[simtoolreal_watchdog] "
            f"uptime_sec={snapshot['uptime_sec']:.1f} "
            f"since_env_step_sec={snapshot['seconds_since_env_step']:.1f} "
            f"since_env_reset_sec={snapshot['seconds_since_env_reset']:.1f} "
            f"since_epoch_sec={snapshot['seconds_since_epoch']:.1f} "
            f"env_steps={snapshot['env_step_count']} "
            f"env_resets={snapshot['env_reset_count']} "
            f"last_env_batch={snapshot['last_env_batch']} "
            f"last_epoch={snapshot['last_epoch_num']} "
            f"last_frame={snapshot['last_frame']} "
            f"last_total_time={snapshot['last_total_time']:.1f}"
        )

    # Start one background logger thread that periodically emits watchdog snapshots to stdout.
    def start(self, interval_sec: int) -> None:
        if interval_sec <= 0 or self._thread is not None:
            return

        def _worker() -> None:
            while not self._stop_event.wait(interval_sec):
                print(self.format_status_line(), flush=True)

        self._thread = threading.Thread(
            target=_worker,
            name="simtoolreal-watchdog",
            daemon=True,
        )
        self._thread.start()

    # Stop the background logger thread so teardown does not leave watchdog output running.
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# Minimal multi-observer wrapper for rl_games without Isaac Gym dependencies.
class MultiObserver(AlgoObserver):
    def __init__(self, observers):
        super().__init__()
        self.observers = observers

    def before_init(self, base_name, config, experiment_name):
        for obs in self.observers:
            obs.before_init(base_name, config, experiment_name)

    def after_init(self, algo):
        for obs in self.observers:
            obs.after_init(algo)

    def process_infos(self, infos, done_indices, **kwargs):
        for obs in self.observers:
            obs.process_infos(infos, done_indices, **kwargs)

    def after_steps(self):
        for obs in self.observers:
            obs.after_steps()

    def after_print_stats(self, frame, epoch_num, total_time):
        for obs in self.observers:
            obs.after_print_stats(frame, epoch_num, total_time)


# Ensure rl_games naming assumptions hold when exploration mode parses policy index.
def _normalize_experiment_name_for_rl_games(name: str) -> str:
    if not name:
        return "00_run"
    prefix = name.split("_", 1)[0]
    if prefix.isdigit():
        return name
    return f"00_{name}"


# Install a minimal torch._dynamo shim when Isaac prebundle import is incompatible.
def _install_torch_dynamo_shim_if_needed() -> None:
    shim = types.ModuleType("torch._dynamo")

    def _disable(fn=None, recursive=True, wrapping=True):
        if fn is None:
            return lambda f: f
        return fn

    shim.disable = _disable
    shim.graph_break = lambda *args, **kwargs: None
    sys.modules["torch._dynamo"] = shim


# Install a minimal torchvision stub so Isaac task discovery avoids incompatible prebundled ops.
def _install_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return
    torchvision_module = types.ModuleType("torchvision")
    torchvision_utils = types.ModuleType("torchvision.utils")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_models = types.ModuleType("torchvision.models")
    torchvision_datasets = types.ModuleType("torchvision.datasets")
    torchvision_io = types.ModuleType("torchvision.io")
    torchvision_ops = types.ModuleType("torchvision.ops")
    torchvision_utils.save_image = lambda *args, **kwargs: None
    torchvision_utils.make_grid = lambda *args, **kwargs: None
    torchvision_transforms.Compose = lambda transforms: transforms
    torchvision_transforms.Normalize = lambda *args, **kwargs: ("normalize", args, kwargs)
    torchvision_module.utils = torchvision_utils
    torchvision_module.transforms = torchvision_transforms
    torchvision_module.models = torchvision_models
    torchvision_module.datasets = torchvision_datasets
    torchvision_module.io = torchvision_io
    torchvision_module.ops = torchvision_ops
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.utils"] = torchvision_utils
    sys.modules["torchvision.transforms"] = torchvision_transforms
    sys.modules["torchvision.models"] = torchvision_models
    sys.modules["torchvision.datasets"] = torchvision_datasets
    sys.modules["torchvision.io"] = torchvision_io
    sys.modules["torchvision.ops"] = torchvision_ops


# Build the observer list so Isaac Lab runs log env metrics like successes to TensorBoard.
def _build_observers(
    cfg: DictConfig, progress_watchdog: TrainingProgressWatchdog | None = None
) -> list[AlgoObserver]:
    observers: list[AlgoObserver] = [
        SimToolRealLabAlgoObserver(progress_watchdog=progress_watchdog)
    ]

    if cfg.wandb_activate:
        from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver

        observers.append(WandbAlgoObserver(cfg))

    return observers


def _build_runner(
    cfg: DictConfig,
    env: RLGPUEnvLab,
    progress_watchdog: TrainingProgressWatchdog | None = None,
):
    """Configure and return an rl_games Runner for the given env."""
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    # Register env under a new name so we don't collide with Isaac Gym
    env_configurations.register(
        "rlgpu_lab",
        {
            "vecenv_type": "RLGPU_LAB",
            "env_creator": lambda **kwargs: env,
        },
    )
    vecenv.register(
        "RLGPU_LAB",
        lambda config_name, num_actors, **kwargs: env,
    )

    # Build rl_games config dict from Hydra
    from isaacgymenvs.utils.reformat import omegaconf_to_dict

    rlg_cfg = omegaconf_to_dict(cfg.train)
    rlg_cfg["params"]["config"]["device"] = cfg.rl_device
    rlg_cfg["params"]["config"]["num_actors"] = env.num_envs

    runner = Runner(MultiObserver(_build_observers(cfg, progress_watchdog=progress_watchdog)))
    runner.load(rlg_cfg)
    runner.set_vec_env(env)
    runner.reset()
    return runner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Build one optional progress watchdog controlled by an env var so silent stalls emit periodic state snapshots.
def _build_progress_watchdog_from_env() -> tuple[TrainingProgressWatchdog | None, int]:
    interval_sec = int(os.environ.get("SIMTOOLREAL_WATCHDOG_SEC", "0"))
    if interval_sec <= 0:
        return None, 0
    return TrainingProgressWatchdog(), interval_sec


def launch(cfg: DictConfig) -> None:
    """Build env and runner, then run training or evaluation."""
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from simtoolreal_lab.tasks.simtoolreal.env_lab import SimToolRealEnv
    from simtoolreal_lab.train_cfg_utils import build_env_cfg_from_hydra_cfg

    set_np_formatting()
    rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)
    cfg.train.params.config.name = _normalize_experiment_name_for_rl_games(
        str(cfg.train.params.config.name)
    )
    _install_torch_dynamo_shim_if_needed()

    # Build Isaac Lab env cfg from full Hydra task.env mapping.
    env_cfg = build_env_cfg_from_hydra_cfg(cfg)

    progress_watchdog, watchdog_interval_sec = _build_progress_watchdog_from_env()
    if progress_watchdog is not None:
        progress_watchdog.start(watchdog_interval_sec)
        print(
            f"[simtoolreal_watchdog] enabled interval_sec={watchdog_interval_sec}",
            flush=True,
        )

    env = RLGPUEnvLab(SimToolRealEnv(env_cfg), progress_watchdog=progress_watchdog)
    runner = _build_runner(cfg, env, progress_watchdog=progress_watchdog)

    # Save config snapshot
    if not cfg.test:
        experiment_dir = os.path.join("runs", cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

    try:
        runner.run(
            {
                "train": not cfg.test,
                "play": cfg.test,
                "checkpoint": cfg.checkpoint,
                "sigma": cfg.sigma if cfg.sigma != "" else None,
            }
        )
    finally:
        if progress_watchdog is not None:
            progress_watchdog.stop()


if __name__ == "__main__":
    # Start Isaac Lab app before imports that depend on carb settings.
    _install_torchvision_stub()
    from isaaclab.app import AppLauncher

    # Emit periodic Python stack traces when requested so long-running smoke runs can be diagnosed.
    traceback_timeout_sec = int(os.environ.get("SIMTOOLREAL_DUMP_TRACEBACK_SEC", "0"))
    if traceback_timeout_sec > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(traceback_timeout_sec, repeat=True)

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    # Use local Hydra config root for Isaac Lab training.
    @hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
    def _main(cfg: DictConfig) -> None:
        launch(cfg)

    _main()
    # Force process exit after the training main returns to avoid Isaac teardown hangs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
