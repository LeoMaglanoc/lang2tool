"""rl_games adapter for Isaac Lab DirectRLEnv.

``DirectRLEnv.step()`` returns ``(obs_dict, rew, terminated, truncated, info)``
while rl_games ``IVecEnv`` expects ``(obs_dict, rew, done, info)`` where
``done = terminated | truncated``.

This module provides ``RLGPUEnvLab``, a thin wrapper that bridges the two
conventions and also exposes the ``gym.spaces`` attributes rl_games needs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gym
import numpy as np
import torch
from torch import Tensor

from isaacgymenvs.utils.utils import flatten_dict

try:
    from rl_games.common.algo_observer import AlgoObserver
    from rl_games.common.vecenv import IVecEnv
except ImportError:
    # Fall back to object so the module is importable for linting
    AlgoObserver = object  # type: ignore[misc, assignment]
    IVecEnv = object  # type: ignore[misc, assignment]


class SimToolRealLabAlgoObserver(AlgoObserver):
    """Log Isaac Lab env metrics through rl_games without importing legacy Isaac Gym tasks."""

    # Initialize one observer state container for TensorBoard-compatible env metrics.
    def __init__(self, progress_watchdog: Any | None = None) -> None:
        super().__init__()
        self.algo = None
        self.writer = None
        self.ep_infos: list[dict[str, Any]] = []
        self.direct_info: dict[str, Any] = {}
        self.episode_cumulative: dict[str, Tensor] = {}
        self.episode_cumulative_avg: dict[str, deque[float]] = {}
        self.new_finished_episodes = False
        self.progress_watchdog = progress_watchdog

    # Capture the rl_games writer once the algorithm instance is ready.
    def after_init(self, algo) -> None:
        self.algo = algo
        self.writer = algo.writer

    # Flatten env info dicts into scalar summaries and accumulate completed-episode metrics.
    def process_infos(self, infos, done_indices, **kwargs) -> None:
        if not isinstance(infos, dict):
            return

        if "episode" in infos:
            self.ep_infos.append(infos["episode"])

        if "episode_cumulative" in infos:
            for key, value in infos["episode_cumulative"].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value

            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()

                for key, value in infos["episode_cumulative"].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque(
                            [], maxlen=self.algo.games_to_track
                        )

                    self.episode_cumulative_avg[key].append(
                        self.episode_cumulative[key][done_idx].item()
                    )
                    self.episode_cumulative[key][done_idx] = 0

        infos_flat = flatten_dict(infos, prefix="", separator="/")
        self.direct_info = {}
        for key, value in infos_flat.items():
            if (
                isinstance(value, float)
                or isinstance(value, int)
                or (isinstance(value, torch.Tensor) and len(value.shape) == 0)
            ):
                self.direct_info[key] = value

        for tag in ["successes", "closest_keypoint_max_dist", "discounted_reward"]:
            if tag in infos:
                self.direct_info[tag] = infos[tag].mean()
                self.direct_info[f"{tag}_median"] = torch.median(infos[tag]).item()
                self.direct_info[f"{tag}_max"] = infos[tag].max()
                for key in infos:
                    if key.startswith(f"{tag}_per_block"):
                        self.direct_info[key] = torch.mean(infos[key]).item()

        if "true_objective" in infos:
            self.direct_info["true_objective_mean"] = infos["true_objective"].mean()
            self.direct_info["true_objective_max"] = infos["true_objective"].max()

    # Write env scalar summaries to TensorBoard after rl_games prints one training iteration.
    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        if self.progress_watchdog is not None:
            self.progress_watchdog.note_epoch_end(
                frame=int(frame),
                epoch_num=int(epoch_num),
                total_time=float(total_time),
            )

        if self.writer is None:
            return

        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                self.writer.add_scalar("Episode/" + key, torch.mean(infotensor), frame)
            self.ep_infos.clear()

        if self.new_finished_episodes:
            for key in self.episode_cumulative_avg:
                self.writer.add_scalar(
                    f"episode_cumulative/{key}", np.mean(self.episode_cumulative_avg[key]), frame
                )
                self.writer.add_scalar(
                    f"episode_cumulative_min/{key}_min",
                    np.min(self.episode_cumulative_avg[key]),
                    frame,
                )
                self.writer.add_scalar(
                    f"episode_cumulative_max/{key}_max",
                    np.max(self.episode_cumulative_avg[key]),
                    frame,
                )
            self.new_finished_episodes = False

        for key, value in self.direct_info.items():
            self.writer.add_scalar(f"{key}/frame", value, frame)
            self.writer.add_scalar(f"{key}/iter", value, frame)
            self.writer.add_scalar(f"{key}/time", value, frame)


class RLGPUEnvLab(IVecEnv):
    """Adapter wrapping a ``DirectRLEnv`` for rl_games PPO.

    Usage::

        from simtoolreal_lab.utils.rlgames_utils_lab import RLGPUEnvLab
        from simtoolreal_lab.tasks.simtoolreal import SimToolRealEnv, SimToolRealEnvCfg

        env = SimToolRealEnv(SimToolRealEnvCfg(num_envs=1024))
        wrapped = RLGPUEnvLab(env)

        from rl_games.common.env_configurations import register
        register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kw: wrapped})
    """

    # Initialize the rl_games adapter and cache one startup observation for asymmetric critics.
    def __init__(self, env, clip_obs: float = 10.0, progress_watchdog: Any | None = None) -> None:
        self._env = env
        # Keep legacy attribute name expected by rl_games internals.
        self.env = env
        self.clip_obs = clip_obs
        self.progress_watchdog = progress_watchdog

        # rl_games reads these attributes directly
        num_obs: int = env.cfg.num_observations
        num_act: int = env.cfg.num_actions

        self.observation_space = gym.spaces.Box(
            low=-clip_obs, high=clip_obs, shape=(num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_act,), dtype=np.float32)
        self.state_space: Optional[gym.spaces.Space] = None

        # rl_games expects num_actors (= num_envs)
        self.num_actors = env.num_envs
        initial_obs, _ = self._env.reset()
        if self.progress_watchdog is not None:
            self.progress_watchdog.note_env_reset(self.num_envs)
        self._cached_reset_obs = self._format_obs(initial_obs)

    # Map Isaac Lab policy/critic observations onto the rl_games obs/states contract.
    def _format_obs(self, obs_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        formatted_obs = {"obs": obs_dict["policy"]}
        critic_obs = obs_dict.get("critic")
        if critic_obs is not None:
            formatted_obs["states"] = critic_obs
            if self.state_space is None:
                state_dim = int(critic_obs.shape[-1])
                self.state_space = gym.spaces.Box(
                    low=-self.clip_obs,
                    high=self.clip_obs,
                    shape=(state_dim,),
                    dtype=np.float32,
                )
        return formatted_obs

    # ------------------------------------------------------------------
    # IVecEnv interface
    # ------------------------------------------------------------------

    # Step the Lab env and expose both policy and privileged critic observations to rl_games.
    def step(self, actions: Tensor):
        """Step the environment and return (obs_dict, rew, done, info)."""
        obs_dict, rew, terminated, truncated, info = self._env.step(actions)
        if self.progress_watchdog is not None:
            self.progress_watchdog.note_env_step(self.num_envs)
        done = terminated | truncated
        return self._format_obs(obs_dict), rew, done, info

    # Return the cached startup reset once, then fall back to normal Isaac Lab resets.
    def reset(self):
        """Reset all environments and return the initial observation."""
        if self._cached_reset_obs is not None:
            cached_obs = self._cached_reset_obs
            self._cached_reset_obs = None
            return cached_obs
        obs_dict, _ = self._env.reset()
        if self.progress_watchdog is not None:
            self.progress_watchdog.note_env_reset(self.num_envs)
        return self._format_obs(obs_dict)

    # Reset finished environments and preserve the same rl_games obs/states structure.
    def reset_done(self):
        """Called by rl_games after episodes finish — delegate to the env."""
        obs_dict, _ = self._env.reset()
        if self.progress_watchdog is not None:
            self.progress_watchdog.note_env_reset(self.num_envs)
        return self._format_obs(obs_dict)

    # ------------------------------------------------------------------
    # Misc properties rl_games reads
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    # Report one agent per environment so rl_games episode bookkeeping matches this single-policy task.
    def get_number_of_agents(self) -> int:
        return 1

    def has_action_mask(self) -> bool:
        return False

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:  # noqa: D102
        pass

    # Provide rl_games with action/observation space metadata.
    # Report action/observation/state spaces in the format rl_games expects.
    def get_env_info(self):
        info = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        if self.state_space is not None:
            info["state_space"] = self.state_space
        return info

    # Optional hook used by rl_games for curriculum signals.
    def set_train_info(self, env_frames, *args_, **kwargs_):
        return None


def build_rlgames_env(cfg) -> RLGPUEnvLab:
    """Instantiate ``SimToolRealEnv`` and wrap it for rl_games.

    Args:
        cfg: ``SimToolRealEnvCfg`` dataclass instance.

    Returns:
        An ``RLGPUEnvLab`` ready to be passed to the rl_games Runner.
    """
    from simtoolreal_lab.tasks.simtoolreal import SimToolRealEnv

    env = SimToolRealEnv(cfg)
    return RLGPUEnvLab(env)
