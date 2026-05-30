"""Helpers for mapping Hydra training config into Isaac Lab env config."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf

from deployment.isaac.isaac_env_lab import _apply_overrides
from simtoolreal_lab.tasks.simtoolreal.env_lab_cfg import SimToolRealEnvCfg


# Recursively flatten nested task.env config into legacy-style `task.env.*` keys.
def _flatten_legacy_env_overrides(
    env_cfg_dict: Dict[str, Any], parent_key: str = "task.env"
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in env_cfg_dict.items():
        full_key = f"{parent_key}.{key}"
        if isinstance(value, dict):
            out.update(_flatten_legacy_env_overrides(value, parent_key=full_key))
        else:
            out[full_key] = value
    return out


# Build `SimToolRealEnvCfg` from Hydra cfg by reusing parity-proven override mapping.
def build_env_cfg_from_hydra_cfg(cfg) -> SimToolRealEnvCfg:
    env_cfg = SimToolRealEnvCfg()
    task_env_dict = OmegaConf.to_container(cfg.task.env, resolve=True)
    assert isinstance(task_env_dict, dict), "Expected cfg.task.env to resolve to a dict."
    overrides = _flatten_legacy_env_overrides(task_env_dict)
    _apply_overrides(env_cfg, overrides)
    return env_cfg
