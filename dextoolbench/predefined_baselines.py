"""Shared prerecorded-trajectory baseline resolution for benchmark objects."""

from __future__ import annotations

from pathlib import Path

from dextoolbench.metadata import OBJECT_NAME_TO_CATEGORY
from isaacgymenvs.utils.utils import get_repo_root_dir

PREDEFINED_BASELINE_OBJECTS = {
    "cuboid_hammer_v014": "claw_hammer",
    "cylinder_screwdriver_v3009": "long_screwdriver",
}


# Resolve the object whose prerecorded trajectory should be used as the reference baseline.
def resolve_predefined_baseline_object_name(object_name: str) -> str:
    """Return the object name whose prerecorded baseline should back the requested object."""
    return PREDEFINED_BASELINE_OBJECTS.get(str(object_name), str(object_name))


# Build the prerecorded trajectory JSON path for one object/task pair with proxy fallback.
def resolve_predefined_trajectory_path(object_name: str, task_name: str) -> Path:
    """Return the trajectory JSON path used for predefined playback and benchmark baselines."""
    resolved_object_name = resolve_predefined_baseline_object_name(object_name)
    object_category = OBJECT_NAME_TO_CATEGORY[resolved_object_name]
    return (
        get_repo_root_dir()
        / "dextoolbench/trajectories"
        / object_category
        / resolved_object_name
        / f"{task_name}.json"
    )
