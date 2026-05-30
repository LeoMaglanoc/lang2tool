"""Shared supported-object definitions for the LLM evaluation entrypoints."""

from __future__ import annotations

from typing import Dict, List, Tuple

SUPPORTED_LLM_OBJECT_TASKS: Dict[str, str] = {
    "claw_hammer": "swing_down",
    "mallet_hammer": "swing_down",
    "cuboid_hammer_v014": "swing_down",
    "long_screwdriver": "spin_vertical",
    "short_screwdriver": "spin_vertical",
    "cylinder_screwdriver_v3009": "spin_vertical",
}

SUPPORTED_LLM_OBJECT_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "hammer": ("claw_hammer", "mallet_hammer", "cuboid_hammer_v014"),
    "screwdriver": (
        "long_screwdriver",
        "short_screwdriver",
        "cylinder_screwdriver_v3009",
    ),
}


# Return the ordered list of object names supported by LLM eval entrypoints.
def supported_llm_object_names() -> List[str]:
    """Return supported LLM object names in stable UI order."""
    return [
        "claw_hammer",
        "mallet_hammer",
        "cuboid_hammer_v014",
        "long_screwdriver",
        "short_screwdriver",
        "cylinder_screwdriver_v3009",
    ]


# Return the supported LLM task bound to one object name.
def supported_llm_task_name(object_name: str) -> str:
    """Return the fixed supported task for one LLM object."""
    resolved_object_name = str(object_name)
    if resolved_object_name not in SUPPORTED_LLM_OBJECT_TASKS:
        raise ValueError(
            f"Unsupported LLM object `{resolved_object_name}`. "
            f"Supported objects: {', '.join(supported_llm_object_names())}."
        )
    return SUPPORTED_LLM_OBJECT_TASKS[resolved_object_name]


# Return the coarse family label used for ambiguity checks and prompt wording.
def supported_llm_object_family(object_name: str) -> str:
    """Return the supported family label for one known object."""
    resolved_object_name = str(object_name)
    for family_name, object_names in SUPPORTED_LLM_OBJECT_FAMILIES.items():
        if resolved_object_name in object_names:
            return family_name
    raise ValueError(
        f"Unsupported LLM object `{resolved_object_name}`. "
        f"Supported objects: {', '.join(supported_llm_object_names())}."
    )


# Return one minimal metadata structure for dropdowns restricted to the supported families.
def supported_llm_data_structure() -> Dict[str, Dict[str, List[str]]]:
    """Return category/object/task metadata for only the supported LLM objects."""
    return {
        "hammer": {
            "claw_hammer": ["swing_down"],
            "mallet_hammer": ["swing_down"],
            "cuboid_hammer_v014": ["swing_down"],
        },
        "screwdriver": {
            "long_screwdriver": ["spin_vertical"],
            "short_screwdriver": ["spin_vertical"],
            "cylinder_screwdriver_v3009": ["spin_vertical"],
        },
    }
