"""OpenAI tool schema definitions for LLM chat goal generation."""

# Active-tool switch schema for selecting hammer or screwdriver in the viewer/runtime.
SWITCH_ACTIVE_OBJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "switch_active_object",
        "description": (
            "Switch the active tool/object in the current viewer or runtime. "
            "Use this when the user explicitly asks to switch to the hammer or screwdriver."
        ),
        "parameters": {
            "type": "object",
            "required": ["object_name"],
            "properties": {
                "object_name": {
                    "type": "string",
                    "enum": [
                        "claw_hammer",
                        "mallet_hammer",
                        "cuboid_hammer_v014",
                        "long_screwdriver",
                        "short_screwdriver",
                        "cylinder_screwdriver_v3009",
                    ],
                },
            },
        },
    },
}

# GeometricGoalV1 tool schema passed to the OpenAI function tool.
GENERATE_GOALS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_goals",
        "description": (
            "Generate goal poses for the robot. "
            "Call this when the user asks to perform a manipulation task."
        ),
        "parameters": {
            "type": "object",
            "required": [
                "schema_version",
                "task_label",
                "object_frame",
                "contact_point_object",
                "approach_direction_object",
                "tool_axis_object",
                "pregrasp_offset_m",
                "grasp_depth_m",
                "lift_height_m",
                "timing_s",
            ],
            "properties": {
                "schema_version": {"type": "string", "enum": ["v1"]},
                "task_label": {"type": "string"},
                "object_frame": {"type": "string"},
                "contact_point_object": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "approach_direction_object": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "tool_axis_object": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "pregrasp_offset_m": {"type": "number", "minimum": 0.0},
                "grasp_depth_m": {"type": "number", "minimum": 0.0},
                "lift_height_m": {"type": "number", "minimum": 0.0},
                "timing_s": {
                    "type": "object",
                    "required": ["approach", "close", "lift"],
                    "properties": {
                        "approach": {"type": "number"},
                        "close": {"type": "number"},
                        "lift": {"type": "number"},
                    },
                },
            },
        },
    },
}


# Relative goal-delta tool schema for translation/rotation edits of active goals.
APPLY_GOAL_DELTA_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_goal_delta",
        "description": (
            "Apply a relative pose delta to the current active goals. "
            "Use this for commands like move left/right/up/down/forward/back, rotate, "
            "or semantic poses such as upright/flat/head-down/tip-forward/face-table."
        ),
        "parameters": {
            "type": "object",
            "required": ["frame", "target"],
            "properties": {
                "delta_translation_m": {
                    "type": "array",
                    "description": ("Camera-spawn local translation [right_m, up_m, forward_m]."),
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "delta_euler_rad": {
                    "type": "array",
                    "description": "Intrinsic XYZ Euler rotation delta [roll,pitch,yaw] in radians.",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "frame": {"type": "string", "enum": ["camera_spawn"]},
                "target": {"type": "string", "enum": ["all_active_goals"]},
                "semantic_target": {
                    "type": "string",
                    "enum": ["upright", "flat", "head_down", "tip_forward", "face_table"],
                },
                "semantic_preserve_position": {
                    "type": "boolean",
                    "description": (
                        "When true, semantic alignment rotates only and preserves current position."
                    ),
                },
            },
        },
    },
}


# Release tool command schema for opening hand and transitioning to release flow.
RELEASE_TOOL_COMMAND = {
    "type": "function",
    "function": {
        "name": "release_tool",
        "description": (
            "Release the currently grasped tool by opening the hand after a short pre-release "
            "placement motion. Use this for commands like release tool, open hand, drop tool."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# Grasp tool command schema for returning to a hover-above-current-tool target.
GRASP_TOOL_COMMAND = {
    "type": "function",
    "function": {
        "name": "grasp_tool",
        "description": (
            "Re-grasp the current tool by setting a hover goal above its current pose while "
            "preserving tool orientation, then resume normal policy control."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# Read-only sim state schema for current object/goal pose and pose semantics.
GET_SIM_STATE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_sim_state",
        "description": (
            "Get current simulation state for LLM reasoning: current object pose, goal pose, "
            "runtime mode, and semantic pose definitions for the active object."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# Named Lie-trajectory execution schema for strike-target-directed swing/twist commands.
EXECUTE_LIE_TRAJECTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_lie_trajectory",
        "description": (
            "Compile and execute a Lie swing trajectory toward an explicit tabletop XY strike "
            "target. Use this after get_sim_state for commands like swing on strike point a or "
            "swing on the right side of the table. For screwdriver spin_vertical, the same "
            "strike_target_xy is the tabletop hover target above which the screwdriver should "
            "twist. In the shared tabletop frame, front means farther from the camera and back "
            "means closer to the camera."
        ),
        "parameters": {
            "type": "object",
            "required": ["strike_target_xy"],
            "properties": {
                "strike_target_xy": {
                    "type": "array",
                    "description": "World-table strike target [x_m, y_m] on the tabletop plane.",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "target_description": {
                    "type": "string",
                    "description": "Short natural-language description of the intended strike area.",
                },
                "replace_active_goals": {
                    "type": "boolean",
                    "description": "When true, replace the current active goal sequence.",
                },
                "object_name": {
                    "type": "string",
                    "enum": [
                        "claw_hammer",
                        "mallet_hammer",
                        "cuboid_hammer_v014",
                        "long_screwdriver",
                        "short_screwdriver",
                        "cylinder_screwdriver_v3009",
                    ],
                    "description": (
                        "Supported tool chosen for this motion. Swing requests should use one "
                        "of the hammer objects, and twist requests should use one of the "
                        "screwdriver objects."
                    ),
                },
            },
        },
    },
}


# Predefined-motion execution schema for replaying the recorded object-specific motion.
EXECUTE_PREDEFINED_SWING_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_predefined_swing",
        "description": (
            "Execute the recorded predefined object-specific motion directly. "
            "Use this for commands like please do predefined swing or predefined screwdriver motion."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "object_name": {
                    "type": "string",
                    "enum": [
                        "claw_hammer",
                        "mallet_hammer",
                        "cuboid_hammer_v014",
                        "long_screwdriver",
                        "short_screwdriver",
                        "cylinder_screwdriver_v3009",
                    ],
                    "description": ("Supported tool chosen for predefined playback."),
                },
            },
        },
    },
}
