"""Isaac Lab config dataclasses for SimToolReal.

Mirrors SimToolReal.yaml for the Isaac Lab path; used by SimToolRealEnv.
All parameters that mirror the legacy yaml keep the same defaults so that
the pretrained policy checkpoint is immediately compatible.
"""

from __future__ import annotations

import os
from dataclasses import field
from typing import List, Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

# Convenience alias for UrdfFileCfg's joint-drive nested config classes.
_JointDriveCfg = UrdfConverterCfg.JointDriveCfg
_PDGainsCfg = _JointDriveCfg.PDGainsCfg

# Resolve asset root: use the env var if set, otherwise fall back to the
# ``assets/`` directory relative to this file (3 levels up to repo root).
_ASSET_ROOT = os.path.abspath(
    os.environ.get(
        "ASSET_ROOT",
        os.path.join(os.path.dirname(__file__), "../../../assets"),
    )
)

# Legacy per-joint gains/limits from isaacgymenvs/tasks/simtoolreal/utils.py::populate_dof_properties.
_ARM_JOINTS = [f"iiwa14_joint_{i}" for i in range(1, 8)]
_ARM_STIFFNESS = [600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0]
_ARM_DAMPING = [
    27.027026473513512,
    27.027026473513512,
    24.672186769721083,
    22.067474708266914,
    9.752538131173853,
    9.147747263670984,
    9.147747263670984,
]
_ARM_EFFORT_LIMIT = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]

_HAND_JOINTS = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]
# Isaac Lab/URDF names for the same legacy hand semantics (same order as _HAND_JOINTS).
_HAND_JOINTS_LAB = [
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]
_LEGACY_TO_LAB_HAND_JOINTS = dict(zip(_HAND_JOINTS, _HAND_JOINTS_LAB))
_HAND_STIFFNESS = [
    6.95,
    13.2,
    4.76,
    6.62,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    1.38,
    4.76,
    6.62,
    0.9,
    0.9,
]
_HAND_DAMPING = [
    0.28676845,
    0.40845109,
    0.20394083,
    0.24044435,
    0.04190723,
    0.20859232,
    0.24595532,
    0.04243185,
    0.03504461,
    0.2085923,
    0.24595532,
    0.04243185,
    0.03504461,
    0.20859226,
    0.24595528,
    0.04243183,
    0.0350446,
    0.02782345,
    0.20859229,
    0.24595528,
    0.04243183,
    0.0350446,
]
_HAND_ARMATURE = [
    0.0032,
    0.0032,
    0.00265,
    0.00265,
    0.0006,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00012,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
]
_HAND_FRICTION = [
    0.132,
    0.132,
    0.07456,
    0.07456,
    0.01276,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.012,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
]


# Convert ordered legacy joint/value arrays into per-joint mapping dicts for Isaac Lab actuator cfg.
def _joint_map(joint_names: list[str], values: list[float]) -> dict[str, float]:
    return dict(zip(joint_names, values))


# ---------------------------------------------------------------------------
# Robot articulation config
# ---------------------------------------------------------------------------
# Matches assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
KUKA_SHARPA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(
            _ASSET_ROOT, "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        ),
        # Force regeneration so converter options below are always reflected in USD.
        force_usd_conversion=True,
        # Keep links editable so per-link collision material overrides can be applied.
        make_instanceable=False,
        fix_base=True,
        activate_contact_sensors=False,
        # Legacy Gym filters adjacent-link collisions; use coarse proxy here to avoid full self-contact chatter.
        self_collision=False,
        # PD gains for URDF→USD conversion; ImplicitActuatorCfg takes over at runtime.
        joint_drive=_JointDriveCfg(gains=_PDGainsCfg(stiffness=0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Legacy world-frame parity: place robot base at y=0.8.
        pos=(0.0, 0.8, 0.0),
        # Resting pose: iiwa14 arm (7 DOF) + SHARPA hand (22 DOF)
        # Matches hand_arm_default_dof_pos in legacy env.py
        joint_pos={
            "iiwa14_joint_1": -1.571,
            "iiwa14_joint_2": 1.571,
            "iiwa14_joint_3": -0.000,
            "iiwa14_joint_4": 1.376,
            "iiwa14_joint_5": -0.000,
            "iiwa14_joint_6": 1.485,
            "iiwa14_joint_7": 1.308,
            # SHARPA hand: all left_* joints at 0 (relaxed open hand)
            "left_.*": 0.0,
        },
    ),
    actuators={
        # Arm actuator group with legacy KUKA per-joint gains/effort limits.
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=_ARM_JOINTS,
            effort_limit_sim=_joint_map(_ARM_JOINTS, _ARM_EFFORT_LIMIT),
            stiffness=_joint_map(_ARM_JOINTS, _ARM_STIFFNESS),
            damping=_joint_map(_ARM_JOINTS, _ARM_DAMPING),
        ),
        # Hand actuator group with legacy SHARPA per-joint gains/armature/friction.
        "hand_joints": ImplicitActuatorCfg(
            joint_names_expr=_HAND_JOINTS_LAB,
            stiffness=_joint_map(_HAND_JOINTS_LAB, _HAND_STIFFNESS),
            damping=_joint_map(_HAND_JOINTS_LAB, _HAND_DAMPING),
            armature=_joint_map(_HAND_JOINTS_LAB, _HAND_ARMATURE),
            friction=_joint_map(_HAND_JOINTS_LAB, _HAND_FRICTION),
        ),
    },
)


# ---------------------------------------------------------------------------
# Table rigid object config
# ---------------------------------------------------------------------------
TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_ASSET_ROOT, "urdf/table_narrow.urdf"),
        # Force regeneration so converter options below are always reflected in USD.
        force_usd_conversion=True,
        # Keep table collision prim editable for explicit material binding in parity path.
        make_instanceable=False,
        # Avoid importing a fixed-base root joint for table under cloned envs.
        fix_base=False,
        joint_drive=None,  # table has no actuated joints
        # Disable articulation root so Isaac Lab treats this as a RigidObject, not Articulation
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
        # Keep table static in world-frame without articulated root-joint constraints.
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),  # tableResetZ = 0.38
    ),
)


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Full configuration for the SimToolReal Isaac Lab environment.

    All defaults match the legacy SimToolReal.yaml so that the pretrained
    policy checkpoint (140-obs / 29-action) is immediately compatible.
    """

    # -----------------------------------------------------------------------
    # Isaac Lab required fields (DirectRLEnvCfg MISSING fields)
    # -----------------------------------------------------------------------
    # scene: cloned grid of environments.  num_envs / env_spacing are kept as
    # top-level fields for convenience; __post_init__ keeps them in sync.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=1.2)
    # Legacy pretrained policies use controlFrequencyInv=1, so keep decimation=1.
    decimation: int = 1
    # obs/action space dimensions (must match the 140-dim obs / 29-DOF action)
    observation_space: int = 140
    action_space: int = 29

    # -----------------------------------------------------------------------
    # Scene (convenience duplicates — kept in sync with scene fields above)
    # -----------------------------------------------------------------------
    num_envs: int = 128
    env_spacing: float = 1.2  # envSpacing

    def __post_init__(self):
        """Sync scene.num_envs / scene.env_spacing with top-level fields."""
        self.scene.num_envs = self.num_envs
        self.scene.env_spacing = self.env_spacing

    # Episode
    episode_length_s: float = 10.0  # episodeLength=600 steps @ 60 Hz

    # -----------------------------------------------------------------------
    # Observation / action dims (must match pretrained policy)
    # -----------------------------------------------------------------------
    num_observations: int = 140
    num_actions: int = 29

    # -----------------------------------------------------------------------
    # Robot & scene assets
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = KUKA_SHARPA_CFG
    table_cfg: RigidObjectCfg = TABLE_CFG

    # Object to manipulate — a DexToolBench object name (see dextoolbench/objects.py)
    object_name: str = "claw_hammer"
    preloaded_object_names: Optional[List[str]] = None
    preloaded_table_urdfs: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # Physics simulation — matches SimToolReal.yaml [sim] section
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**21,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**27,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_collision_stack_size=2**28,
            gpu_heap_capacity=2**28,
            gpu_temp_buffer_capacity=2**26,
        ),
    )

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------
    dof_speed_scale: float = 1.5  # dofSpeedScale
    hand_moving_average: float = 0.1  # handMovingAverage
    arm_moving_average: float = 0.1  # armMovingAverage
    clamp_abs_observations: float = 10.0  # clampAbsObservations

    # -----------------------------------------------------------------------
    # Reward scales (must match pretrained policy training config)
    # -----------------------------------------------------------------------
    distance_delta_rew_scale: float = 50.0
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15  # metres
    keypoint_rew_scale: float = 200.0
    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003
    object_lin_vel_penalty_scale: float = 0.0
    object_ang_vel_penalty_scale: float = 0.0
    reach_goal_bonus: float = 1000.0
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size_keypoint_reward: bool = True
    fixed_size: List[float] = field(default_factory=lambda: [0.141, 0.03025, 0.0271])

    # -----------------------------------------------------------------------
    # Success / tolerance
    # -----------------------------------------------------------------------
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    max_consecutive_successes: int = 50
    success_steps: int = 10
    fall_distance: float = 0.24
    fall_penalty: float = 0.0
    force_consecutive_near_goal_steps: bool = False

    # -----------------------------------------------------------------------
    # Reset randomisation
    # -----------------------------------------------------------------------
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    randomize_object_rotation: bool = True
    reset_dof_pos_noise_fingers: float = 0.1
    reset_dof_pos_noise_arm: float = 0.1
    reset_dof_vel_noise: float = 0.5
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    start_arm_higher: bool = False

    # -----------------------------------------------------------------------
    # Delays and noise
    # -----------------------------------------------------------------------
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    joint_velocity_obs_noise_std: float = 0.01

    # -----------------------------------------------------------------------
    # Random forces (default all off for eval, on for training)
    # -----------------------------------------------------------------------
    force_scale: float = 2.0
    force_prob_range: List[float] = field(default_factory=lambda: [0.001, 0.1])
    torque_scale: float = 0.0
    torque_prob_range: List[float] = field(default_factory=lambda: [0.001, 0.1])
    force_only_when_lifted: bool = True
    torque_only_when_lifted: bool = True

    # -----------------------------------------------------------------------
    # Goal sampling
    # -----------------------------------------------------------------------
    goal_source: str = "delta"  # "delta" | "fixed_json" | "lie"
    goal_sampling_type: str = "delta"  # "delta" | "absolute" | "coin_flip"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: Optional[List[float]] = field(default_factory=lambda: [-0.35, -0.1, 0.68])
    target_volume_maxs: Optional[List[float]] = field(default_factory=lambda: [0.35, 0.2, 1.05])
    use_fixed_goal_states: bool = False
    fixed_goal_states: Optional[List] = None
    fixed_goal_states_json_path: Optional[str] = None
    lie_object_names: Optional[List[str]] = None
    lie_sampling_seed: int = 0
    lie_object_sampling: str = "uniform"

    # -----------------------------------------------------------------------
    # Fixed init pose (for eval)
    # -----------------------------------------------------------------------
    use_fixed_init_object_pose: bool = False
    object_start_pose: Optional[List[float]] = None  # [x,y,z,qx,qy,qz,qw]
    goal_object_pose: Optional[List[float]] = None

    # -----------------------------------------------------------------------
    # Friction overrides
    # -----------------------------------------------------------------------
    robot_friction: float = 0.5
    fingertip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------
    force_no_reset: bool = False
    debug_trace_path: Optional[str] = None
    reset_when_dropped: bool = True
    eval_success_tolerance: Optional[float] = None
    # Keep object stable before first hand interaction to avoid random physics blow-ups in eval.
    stabilize_object_pre_contact: bool = True
    pre_contact_distance_threshold: float = 0.08
    pre_contact_max_drift: float = 0.03
    pre_contact_max_speed: float = 0.2
