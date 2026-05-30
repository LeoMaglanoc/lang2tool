"""SimToolReal Isaac Lab environment — DirectRLEnv port of env.py.

Migration from Isaac Gym VecTask:
  gym.create_sim()                      → SimulationContext (DirectRLEnv)
  gym.create_env() / clone              → scene.clone_environments()
  gym.load_asset()                      → ArticulationCfg / RigidObjectCfg
  gym.acquire_dof_state_tensor()        → robot.data.joint_pos / joint_vel
  gym.acquire_actor_root_state_tensor() → obj.data.root_pos_w / root_quat_w
  gym.refresh_*_tensor()               → automatic each sim step
  gym.set_dof_position_target_tensor()  → robot.set_joint_position_target()
  gymtorch.wrap_tensor()                → not needed (native torch)

Observation vector (140 dims) and action vector (29 dims) are identical to
the legacy env so that the pretrained checkpoint is compatible without changes.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

# Isaac Lab imports — only available inside the Docker container.
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv
    from isaaclab.scene import InteractiveScene  # noqa: F401
    from isaaclab.sim.utils import bind_physics_material

    _ISAACLAB_AVAILABLE = True
except Exception:
    _ISAACLAB_AVAILABLE = False

    # Provide a dummy base so that the module can be imported for linting/static
    # analysis without Isaac Lab installed.
    class DirectRLEnv:  # type: ignore[no-redef]
        pass


from dextoolbench.eval_config import OBJECT_CATEGORY_TO_TABLE_URDF
from dextoolbench.metadata import OBJECT_NAME_TO_CATEGORY
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.observation_action_utils_sharpa import (
    JOINT_NAMES_ISAACGYM,
    Q_LOWER_LIMITS_np,
    Q_UPPER_LIMITS_np,
)
from isaacgymenvs.utils.torch_jit_utils import (
    quat_rotate,
    scale,
    tensor_clamp,
    torch_rand_float,
    unscale,
)

# Support direct script entrypoints that import this module without the repo root on sys.path.
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from new_rl_policy.common import build_random_lie_artifact, resolve_lie_object_names

try:
    from simtoolreal_lab.tasks.simtoolreal.env_lab_cfg import (
        _LEGACY_TO_LAB_HAND_JOINTS,
        TABLE_CFG,
        SimToolRealEnvCfg,
    )
except Exception:  # pragma: no cover - unit-test import fallback without Isaac Lab app state.
    _LEGACY_TO_LAB_HAND_JOINTS = {}
    TABLE_CFG = None

    class SimToolRealEnvCfg:  # type: ignore[no-redef]
        pass


from simtoolreal_lab.tasks.simtoolreal.quaternion_interface_utils import (
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from simtoolreal_lab.tasks.simtoolreal.queue_utils import (
    update_delay_queue,
)
from simtoolreal_lab.tasks.simtoolreal.reset_utils import (
    compute_termination_and_truncation,
)
from simtoolreal_lab.tasks.simtoolreal.reward_utils import (
    compute_action_penalties,
    compute_keypoint_reward,
    compute_lifting_reward,
    compute_near_goal_success,
)
from simtoolreal_lab.tasks.simtoolreal.stability_utils import (
    compute_pre_contact_stabilization_mask,
)


def _quat_mul_xyzw(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply two unit quaternions in xyzw convention."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _sample_random_quat(n: int, device: str) -> Tensor:
    """Sample n uniformly random unit quaternions (xyzw)."""
    u = torch.rand(n, 3, device=device)
    q = torch.stack(
        [
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
        ],
        dim=-1,
    )
    return q


def _axis_angle_to_quat_xyzw(axis: Tensor, angle_rad: Tensor) -> Tensor:
    """Convert axis-angle to unit quaternion (xyzw)."""
    half = angle_rad * 0.5
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    axis_norm = torch.nn.functional.normalize(axis, dim=-1)
    return torch.cat([axis_norm * sin_half.unsqueeze(-1), cos_half.unsqueeze(-1)], dim=-1)


# Resolve repo-local asset paths for preloaded table URDF lists.
def _resolve_asset_path(asset_path: str) -> str:
    """Return an absolute asset path for a URDF referenced by eval overrides."""
    resolved_path = Path(asset_path)
    if resolved_path.is_absolute():
        return str(resolved_path)
    return str(Path(__file__).resolve().parents[3] / "assets" / resolved_path)


# Build one distinct prim-safe table key from a URDF path.
def _table_key_from_urdf(table_urdf: str) -> str:
    """Return a stable prim-safe key for one table URDF path."""
    return Path(table_urdf).stem.replace("-", "_")


# Build one object rigid-object cfg for a named DexToolBench object.
def _build_object_cfg(
    object_name: str,
    *,
    prim_suffix: str,
    table_reset_z: float,
    table_object_z_offset: float,
) -> RigidObjectCfg:
    """Return a RigidObjectCfg for one named manipulation object."""
    obj_info = NAME_TO_OBJECT[object_name]
    collider_type = "convex_decomposition" if obj_info.need_vhacd else "convex_hull"
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_suffix}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(obj_info.urdf_path),
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            joint_drive=None,
            collider_type=collider_type,
            replace_cylinders_with_capsules=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, table_reset_z + table_object_z_offset),
        ),
    )


# Build one table rigid-object cfg for a specific table URDF.
def _build_table_cfg(table_urdf: str, *, prim_suffix: str, table_reset_z: float) -> RigidObjectCfg:
    """Return a RigidObjectCfg for one table URDF used by the live LLM runtime."""
    table_cfg = copy.deepcopy(TABLE_CFG)
    table_cfg.prim_path = f"/World/envs/env_.*/{prim_suffix}"
    table_cfg.spawn.asset_path = _resolve_asset_path(table_urdf)
    table_cfg.init_state.pos = (0.0, 0.0, table_reset_z)
    return table_cfg


# ---------------------------------------------------------------------------
# Default DOF positions
# ---------------------------------------------------------------------------
_KUKA_REST_POS = torch.tensor(
    [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308], dtype=torch.float32
)
_KUKA_REST_POS_HIGHER = torch.tensor(
    [-1.571, 1.571 - math.radians(10), -0.000, 1.376 + math.radians(10), -0.000, 1.485, 1.308],
    dtype=torch.float32,
)

# Fingertip body names for the SHARPA left hand
_LEFT_SHARPA_FINGERTIPS = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
_LEFT_SHARPA_FINGERTIP_OFFSETS = np.array(
    [[0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0]],
    dtype=np.float32,
)
# Legacy Gym uses iiwa14_link_7 as palm reference for observation features.
_PALM_BODY_NAME = "iiwa14_link_7"
_PALM_OFFSET = np.array([-0.00, -0.02, 0.16], dtype=np.float32)

# Obs / state lists (matches legacy env defaults)
_OBS_LIST = [
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
]
_STATE_LIST = [
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "palm_vel",
    "object_rot",
    "object_vel",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
    "closest_keypoint_max_dist",
    "closest_fingertip_dist",
    "lifted_object",
    "progress",
    "successes",
    "reward",
]


class SimToolRealEnv(DirectRLEnv):
    """Isaac Lab port of the SimToolReal dexterous manipulation environment.

    Inherits from ``DirectRLEnv`` which manages the simulation loop,
    scene creation, and the standard gym interface.  Only the task-specific
    methods (_setup_scene, _pre_physics_step, _apply_action,
    _get_observations, _get_rewards, _get_dones, _reset_idx) need to be
    implemented here.
    """

    cfg: SimToolRealEnvCfg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # Seed the trace step counter before the DirectRLEnv base starts stepping telemetry.
    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: Optional[str] = None, **kwargs):
        """Initialise buffers that depend on cfg before super().__init__."""
        self.num_arm_dofs = 7
        self.num_hand_dofs = 22
        self.num_hand_arm_dofs = self.num_arm_dofs + self.num_hand_dofs
        self.num_fingertips = 5
        self._debug_trace_step = 0

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Shorthand aliases populated in _setup_scene / post-init
        self._init_buffers()

    # ------------------------------------------------------------------
    # Scene setup (Phase 1)
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Spawn robot, table, and object; clone to all envs; add lights."""
        # Build physics materials up-front so spawn configs can bind them consistently.
        self._configure_physics_materials_for_parity()
        default_lie_object_names = (
            resolve_lie_object_names(",".join(self.cfg.lie_object_names))
            if self.cfg.lie_object_names
            else []
        )
        self._active_object_name = self.cfg.object_name
        self._preloaded_object_names = list(
            dict.fromkeys(
                (self.cfg.preloaded_object_names or [self.cfg.object_name])
                + default_lie_object_names
            ).keys()
        )
        if self.cfg.object_name not in self._preloaded_object_names:
            self._preloaded_object_names.append(self.cfg.object_name)
        self._preloaded_table_urdfs: List[str] = []
        self._table_urdf_by_name: Dict[str, str] = {}
        for table_urdf in self.cfg.preloaded_table_urdfs or [self.cfg.table_cfg.spawn.asset_path]:
            table_name = _table_key_from_urdf(table_urdf)
            if table_name in self._table_urdf_by_name:
                continue
            self._table_urdf_by_name[table_name] = table_urdf
            self._preloaded_table_urdfs.append(table_urdf)
        self._table_name_by_object_name = {}
        for object_name in self._preloaded_object_names:
            category = OBJECT_NAME_TO_CATEGORY[object_name]
            table_urdf = OBJECT_CATEGORY_TO_TABLE_URDF[category]
            table_name = _table_key_from_urdf(table_urdf)
            if table_name not in self._table_urdf_by_name:
                self._table_urdf_by_name[table_name] = table_urdf
                self._preloaded_table_urdfs.append(table_urdf)
            self._table_name_by_object_name[object_name] = table_name

        # Robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        self._tables_by_name: Dict[str, RigidObject] = {}
        for table_urdf in self._preloaded_table_urdfs:
            table_name = _table_key_from_urdf(table_urdf)
            self._tables_by_name[table_name] = RigidObject(
                _build_table_cfg(
                    table_urdf,
                    prim_suffix=f"Table_{table_name}",
                    table_reset_z=self.cfg.table_reset_z,
                )
            )
        self._objects_by_name: Dict[str, RigidObject] = {}
        for object_name in self._preloaded_object_names:
            self._objects_by_name[object_name] = RigidObject(
                _build_object_cfg(
                    object_name,
                    prim_suffix=f"Object_{object_name}",
                    table_reset_z=self.cfg.table_reset_z,
                    table_object_z_offset=self.cfg.table_object_z_offset,
                )
            )

        # Clone to num_envs environments arranged in a grid (must happen BEFORE registering)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Register assets with the scene AFTER cloning so Isaac Lab initializes them correctly
        self.scene.articulations["robot"] = self.robot
        for table_name, table in self._tables_by_name.items():
            self.scene.rigid_objects[f"table_{table_name}"] = table
        for object_name, obj in self._objects_by_name.items():
            self.scene.rigid_objects[f"object_{object_name}"] = obj
        self._set_active_object_metadata()

        # Dome lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # NOTE: _resolve_body_indices() is called in _init_buffers() (after sim.reset())

    # Configure reusable physics materials for legacy friction parity bindings.
    def _configure_physics_materials_for_parity(self) -> None:
        """Prepare reusable physics material cfg objects for robot/object/table."""
        self._robot_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.robot_friction,
            dynamic_friction=self.cfg.robot_friction,
            restitution=0.0,
        )
        self._fingertip_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.fingertip_friction,
            dynamic_friction=self.cfg.fingertip_friction,
            restitution=0.0,
        )
        self._object_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.object_friction,
            dynamic_friction=self.cfg.object_friction,
            restitution=0.0,
        )
        self._table_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.cfg.table_friction,
            dynamic_friction=self.cfg.table_friction,
            restitution=0.0,
        )

    # Apply robot/object/table/fingertip materials after concrete prims exist in env_0.
    def _apply_collision_material_overrides(self) -> None:
        """Bind per-asset physics materials to match legacy friction behavior."""
        if not _ISAACLAB_AVAILABLE:
            return
        import omni.usd

        robot_mat_path = "/World/PhysicsMaterials/robotMaterial"
        table_mat_path = "/World/PhysicsMaterials/tableMaterial"
        object_mat_path = "/World/PhysicsMaterials/objectMaterial"
        fingertip_mat_path = "/World/PhysicsMaterials/fingertipMaterial"
        self._robot_material_cfg.func(robot_mat_path, self._robot_material_cfg)
        self._table_material_cfg.func(table_mat_path, self._table_material_cfg)
        self._object_material_cfg.func(object_mat_path, self._object_material_cfg)
        self._fingertip_material_cfg.func(fingertip_mat_path, self._fingertip_material_cfg)

        self._debug_material_bind_counts = {
            "robot": 0,
            "table": 0,
            "object": 0,
            "fingertip": 0,
        }
        stage = omni.usd.get_context().get_stage()
        base_prim = "/World/envs/env_0/Robot"
        # Bind robot material per-link collision prim to avoid non-collider root warnings.
        for body_name in self.robot.data.body_names:
            if self._bind_physics_material_if_editable(
                stage=stage,
                prim_path=f"{base_prim}/{body_name}/collisions",
                material_path=robot_mat_path,
                stronger_than_descendants=False,
            ):
                self._debug_material_bind_counts["robot"] += 1
        for table_name in self._tables_by_name:
            root_path = f"/World/envs/env_0/Table_{table_name}"
            table_bound = self._bind_physics_material_if_editable(
                stage=stage,
                prim_path=f"{root_path}/box/collisions",
                material_path=table_mat_path,
                stronger_than_descendants=False,
            )
            if table_bound:
                self._debug_material_bind_counts["table"] += 1
            else:
                self._debug_material_bind_counts["table"] += self._bind_material_under_root(
                    stage=stage,
                    root_path=root_path,
                    material_path=table_mat_path,
                    stronger_than_descendants=False,
                )

        for object_name in self._objects_by_name:
            root_path = f"/World/envs/env_0/Object_{object_name}"
            self._debug_material_bind_counts["object"] += self._bind_material_under_root(
                stage=stage,
                root_path=root_path,
                material_path=object_mat_path,
                stronger_than_descendants=False,
            )
        for body_name in _LEFT_SHARPA_FINGERTIPS:
            if self._bind_physics_material_if_editable(
                stage=stage,
                prim_path=f"{base_prim}/{body_name}/collisions",
                material_path=fingertip_mat_path,
                stronger_than_descendants=True,
            ):
                self._debug_material_bind_counts["fingertip"] += 1

    # Bind a physics material only when the target prim exists and is editable.
    def _bind_physics_material_if_editable(
        self, stage, prim_path: str, material_path: str, stronger_than_descendants: bool
    ) -> bool:
        """Skip binding on missing or instanced prims to avoid noisy USD warnings."""
        from pxr import PhysxSchema, UsdPhysics

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
        has_particle_system = prim.IsA(PhysxSchema.PhysxParticleSystem)
        if not (has_collider or has_deformable_body or has_particle_system):
            return False
        bind_physics_material(
            prim_path,
            material_path,
            stage=stage,
            stronger_than_descendants=stronger_than_descendants,
        )
        return True

    # Recursively bind a material to all collider prims below a root path.
    def _bind_material_under_root(
        self, stage, root_path: str, material_path: str, stronger_than_descendants: bool
    ) -> int:
        """Best-effort fallback for assets whose collider prim paths differ from hardcoded defaults."""
        from pxr import Usd

        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            return 0
        num_bound = 0
        for prim in Usd.PrimRange(root_prim):
            prim_path = str(prim.GetPath())
            if self._bind_physics_material_if_editable(
                stage=stage,
                prim_path=prim_path,
                material_path=material_path,
                stronger_than_descendants=stronger_than_descendants,
            ):
                num_bound += 1
        return num_bound

    # Rebind the env-facing object/table aliases and per-object metadata to the active object.
    def _set_active_object_metadata(self) -> None:
        """Refresh active object/table aliases and keypoint metadata for the current object."""
        env0_object_name = (
            self._env_object_name(0)
            if hasattr(self, "_active_object_indices")
            else self._active_object_name
        )
        self._active_object_name = env0_object_name
        self.obj = self._objects_by_name[env0_object_name]
        active_table_name = self._table_name_by_object_name[env0_object_name]
        self.table = self._tables_by_name[active_table_name]
        self.cfg.object_name = env0_object_name
        if hasattr(self, "object_scales"):
            self._setup_object_info()

    # Return one tensor selecting all env instances in this scene.
    def _scene_env_ids(self) -> Tensor:
        """Return all env ids on the active device."""
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    # Resolve the active object name for one environment id.
    def _env_object_name(self, env_id: int) -> str:
        """Return the currently active object name for one environment."""
        if not hasattr(self, "_active_object_indices"):
            return self._active_object_name
        return self._preloaded_object_name_list[
            int(self._active_object_indices[int(env_id)].item())
        ]

    # Assign one active object to the requested environments and refresh per-env metadata.
    def _assign_active_object_to_env_ids(self, env_ids: Tensor, object_name: str) -> None:
        """Set the active object for the specified env ids."""
        object_index = self._object_name_to_index[object_name]
        self._active_object_indices[env_ids] = object_index
        self._apply_object_info_for_env_ids(env_ids, object_name)
        if (env_ids == 0).any():
            self._set_active_object_metadata()

    # Fill object scales and keypoint offsets for one env subset and object.
    def _apply_object_info_for_env_ids(self, env_ids: Tensor, object_name: str) -> None:
        """Populate object-info buffers for the specified env ids."""
        obj_info = NAME_TO_OBJECT.get(object_name)
        fixed_size = torch.tensor(self.cfg.fixed_size, dtype=torch.float32, device=self.device)
        if obj_info is None:
            scales = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        else:
            scales = torch.tensor(list(obj_info.scale), dtype=torch.float32, device=self.device)
        self.object_scales[env_ids] = scales
        _set_legacy_keypoints_for_env_ids(
            self.object_keypoint_offsets,
            self.object_keypoint_offsets_fixed_size,
            env_ids,
            scales,
            self.cfg.keypoint_scale,
            self.cfg.object_base_size,
            fixed_size,
        )

    # Park inactive preloaded objects/tables away from the workspace and activate the current table.
    def _park_inactive_scene_assets(self, env_ids: Optional[Tensor] = None) -> None:
        """Move inactive objects/tables away from the robot workspace without rebuilding scene assets."""
        selected_env_ids = self._scene_env_ids() if env_ids is None else env_ids
        for table_index, (table_name, table) in enumerate(self._tables_by_name.items()):
            active_env_ids = []
            inactive_env_ids = []
            for env_id in selected_env_ids.detach().cpu().tolist():
                env_object_name = self._env_object_name(env_id)
                active_name = self._table_name_by_object_name[env_object_name]
                (active_env_ids if active_name == table_name else inactive_env_ids).append(env_id)
            if active_env_ids:
                active_ids_t = torch.tensor(active_env_ids, device=self.device, dtype=torch.long)
                active_pose = torch.tensor(
                    [[0.0, 0.0, self.cfg.table_reset_z, 1.0, 0.0, 0.0, 0.0]],
                    device=self.device,
                ).expand(len(active_ids_t), -1)
                table.write_root_pose_to_sim(active_pose, env_ids=active_ids_t)
                table.write_root_velocity_to_sim(
                    torch.zeros((len(active_ids_t), 6), device=self.device), env_ids=active_ids_t
                )
            if inactive_env_ids:
                inactive_ids_t = torch.tensor(
                    inactive_env_ids, device=self.device, dtype=torch.long
                )
                parked_pose = torch.tensor(
                    [[10.0 + 2.0 * table_index, 0.0, self.cfg.table_reset_z, 1.0, 0.0, 0.0, 0.0]],
                    device=self.device,
                ).expand(len(inactive_ids_t), -1)
                table.write_root_pose_to_sim(parked_pose, env_ids=inactive_ids_t)
                table.write_root_velocity_to_sim(
                    torch.zeros((len(inactive_ids_t), 6), device=self.device),
                    env_ids=inactive_ids_t,
                )

        for object_index, (object_name, obj) in enumerate(self._objects_by_name.items()):
            inactive_env_ids = [
                env_id
                for env_id in selected_env_ids.detach().cpu().tolist()
                if self._env_object_name(env_id) != object_name
            ]
            if not inactive_env_ids:
                continue
            inactive_ids_t = torch.tensor(inactive_env_ids, device=self.device, dtype=torch.long)
            parked_pose = torch.tensor(
                [
                    [
                        10.0 + 2.0 * object_index,
                        0.0,
                        self.cfg.table_reset_z + self.cfg.table_object_z_offset,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ],
                device=self.device,
            ).expand(len(inactive_ids_t), -1)
            obj.write_root_pose_to_sim(parked_pose, env_ids=inactive_ids_t)
            obj.write_root_velocity_to_sim(
                torch.zeros((len(inactive_ids_t), 6), device=self.device), env_ids=inactive_ids_t
            )

    # Switch the active preloaded object in-place and optionally reset the env state around it.
    def switch_active_object(
        self,
        object_name: str,
        *,
        object_start_pose: Optional[List[float]] = None,
        reset: bool = True,
    ) -> None:
        """Switch the active object/table pairing without creating a new simulation context."""
        if object_name not in self._objects_by_name:
            raise ValueError(f"Unknown preloaded object `{object_name}`.")
        self._active_object_name = object_name
        if object_start_pose is not None:
            self.cfg.object_start_pose = list(object_start_pose)
        self._assign_active_object_to_env_ids(self._scene_env_ids(), object_name)
        self._set_active_object_metadata()
        self._park_inactive_scene_assets()
        if reset:
            self._reset_idx(self._scene_env_ids())

    def _resolve_body_indices(self) -> None:
        """Cache robot body indices for fingertips and palm (post-scene-build)."""
        body_names = self.robot.data.body_names

        self.fingertip_indices = torch.tensor(
            [body_names.index(n) for n in _LEFT_SHARPA_FINGERTIPS],
            dtype=torch.long,
            device=self.device,
        )

        # Keep strict parity with legacy env: palm reference must be iiwa14_link_7.
        if _PALM_BODY_NAME not in body_names:
            raise ValueError(f"{_PALM_BODY_NAME} not found in robot body names: {body_names}")
        self.palm_index = body_names.index(_PALM_BODY_NAME)

    # Resolve policy-facing legacy joint order to actual Lab articulation joint indices.
    def _resolve_policy_joint_indices(self) -> None:
        """Map legacy action/observation DOF semantics onto Lab joint names."""
        robot_joint_names = list(self.robot.data.joint_names)
        legacy_names = list(JOINT_NAMES_ISAACGYM)

        policy_joint_indices: list[int] = []
        policy_joint_lab_names: list[str] = []
        for legacy_name in legacy_names:
            lab_name = _LEGACY_TO_LAB_HAND_JOINTS.get(legacy_name, legacy_name)
            if lab_name not in robot_joint_names:
                raise ValueError(
                    f"Missing joint for policy mapping: legacy='{legacy_name}', lab='{lab_name}'"
                )
            policy_joint_indices.append(robot_joint_names.index(lab_name))
            policy_joint_lab_names.append(lab_name)

        self.policy_joint_names = legacy_names
        self.policy_joint_lab_names = policy_joint_lab_names
        self.policy_joint_indices = torch.tensor(
            policy_joint_indices, dtype=torch.long, device=self.device
        )
        self.policy_joint_ids = policy_joint_indices

    # ------------------------------------------------------------------
    # Buffer initialisation (called in __init__ after super())
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        """Allocate all per-env state tensors (called after sim.reset() so data is available)."""
        # Resolve body indices now that sim.reset() has initialised all articulation data
        self._resolve_body_indices()
        self._resolve_policy_joint_indices()
        # Apply material bindings once body names and concrete env_0 prims are available.
        self._apply_collision_material_overrides()

        n = self.num_envs
        dev = self.device
        num_dofs = self.num_hand_arm_dofs

        # DOF limits populated from articulation data (available after sim.reset())
        # Keep policy interface parity with legacy fixed limits/constants used for
        # normalization and action scaling in Isaac Gym codepath.
        self.arm_hand_dof_lower_limits = torch.tensor(Q_LOWER_LIMITS_np, device=dev).float()
        self.arm_hand_dof_upper_limits = torch.tensor(Q_UPPER_LIMITS_np, device=dev).float()

        # Control targets
        self.prev_targets = torch.zeros((n, num_dofs), device=dev)
        self.cur_targets = torch.zeros((n, num_dofs), device=dev)
        self.actions = torch.zeros((n, self.cfg.num_actions), device=dev)

        # Success / episode tracking
        self.successes = torch.zeros(n, device=dev)
        self._is_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.near_goal_steps = torch.zeros(n, dtype=torch.int, device=dev)
        self.lifted_object = torch.zeros(n, dtype=torch.bool, device=dev)
        self.closest_keypoint_max_dist = -torch.ones(n, device=dev)
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(n, device=dev)
        self.closest_fingertip_dist = -torch.ones((n, self.num_fingertips), device=dev)
        self.furthest_hand_dist = -torch.ones(n, device=dev)
        self.has_interacted_with_object = torch.zeros(n, dtype=torch.bool, device=dev)

        # Object state (filled each step from sim)
        self.object_pos = torch.zeros((n, 3), device=dev)
        self.object_rot = torch.zeros((n, 4), device=dev)  # xyzw
        self.object_linvel = torch.zeros((n, 3), device=dev)
        self.object_angvel = torch.zeros((n, 3), device=dev)
        self.object_init_state = torch.zeros((n, 7), device=dev)

        # Goal state
        self.goal_pos = torch.zeros((n, 3), device=dev)
        self.goal_rot = torch.zeros((n, 4), device=dev)  # xyzw
        self.goal_pose = torch.zeros((n, 7), device=dev)
        self.goal_states = torch.zeros((n, 13), device=dev)

        # Object scales (from DexToolBench object info)
        self.object_scales = torch.zeros((n, 3), device=dev)
        self.object_scale_noise_multiplier = torch.ones((n, 3), device=dev)

        # Keypoints: match legacy policy inputs (4 keypoints, not all 8 corners).
        self.num_keypoints = 4
        self.obj_keypoint_pos = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.goal_keypoint_pos = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.keypoints_max_dist = torch.zeros(n, device=dev)
        self.keypoints_max_dist_fixed_size = torch.zeros(n, device=dev)
        self.keypoints_rel_palm = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.keypoints_rel_goal = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.observed_keypoints_rel_palm = torch.zeros_like(self.keypoints_rel_palm)
        self.observed_keypoints_rel_goal = torch.zeros_like(self.keypoints_rel_goal)

        # Fixed-size keypoints (for fixed-size keypoint reward)
        self.obj_keypoint_pos_fixed_size = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.goal_keypoint_pos_fixed_size = torch.zeros((n, self.num_keypoints, 3), device=dev)

        # Fingertip positions
        self.fingertip_pos = torch.zeros((n, self.num_fingertips, 3), device=dev)
        self.fingertip_pos_rel_palm = torch.zeros((n, self.num_fingertips, 3), device=dev)
        self.curr_fingertip_distances = torch.zeros((n, self.num_fingertips), device=dev)

        # Palm
        self.palm_center_pos = torch.zeros((n, 3), device=dev)
        self.palm_rot = torch.zeros((n, 4), device=dev)
        self.palm_state = torch.zeros((n, 13), device=dev)

        # Reward buffer (needed for obs)
        self.rew_buf = torch.zeros(n, device=dev)

        # Prev episode tracking
        self.prev_episode_successes = torch.zeros(n, device=dev)

        # Total episode closest keypoint
        self.total_episode_closest_keypoint_max_dist = torch.zeros(n, device=dev)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(n, device=dev)
        self.prev_episode_closest_keypoint_max_dist = 1000.0 * torch.ones(n, device=dev)

        # Curriculum scale (for tyler curriculum — starts at 0)
        self._tyler_curriculum_scale = 0.0
        self.turn_off_extra_obs_scale = 1.0
        self.turn_off_palm_vel_obs_scale = 1.0
        self.turn_off_object_vel_obs_scale = 1.0

        # Fingertip offsets (metres, in local fingertip frame)
        self.fingertip_offsets = (
            torch.from_numpy(_LEFT_SHARPA_FINGERTIP_OFFSETS).to(dev).unsqueeze(0).expand(n, -1, -1)
        )
        self.palm_offset_t = torch.from_numpy(_PALM_OFFSET).to(dev).unsqueeze(0).expand(n, -1)

        # Obs/action delay queues (ring buffers)
        self._init_delay_queues()

        # Object keypoint offsets (will be populated once object info is loaded)
        self.object_keypoint_offsets = torch.zeros((n, self.num_keypoints, 3), device=dev)
        self.object_keypoint_offsets_fixed_size = torch.zeros(
            (n, self.num_keypoints, 3), device=dev
        )

        # Observed object state (possibly delayed/noisy)
        self.observed_object_state = torch.zeros((n, 13), device=dev)
        self.object_state = torch.zeros((n, 13), device=dev)

        # Force/torque perturbation buffers (not used in basic eval but needed for training)
        self.rb_forces = torch.zeros((n, 1, 3), device=dev)  # (n, num_bodies, 3)
        self.rb_torques = torch.zeros((n, 1, 3), device=dev)

        # Debug telemetry buffers for policy-interface parity diagnostics.
        self._debug_last_action_delay_idx = torch.full((n,), -1, device=dev, dtype=torch.long)
        self._debug_last_obs_delay_idx = torch.full((n,), -1, device=dev, dtype=torch.long)
        self._debug_last_object_state_delay_idx = torch.full((n,), -1, device=dev, dtype=torch.long)
        self._debug_last_applied_action = torch.zeros((n, self.cfg.num_actions), device=dev)
        self._debug_last_arm_targets_unclamped = torch.zeros((n, self.num_arm_dofs), device=dev)
        self._debug_last_hand_targets_unclamped = torch.zeros((n, self.num_hand_dofs), device=dev)
        self._debug_last_arm_clamp_ratio = torch.zeros(n, device=dev)
        self._debug_last_hand_clamp_ratio = torch.zeros(n, device=dev)
        self._debug_done_object_z_low = torch.zeros(n, dtype=torch.bool, device=dev)
        self._debug_done_hand_far = torch.zeros(n, dtype=torch.bool, device=dev)
        self._debug_done_dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._debug_done_max_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._debug_done_timeout = torch.zeros(n, dtype=torch.bool, device=dev)

        # Default DOF positions
        rest = _KUKA_REST_POS_HIGHER if self.cfg.start_arm_higher else _KUKA_REST_POS
        hand_arm_default = torch.zeros(num_dofs, device=dev)
        hand_arm_default[: self.num_arm_dofs] = rest.to(dev)
        self.hand_arm_default_dof_pos = hand_arm_default
        self._preloaded_object_name_list = list(self._objects_by_name.keys())
        self._object_name_to_index = {
            object_name: index for index, object_name in enumerate(self._preloaded_object_name_list)
        }
        self._active_object_indices = torch.full(
            (n,),
            self._object_name_to_index[self.cfg.object_name],
            device=dev,
            dtype=torch.long,
        )
        base_seed = int(getattr(self.cfg, "seed", 0) or 0)
        self._lie_rng = random.Random(base_seed + int(self.cfg.lie_sampling_seed))
        self._lie_object_names = resolve_lie_object_names(
            ",".join(self.cfg.lie_object_names) if self.cfg.lie_object_names else None
        )
        self._lie_goal_sequences: List[Optional[Tensor]] = [None for _ in range(n)]
        self._lie_goal_sequence_lengths = torch.zeros(n, device=dev, dtype=torch.long)
        self._set_active_object_metadata()
        self._park_inactive_scene_assets()

    def _init_delay_queues(self) -> None:
        """Initialise ring-buffer delay queues for observations and actions."""
        n = self.num_envs
        dev = self.device
        self.obs_queue = torch.zeros(
            (n, self.cfg.obs_delay_max, self.cfg.num_observations), device=dev
        )
        self.action_queue = torch.zeros(
            (n, self.cfg.action_delay_max, self.cfg.num_actions), device=dev
        )
        self.object_state_queue = torch.zeros((n, self.cfg.object_state_delay_max, 13), device=dev)

    # ------------------------------------------------------------------
    # Physics step (Phase 2)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Apply delayed + smoothed joint position targets before stepping sim."""
        actions = actions.to(self.device)

        # Update action delay queue (ring buffer, index 0 = most recent)
        self.action_queue = self._update_queue(
            self.action_queue,
            actions,
            is_episode_start=self.episode_length_buf <= 1,
        )

        if self.cfg.use_action_delay:
            delay_idx = torch.randint(
                0, self.action_queue.shape[1], (self.num_envs,), device=self.device
            )
            self._debug_last_action_delay_idx = delay_idx.clone()
            actions = self.action_queue[torch.arange(self.num_envs), delay_idx].clone()
        else:
            self._debug_last_action_delay_idx.fill_(-1)

        self.actions = actions.clone()
        self._debug_last_applied_action = actions.clone()

    def _apply_action(self) -> None:
        """Compute joint position targets and send to the articulation."""
        actions = self.actions
        dof_lower = self.arm_hand_dof_lower_limits
        dof_upper = self.arm_hand_dof_upper_limits
        speed = self.cfg.dof_speed_scale
        dt = self.cfg.sim.dt

        # Arm: relative to previous target (matches legacy useRelativeControl=False default)
        targets_arm_unclamped = self.prev_targets[:, :7] + speed * dt * actions[:, :7]
        targets_arm = tensor_clamp(targets_arm_unclamped, dof_lower[:7], dof_upper[:7])
        self._debug_last_arm_targets_unclamped = targets_arm_unclamped.clone()
        arm_clamped = (targets_arm_unclamped < dof_lower[:7]) | (
            targets_arm_unclamped > dof_upper[:7]
        )
        self._debug_last_arm_clamp_ratio = arm_clamped.float().mean(dim=-1)

        # Smooth arm targets
        targets_arm = (
            self.cfg.arm_moving_average * targets_arm
            + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, :7]
        )

        self.cur_targets[:, :7] = targets_arm

        # Hand: scale normalised actions to joint range
        hand_targets_unclamped = scale(
            actions[:, 7 : self.num_hand_arm_dofs],
            dof_lower[7 : self.num_hand_arm_dofs],
            dof_upper[7 : self.num_hand_arm_dofs],
        )
        self._debug_last_hand_targets_unclamped = hand_targets_unclamped.clone()
        hand_targets = hand_targets_unclamped
        hand_targets = (
            self.cfg.hand_moving_average * hand_targets
            + (1.0 - self.cfg.hand_moving_average)
            * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
        )
        hand_clamped = (hand_targets < dof_lower[7 : self.num_hand_arm_dofs]) | (
            hand_targets > dof_upper[7 : self.num_hand_arm_dofs]
        )
        self._debug_last_hand_clamp_ratio = hand_clamped.float().mean(dim=-1)
        hand_targets = tensor_clamp(
            hand_targets,
            dof_lower[7 : self.num_hand_arm_dofs],
            dof_upper[7 : self.num_hand_arm_dofs],
        )
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = hand_targets

        # Send to simulation (Isaac Lab API)
        self.robot.set_joint_position_target(
            self.cur_targets[:, : self.num_hand_arm_dofs],
            joint_ids=self.policy_joint_ids,
        )
        self.prev_targets[:, : self.num_hand_arm_dofs] = self.cur_targets[
            :, : self.num_hand_arm_dofs
        ].clone()

    # ------------------------------------------------------------------
    # Sim buffer update (called before obs/reward)
    # ------------------------------------------------------------------

    def _populate_sim_buffers(self) -> None:
        """Refresh all derived state from the simulator tensors."""
        # Object state from root tensor, gathered per env from the currently active object asset.
        for object_name, obj in self._objects_by_name.items():
            object_index = self._object_name_to_index[object_name]
            env_ids = torch.where(self._active_object_indices == object_index)[0]
            if len(env_ids) == 0:
                continue
            obj_data = obj.data
            self.object_pos[env_ids] = obj_data.root_pos_w[env_ids]
            self.object_rot[env_ids] = quat_wxyz_to_xyzw(obj_data.root_quat_w[env_ids])
            self.object_linvel[env_ids] = obj_data.root_lin_vel_w[env_ids]
            self.object_angvel[env_ids] = obj_data.root_ang_vel_w[env_ids]

        # Palm state
        robot_data = self.robot.data
        body_pos_w = robot_data.body_pos_w  # (N, num_bodies, 3)
        # Isaac Lab exposes quaternions as wxyz; convert to legacy policy convention xyzw.
        body_quat_w = quat_wxyz_to_xyzw(robot_data.body_quat_w)  # (N, num_bodies, 4)
        body_vel_w = robot_data.body_vel_w  # (N, num_bodies, 6) [lin, ang]

        self.palm_rot = body_quat_w[:, self.palm_index]
        palm_pos_raw = body_pos_w[:, self.palm_index]
        self.palm_center_pos = palm_pos_raw + quat_rotate(self.palm_rot, self.palm_offset_t)

        lin_vel = body_vel_w[:, self.palm_index, :3]
        ang_vel = body_vel_w[:, self.palm_index, 3:]
        self.palm_state = torch.cat([palm_pos_raw, self.palm_rot, lin_vel, ang_vel], dim=-1)

        # Fingertip positions (with offsets in local frame)
        ft_pos = body_pos_w[:, self.fingertip_indices]  # (N, 5, 3)
        ft_quat = body_quat_w[:, self.fingertip_indices]  # (N, 5, 4)
        ft_pos_offset = ft_pos + torch.stack(
            [
                quat_rotate(ft_quat[:, i], self.fingertip_offsets[:, i])
                for i in range(self.num_fingertips)
            ],
            dim=1,
        )
        self.fingertip_pos = ft_pos_offset

        # Fingertip distances to object
        obj_repeat = self.object_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
        ft_rel_obj = ft_pos_offset - obj_repeat
        self.curr_fingertip_distances = torch.norm(ft_rel_obj, dim=-1)  # (N, 5)

        # Initialise on first call (where stored value is -1)
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0,
            self.curr_fingertip_distances,
            self.closest_fingertip_dist,
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0,
            self.curr_fingertip_distances[:, 0],
            self.furthest_hand_dist,
        )

        # Stabilize pre-contact object dynamics so eval does not fail from random physics impulses.
        if self.cfg.stabilize_object_pre_contact:
            min_allowed_z = self.cfg.table_reset_z + 0.05
            unstable_mask, self.has_interacted_with_object = compute_pre_contact_stabilization_mask(
                has_interacted=self.has_interacted_with_object,
                curr_fingertip_distances=self.curr_fingertip_distances,
                object_pos=self.object_pos,
                object_init_pos=self.object_init_state[:, :3],
                object_linvel=self.object_linvel,
                object_angvel=self.object_angvel,
                distance_threshold=self.cfg.pre_contact_distance_threshold,
                max_drift=self.cfg.pre_contact_max_drift,
                max_speed=self.cfg.pre_contact_max_speed,
                min_allowed_z=min_allowed_z,
            )
            if unstable_mask.any():
                env_ids = unstable_mask.nonzero(as_tuple=False).squeeze(-1)
                stable_pose = self.object_init_state[env_ids, :7].clone()
                stable_pose[:, 3:7] = quat_xyzw_to_wxyz(stable_pose[:, 3:7])
                self._write_active_object_pose_by_env_ids(
                    env_ids,
                    stable_pose,
                    torch.zeros((len(env_ids), 6), device=self.device),
                )
                self.object_pos[env_ids] = self.object_init_state[env_ids, :3]
                self.object_rot[env_ids] = self.object_init_state[env_ids, 3:7]
                self.object_linvel[env_ids] = 0.0
                self.object_angvel[env_ids] = 0.0
                obj_repeat = self.object_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
                ft_rel_obj = ft_pos_offset - obj_repeat
                self.curr_fingertip_distances = torch.norm(ft_rel_obj, dim=-1)

        # Refresh object states after potential pre-contact stabilization correction.
        self.object_state = torch.cat(
            [self.object_pos, self.object_rot, self.object_linvel, self.object_angvel], dim=-1
        )
        self.object_state_queue = self._update_queue(
            self.object_state_queue,
            self.object_state,
            is_episode_start=self.episode_length_buf <= 1,
        )
        self.observed_object_state = self.object_state.clone()
        if self.cfg.use_object_state_delay_noise:
            delay_idx = torch.randint(
                0, self.object_state_queue.shape[1], (self.num_envs,), device=self.device
            )
            self._debug_last_object_state_delay_idx = delay_idx.clone()
            self.observed_object_state = self.object_state_queue[
                torch.arange(self.num_envs), delay_idx
            ].clone()
            xyz_std = self.cfg.object_state_xyz_noise_std
            rot_deg = self.cfg.object_state_rotation_noise_degrees
            self.observed_object_state[:, :3] += (
                torch.randn_like(self.observed_object_state[:, :3]) * xyz_std
            )
            self.observed_object_state[:, 3:7] = self._perturb_quat(
                self.observed_object_state[:, 3:7],
                math.radians(rot_deg),
            )
        else:
            self._debug_last_object_state_delay_idx.fill_(-1)

        # Fingertip pos relative to palm
        palm_repeat = self.palm_center_pos.unsqueeze(1).expand(-1, self.num_fingertips, -1)
        self.fingertip_pos_rel_palm = ft_pos_offset - palm_repeat

        # Keypoints: rotate offsets by object rotation, add object position
        obj_rot_repeat = self.object_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1)
        kp_offsets = self.object_keypoint_offsets  # (N, K, 3)
        kp_offsets_fixed = self.object_keypoint_offsets_fixed_size

        # Flatten for quat_rotate then reshape
        kp_world = self.object_pos.unsqueeze(1) + torch.stack(
            [
                quat_rotate(obj_rot_repeat[:, i], kp_offsets[:, i])
                for i in range(self.num_keypoints)
            ],
            dim=1,
        )
        kp_world_fixed = self.object_pos.unsqueeze(1) + torch.stack(
            [
                quat_rotate(obj_rot_repeat[:, i], kp_offsets_fixed[:, i])
                for i in range(self.num_keypoints)
            ],
            dim=1,
        )
        self.obj_keypoint_pos = kp_world
        self.obj_keypoint_pos_fixed_size = kp_world_fixed

        # Goal keypoints
        goal_rot_repeat = self.goal_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1)
        goal_kp = self.goal_pos.unsqueeze(1) + torch.stack(
            [
                quat_rotate(goal_rot_repeat[:, i], kp_offsets[:, i])
                for i in range(self.num_keypoints)
            ],
            dim=1,
        )
        goal_kp_fixed = self.goal_pos.unsqueeze(1) + torch.stack(
            [
                quat_rotate(goal_rot_repeat[:, i], kp_offsets_fixed[:, i])
                for i in range(self.num_keypoints)
            ],
            dim=1,
        )
        self.goal_keypoint_pos = goal_kp
        self.goal_keypoint_pos_fixed_size = goal_kp_fixed

        # Keypoints relative to palm and to goal
        palm_kp = self.palm_center_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1)
        self.keypoints_rel_palm = kp_world - palm_kp
        self.keypoints_rel_goal = kp_world - goal_kp
        self.observed_keypoints_rel_palm = self.keypoints_rel_palm.clone()
        self.observed_keypoints_rel_goal = self.keypoints_rel_goal.clone()
        if self.cfg.use_object_state_delay_noise:
            # Match legacy semantics: derive observed keypoints from delayed/noisy observed object pose.
            observed_pos = self.observed_object_state[:, :3]
            observed_rot = self.observed_object_state[:, 3:7]
            observed_rot_repeat = observed_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1)
            observed_kp = observed_pos.unsqueeze(1) + torch.stack(
                [
                    quat_rotate(
                        observed_rot_repeat[:, i],
                        kp_offsets[:, i] * self.object_scale_noise_multiplier,
                    )
                    for i in range(self.num_keypoints)
                ],
                dim=1,
            )
            self.observed_keypoints_rel_goal = observed_kp - goal_kp
            self.observed_keypoints_rel_palm = observed_kp - palm_kp

        # Max keypoint distance to goal
        self.keypoints_max_dist = torch.norm(self.keypoints_rel_goal, dim=-1).max(dim=-1).values
        self.keypoints_max_dist_fixed_size = (
            torch.norm(kp_world_fixed - goal_kp_fixed, dim=-1).max(dim=-1).values
        )

        # Update closest keypoint distance (episode best)
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0,
            self.keypoints_max_dist,
            torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist),
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            torch.minimum(
                self.closest_keypoint_max_dist_fixed_size, self.keypoints_max_dist_fixed_size
            ),
        )

    # ------------------------------------------------------------------
    # Observations (Phase 2)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Build the 140-dim observation vector and asymmetric state vector."""
        self._populate_sim_buffers()

        n = self.num_envs
        num_dofs = self.num_hand_arm_dofs
        dof_pos = self.robot.data.joint_pos[:, self.policy_joint_indices]
        dof_vel = self.robot.data.joint_vel[:, self.policy_joint_indices]

        obs_dict: dict = {}

        # Joint positions (unscaled to [-1, 1])
        obs_dict["joint_pos"] = unscale(
            dof_pos, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
        )
        # Joint velocities + noise
        obs_dict["joint_vel"] = dof_vel + (
            torch.randn_like(dof_vel) * self.cfg.joint_velocity_obs_noise_std
        )
        obs_dict["prev_action_targets"] = self.prev_targets[:, :num_dofs].clone()
        obs_dict["palm_pos"] = self.palm_center_pos
        obs_dict["palm_rot"] = self.palm_rot
        obs_dict["palm_vel"] = self.palm_state[:, 7:13] * self.turn_off_palm_vel_obs_scale

        # Object state — use observed (delayed/noisy) for the policy
        if self.cfg.use_object_state_delay_noise:
            obs_dict["object_rot"] = self.observed_object_state[:, 3:7]
            obs_dict["object_vel"] = (
                self.observed_object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
            )
            # Policy obs uses first 4 keypoints only (matches pretrained policy: 4*3=12 each)
            obs_dict["keypoints_rel_palm"] = self.observed_keypoints_rel_palm[:, :4, :].reshape(
                n, -1
            )
            obs_dict["keypoints_rel_goal"] = self.observed_keypoints_rel_goal[:, :4, :].reshape(
                n, -1
            )
        else:
            obs_dict["object_rot"] = self.object_rot
            obs_dict["object_vel"] = self.object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
            obs_dict["keypoints_rel_palm"] = self.keypoints_rel_palm[:, :4, :].reshape(n, -1)
            obs_dict["keypoints_rel_goal"] = self.keypoints_rel_goal[:, :4, :].reshape(n, -1)

        obs_dict["fingertip_pos_rel_palm"] = self.fingertip_pos_rel_palm.reshape(n, -1)
        obs_dict["object_scales"] = self.object_scales * self.object_scale_noise_multiplier

        # State (critic) observations — clean values
        obs_dict["closest_keypoint_max_dist"] = (
            self.closest_keypoint_max_dist_fixed_size.unsqueeze(-1)
            if self.cfg.fixed_size_keypoint_reward
            else self.closest_keypoint_max_dist.unsqueeze(-1)
        ) * self.turn_off_extra_obs_scale
        obs_dict["closest_fingertip_dist"] = (
            self.closest_fingertip_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        obs_dict["lifted_object"] = (
            self.lifted_object.unsqueeze(-1).float() * self.turn_off_extra_obs_scale
        )
        obs_dict["progress"] = (
            torch.log(self.episode_length_buf / 10.0 + 1).unsqueeze(-1)
            * self.turn_off_extra_obs_scale
        )
        obs_dict["successes"] = (
            torch.log(self.successes + 1).unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        obs_dict["reward"] = 0.01 * self.rew_buf.unsqueeze(-1) * self.turn_off_extra_obs_scale

        # Concatenate policy obs (140 dims)
        policy_obs = torch.cat([obs_dict[k].reshape(n, -1) for k in _OBS_LIST], dim=-1)

        # Clamp
        policy_obs = torch.clamp(
            policy_obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations
        )

        # Update obs delay queue
        self.obs_queue = self._update_queue(
            self.obs_queue,
            policy_obs,
            is_episode_start=self.episode_length_buf <= 1,
        )
        if self.cfg.use_obs_delay:
            delay_idx = torch.randint(
                0, self.obs_queue.shape[1], (self.num_envs,), device=self.device
            )
            self._debug_last_obs_delay_idx = delay_idx.clone()
            policy_obs = self.obs_queue[torch.arange(self.num_envs), delay_idx].clone()
        else:
            self._debug_last_obs_delay_idx.fill_(-1)

        # Concatenate state obs (for asymmetric critic, if needed)
        state_obs = torch.cat([obs_dict[k].reshape(n, -1) for k in _STATE_LIST], dim=-1)
        state_obs = torch.clamp(
            state_obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations
        )

        return {"policy": policy_obs, "critic": state_obs}

    # ------------------------------------------------------------------
    # Rewards (Phase 2)
    # ------------------------------------------------------------------

    def _get_rewards(self) -> Tensor:
        """Compute per-env reward. Returns shape (N,)."""
        # --- Lifting ---
        lifting_rew, lift_bonus_rew, lifted_object_new = compute_lifting_reward(
            object_pos_z=self.object_pos[:, 2],
            object_init_z=self.object_init_state[:, 2],
            lifted_object=self.lifted_object,
            lifting_bonus_threshold=self.cfg.lifting_bonus_threshold,
            lifting_bonus=self.cfg.lifting_bonus,
        )
        self.lifted_object = lifted_object_new

        # --- Fingertip delta ---
        fingertip_delta_rew, _hand_delta = self._distance_delta_rewards(lifted_object_new)

        # --- Keypoint ---
        if self.cfg.fixed_size_keypoint_reward:
            kp_dist = self.keypoints_max_dist_fixed_size
        else:
            kp_dist = self.keypoints_max_dist
        keypoint_rew, self.closest_keypoint_max_dist = compute_keypoint_reward(
            keypoints_max_dist=kp_dist,
            closest_keypoint_max_dist=self.closest_keypoint_max_dist,
            lifted_object=lifted_object_new,
        )

        # --- Near-goal / success ---
        near_goal, self.near_goal_steps, is_success = compute_near_goal_success(
            keypoints_max_dist=kp_dist,
            near_goal_steps=self.near_goal_steps,
            success_tolerance=(
                self.cfg.eval_success_tolerance
                if self.cfg.eval_success_tolerance is not None
                else self.cfg.success_tolerance
            ),
            keypoint_scale=self.cfg.keypoint_scale,
            success_steps=self.cfg.success_steps,
            force_consecutive=self.cfg.force_consecutive_near_goal_steps,
        )
        self.successes += is_success.float()
        self._reset_goal_on_success(is_success)

        # --- Action penalties ---
        # Match legacy penalty terms: depend on current joint velocities, not raw actions.
        kuka_penalty, hand_penalty = compute_action_penalties(
            dof_vel=self.robot.data.joint_vel[:, self.policy_joint_indices],
            num_arm_dofs=self.num_arm_dofs,
            kuka_actions_penalty_scale=self.cfg.kuka_actions_penalty_scale,
            hand_actions_penalty_scale=self.cfg.hand_actions_penalty_scale,
        )

        # --- Object velocity penalties ---
        obj_linvel_penalty = -self.cfg.object_lin_vel_penalty_scale * torch.sum(
            torch.square(self.object_linvel), dim=-1
        )
        obj_angvel_penalty = -self.cfg.object_ang_vel_penalty_scale * torch.sum(
            torch.square(self.object_angvel), dim=-1
        )

        # --- Bonus for being near goal ---
        bonus_rew = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)

        reward = (
            fingertip_delta_rew * self.cfg.distance_delta_rew_scale
            + lifting_rew * self.cfg.lifting_rew_scale
            + lift_bonus_rew
            + keypoint_rew * self.cfg.keypoint_rew_scale
            + kuka_penalty
            + hand_penalty
            + bonus_rew
            + obj_linvel_penalty
            + obj_angvel_penalty
        )

        self.rew_buf = reward
        self._is_success = is_success  # cache for _get_dones
        # Keep DirectRLEnv extras populated each step so rl_games observers can log completed-episode success stats.
        self.extras = self._get_extras()
        return reward

    def _distance_delta_rewards(self, lifted_object: Tensor):
        """Fingertip approach reward (only before object is lifted)."""
        delta_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(
            self.closest_fingertip_dist, self.curr_fingertip_distances
        )
        delta_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(
            self.furthest_hand_dist, self.curr_fingertip_distances[:, 0]
        )

        fingertip_rew = torch.clip(delta_closest, 0, 10).sum(dim=-1) * (~lifted_object).float()
        hand_penalty = torch.clip(delta_furthest, -10, 0) * (~lifted_object).float()
        return fingertip_rew, hand_penalty

    # ------------------------------------------------------------------
    # Done / reset (Phase 2)
    # ------------------------------------------------------------------

    # Record raw termination booleans before Isaac Lab dispatches any automatic reset.
    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """Compute per-env terminated and truncated flags."""
        object_z_low = self.object_pos[:, 2] < 0.1
        hand_far = self.curr_fingertip_distances.max(dim=-1).values > 1.5
        if self.cfg.reset_when_dropped:
            dropped = (self.object_pos[:, 2] < self.object_init_state[:, 2]) & self.lifted_object
        else:
            dropped = torch.zeros_like(object_z_low)
        max_consecutive_successes = self._resolved_max_consecutive_successes_tensor()
        max_success = (max_consecutive_successes > 0) & (
            self.successes >= max_consecutive_successes.float()
        )
        timeout = self.episode_length_buf >= self.max_episode_length
        self._debug_done_object_z_low = object_z_low
        self._debug_done_hand_far = hand_far
        self._debug_done_dropped = dropped
        self._debug_done_max_success = max_success
        self._debug_done_timeout = timeout

        # Match legacy Gym reset criteria.
        terminated, truncated = compute_termination_and_truncation(
            object_pos_z=self.object_pos[:, 2],
            curr_fingertip_distances=self.curr_fingertip_distances,
            lifted_object=self.lifted_object,
            object_init_z=self.object_init_state[:, 2],
            episode_length_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            successes=self.successes,
            is_success=self._is_success,
            max_consecutive_successes=max_consecutive_successes,
            reset_when_dropped=self.cfg.reset_when_dropped,
        )
        # Trace reset booleans before Isaac Lab applies reset_idx so LLM debugging sees the cause.
        self._write_debug_trace_event(
            "get_dones_eval",
            {
                "step": self._debug_trace_step,
                "episode_length_buf": int(self.episode_length_buf[0].item()),
                "successes": float(self.successes[0].item()),
                "is_success": bool(self._is_success[0].item()),
                "object_z_low": bool(object_z_low[0].item()),
                "hand_far": bool(hand_far[0].item()),
                "dropped": bool(dropped[0].item()),
                "max_success": bool(max_success[0].item()),
                "timeout": bool(timeout[0].item()),
                "raw_terminated": bool(terminated[0].item()),
                "raw_truncated": bool(truncated[0].item()),
                "object_pos": self.object_pos[0].tolist(),
                "max_fingertip_distance": float(self.curr_fingertip_distances[0].max().item()),
                "force_no_reset": bool(self.cfg.force_no_reset),
                "max_consecutive_successes": int(max_consecutive_successes[0].item()),
            },
        )
        if self.cfg.force_no_reset:
            terminated = torch.zeros_like(terminated)
            truncated = torch.zeros_like(truncated)
        self._write_debug_trace_event(
            "get_dones_returned",
            {
                "step": self._debug_trace_step,
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
            },
        )
        self._debug_trace_step += 1
        return terminated, truncated

    # Log every hard env reset entry so runner logs can be matched to the low-level reset path.
    def _reset_idx(self, env_ids: Tensor) -> None:
        """Reset specified environments to initial conditions."""
        if len(env_ids) == 0:
            return

        # Trace reset_idx entry before robot/object state is rewritten back to spawn.
        self._write_debug_trace_event(
            "reset_idx_called",
            {
                "step": self._debug_trace_step,
                "env_ids": env_ids.detach().cpu().tolist(),
                "episode_length_buf": int(self.episode_length_buf[0].item()),
                "successes": float(self.successes[0].item()),
                "done_reasons": {
                    "object_z_low": bool(self._debug_done_object_z_low[0].item()),
                    "hand_far": bool(self._debug_done_hand_far[0].item()),
                    "dropped": bool(self._debug_done_dropped[0].item()),
                    "max_success": bool(self._debug_done_max_success[0].item()),
                    "timeout": bool(self._debug_done_timeout[0].item()),
                },
            },
        )

        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        dev = self.device

        # --- Reset episode tracking ---
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.has_interacted_with_object[env_ids] = False
        self.rew_buf[env_ids] = 0.0

        # --- Reset robot DOFs ---
        dof_lower = self.arm_hand_dof_lower_limits
        dof_upper = self.arm_hand_dof_upper_limits
        default_pos = self.hand_arm_default_dof_pos

        delta_max = dof_upper - default_pos
        delta_min = dof_lower - default_pos
        rand_f = torch_rand_float(0.0, 1.0, (n_reset, self.num_hand_arm_dofs), device=dev)
        rand_delta = delta_min + (delta_max - delta_min) * rand_f

        noise_coeff = torch.zeros(self.num_hand_arm_dofs, device=dev)
        noise_coeff[:7] = self.cfg.reset_dof_pos_noise_arm
        noise_coeff[7:] = self.cfg.reset_dof_pos_noise_fingers

        robot_pos = tensor_clamp(
            default_pos + noise_coeff * rand_delta,
            dof_lower,
            dof_upper,
        )

        # Velocity noise
        rand_vel = torch_rand_float(-1.0, 1.0, (n_reset, self.num_hand_arm_dofs), device=dev)
        robot_vel = self.cfg.reset_dof_vel_noise * rand_vel

        self.robot.write_joint_state_to_sim(
            robot_pos,
            robot_vel,
            joint_ids=self.policy_joint_ids,
            env_ids=env_ids,
        )
        self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
        self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos

        # --- Reset object pose ---
        if self.cfg.goal_source == "lie":
            sampled_object_name, lie_artifact = self._sample_lie_reset_assignment()
            self._assign_active_object_to_env_ids(env_ids, sampled_object_name)
            goals_tensor = torch.tensor(lie_artifact["goals"], device=dev, dtype=torch.float32)
            for env_id in env_ids.detach().cpu().tolist():
                self._lie_goal_sequences[int(env_id)] = goals_tensor
            self._lie_goal_sequence_lengths[env_ids] = goals_tensor.shape[0]
        self._park_inactive_scene_assets(env_ids)
        obj_pos, obj_rot = self._sample_object_pose(n_reset, dev)
        # Isaac Lab write API expects quaternions in wxyz.
        obj_pose = torch.cat([obj_pos, quat_xyzw_to_wxyz(obj_rot)], dim=-1)  # (n_reset, 7)
        self._write_active_object_pose_by_env_ids(
            env_ids,
            obj_pose,
            torch.zeros((n_reset, 6), device=dev),
        )
        self.object_init_state[env_ids, :3] = obj_pos
        self.object_init_state[env_ids, 3:7] = obj_rot

        # --- Reset goal ---
        self._reset_goal(env_ids, obj_pos, obj_rot)

        # Flush delay queues for reset envs
        self.obs_queue[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self.object_state_queue[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    # Write active-object root pose and velocity for env ids grouped by object identity.
    def _write_active_object_pose_by_env_ids(
        self, env_ids: Tensor, pose_wxyz: Tensor, velocity: Tensor
    ) -> None:
        """Write root pose/velocity to the active object asset for each env id."""
        for object_name, object_index in self._object_name_to_index.items():
            object_env_ids = env_ids[self._active_object_indices[env_ids] == object_index]
            if len(object_env_ids) == 0:
                continue
            selection_mask = torch.isin(env_ids, object_env_ids)
            self._objects_by_name[object_name].write_root_pose_to_sim(
                pose_wxyz[selection_mask], env_ids=object_env_ids
            )
            self._objects_by_name[object_name].write_root_velocity_to_sim(
                velocity[selection_mask], env_ids=object_env_ids
            )

    # Return the active per-env success cap tensor used by Lie trajectories and fixed-goal modes.
    def _resolved_max_consecutive_successes_tensor(self) -> Tensor:
        """Return the current success cap for each environment."""
        if getattr(self.cfg, "goal_source", "delta") == "lie":
            return self._lie_goal_sequence_lengths.clone()
        return torch.full_like(
            self.successes,
            int(getattr(self, "max_consecutive_successes", self.cfg.max_consecutive_successes)),
            dtype=torch.long,
        )

    # Sample one Lie-training object and trajectory artifact for the next episode batch.
    def _sample_lie_reset_assignment(self) -> tuple[str, dict]:
        """Return one object name plus compiled Lie artifact for a reset batch."""
        if self.cfg.lie_object_sampling != "uniform":
            raise ValueError(f"Unsupported lie_object_sampling `{self.cfg.lie_object_sampling}`.")
        object_name = self._lie_rng.choice(self._lie_object_names)
        return object_name, build_random_lie_artifact(object_name, rng=self._lie_rng)

    def _sample_object_pose(self, n: int, device: str) -> tuple[Tensor, Tensor]:
        """Sample random initial object position and rotation."""
        if self.cfg.use_fixed_init_object_pose and self.cfg.object_start_pose is not None:
            pose = torch.tensor(self.cfg.object_start_pose, device=device, dtype=torch.float32)
            pos = pose[:3].unsqueeze(0).expand(n, -1).clone()
            rot = pose[3:7].unsqueeze(0).expand(n, -1).clone()
            return pos, rot

        table_z = self.cfg.table_reset_z + self.cfg.table_object_z_offset
        pos = torch.zeros((n, 3), device=device)
        pos[:, 0] = torch_rand_float(
            -self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_x, (n, 1), device=device
        ).squeeze(-1)
        pos[:, 1] = torch_rand_float(
            -self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_y, (n, 1), device=device
        ).squeeze(-1)
        pos[:, 2] = table_z + torch_rand_float(
            -self.cfg.reset_position_noise_z, self.cfg.reset_position_noise_z, (n, 1), device=device
        ).squeeze(-1)

        if self.cfg.randomize_object_rotation:
            rot = _sample_random_quat(n, device)
        else:
            # Identity quaternion (xyzw)
            rot = torch.zeros((n, 4), device=device)
            rot[:, 3] = 1.0

        return pos, rot

    # Load fixed-goal trajectories from JSON on first use so Lab matches legacy fixedGoalStatesJsonPath support.
    def _ensure_fixed_goal_states_loaded(self) -> None:
        """Load fixed goal states from JSON once when the config provides only a file path."""
        if self.cfg.goal_source != "fixed_json" and not self.cfg.use_fixed_goal_states:
            return
        if self.cfg.fixed_goal_states is not None:
            return
        if not self.cfg.fixed_goal_states_json_path:
            return
        with open(self.cfg.fixed_goal_states_json_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        loaded_goals = payload.get("goals") if isinstance(payload, dict) else None
        if not isinstance(loaded_goals, list) or len(loaded_goals) == 0:
            raise ValueError(
                f"Expected non-empty goals list in {self.cfg.fixed_goal_states_json_path}."
            )
        self.cfg.fixed_goal_states = loaded_goals
        self.max_consecutive_successes = len(loaded_goals)

    def _reset_goal(self, env_ids: Tensor, obj_pos: Tensor, obj_rot: Tensor) -> None:
        """Sample and set goal pose for the given environments."""
        n = len(env_ids)
        dev = self.device

        self._ensure_fixed_goal_states_loaded()
        if self.cfg.goal_source == "lie":
            goal_pos = torch.zeros((n, 3), device=dev)
            goal_rot = torch.zeros((n, 4), device=dev)
            for local_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
                goal_sequence = self._lie_goal_sequences[int(env_id)]
                if goal_sequence is None or len(goal_sequence) == 0:
                    raise ValueError(f"Missing Lie goal sequence for env {env_id}.")
                goal_index = int(self.successes[env_id].item()) % int(goal_sequence.shape[0])
                goal_pos[local_index] = goal_sequence[goal_index, :3]
                goal_rot[local_index] = goal_sequence[goal_index, 3:7]
            self.goal_pos[env_ids] = goal_pos
            self.goal_rot[env_ids] = goal_rot
        elif (
            self.cfg.goal_source == "fixed_json" or self.cfg.use_fixed_goal_states
        ) and self.cfg.fixed_goal_states is not None:
            # Cycle through the provided goal states
            goals = self.cfg.fixed_goal_states
            idx = (self.successes[env_ids].long() % len(goals)).cpu().tolist()
            goal_data = torch.tensor(
                [goals[i] for i in idx], device=dev, dtype=torch.float32
            )  # (n, 7)
            self.goal_pos[env_ids] = goal_data[:, :3]
            self.goal_rot[env_ids] = goal_data[:, 3:7]
        elif self.cfg.goal_sampling_type == "delta":
            # Random delta translation + rotation around current object pose
            rand_dir = torch.nn.functional.normalize(torch.randn((n, 3), device=dev), dim=-1)
            goal_pos = obj_pos + rand_dir * self.cfg.delta_goal_distance
            # Random rotation delta
            axis = torch.nn.functional.normalize(torch.randn((n, 3), device=dev), dim=-1)
            angle = torch.rand(n, device=dev) * math.radians(self.cfg.delta_rotation_degrees)
            delta_rot = _axis_angle_to_quat_xyzw(axis, angle)
            goal_rot = _quat_mul_xyzw(delta_rot, obj_rot)
            self.goal_pos[env_ids] = goal_pos
            self.goal_rot[env_ids] = goal_rot
        else:
            # Absolute: sample from target volume
            vol_min = torch.tensor(self.cfg.target_volume_mins, device=dev)
            vol_max = torch.tensor(self.cfg.target_volume_maxs, device=dev)
            goal_pos = vol_min + torch.rand((n, 3), device=dev) * (vol_max - vol_min)
            goal_rot = _sample_random_quat(n, dev)
            self.goal_pos[env_ids] = goal_pos
            self.goal_rot[env_ids] = goal_rot

        self.goal_pose[env_ids] = torch.cat(
            [self.goal_pos[env_ids], self.goal_rot[env_ids]], dim=-1
        )
        self.goal_states[env_ids, :7] = self.goal_pose[env_ids]

    def _reset_goal_on_success(self, is_success: Tensor) -> None:
        """Reset only goal state on success, mirroring legacy `reset_goal_buf` flow."""
        success_env_ids = torch.where(is_success)[0]
        if len(success_env_ids) == 0:
            return
        self._reset_goal(
            success_env_ids,
            self.object_pos[success_env_ids].clone(),
            self.object_rot[success_env_ids].clone(),
        )
        self.near_goal_steps[success_env_ids] = 0
        self.closest_keypoint_max_dist[success_env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[success_env_ids] = -1.0
        # Legacy behavior resets progress when max_consecutive_successes > 0.
        success_caps = self._resolved_max_consecutive_successes_tensor()[success_env_ids]
        if torch.any(success_caps > 0):
            self.episode_length_buf[success_env_ids] = 0

    # Resolve the active success cap from live overrides before falling back to static config.
    def _resolved_max_consecutive_successes(self) -> int:
        """Return the current success cap used by done handling and goal advancement."""
        return int(getattr(self, "max_consecutive_successes", self.cfg.max_consecutive_successes))

    def _perturb_quat(self, quat_xyzw: Tensor, max_angle_rad: float) -> Tensor:
        """Add small random rotation to a quaternion batch."""
        n = quat_xyzw.shape[0]
        axis = torch.nn.functional.normalize(torch.randn((n, 3), device=self.device), dim=-1)
        angle = torch.rand(n, device=self.device) * max_angle_rad
        delta = _axis_angle_to_quat_xyzw(axis, angle)
        return _quat_mul_xyzw(delta, quat_xyzw)

    @staticmethod
    def _update_queue(queue: Tensor, current: Tensor, is_episode_start: Tensor) -> Tensor:
        """Update queue and fill it on episode start to match Gym delay behavior."""
        # Keep method for compatibility but delegate to pure helper used by tests.
        return update_delay_queue(queue, current, is_episode_start)

    def _setup_dof_limits(self) -> None:
        """Read DOF position limits from the articulation after scene build."""
        # Keep legacy policy-interface constants regardless of simulator soft limits.
        self.arm_hand_dof_lower_limits = torch.tensor(Q_LOWER_LIMITS_np, device=self.device).float()
        self.arm_hand_dof_upper_limits = torch.tensor(Q_UPPER_LIMITS_np, device=self.device).float()

    # Append one JSONL debug event when the Isaac Lab env trace path is enabled.
    def _write_debug_trace_event(self, event: str, payload: dict) -> None:
        """Append one structured Isaac Lab debug trace event when tracing is enabled."""
        if not self.cfg.debug_trace_path:
            return
        trace_path = Path(self.cfg.debug_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")

    def _setup_object_info(self) -> None:
        """Load object scales and keypoint offsets from DexToolBench."""
        self._apply_object_info_for_env_ids(self._scene_env_ids(), self.cfg.object_name)

    # Export runtime/state diagnostics used by eval and parity tests.
    def get_runtime_snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of sim/policy runtime parity signals."""
        self._populate_sim_buffers()

        def _tensor_to_list(t: Optional[Tensor], max_items: Optional[int] = None):
            if t is None:
                return None
            if not isinstance(t, torch.Tensor):
                return t
            data = t.detach().cpu()
            if max_items is not None:
                data = data.reshape(-1)[:max_items]
            return data.tolist()

        env0_object_name = self._env_object_name(0)
        obj_info = NAME_TO_OBJECT.get(env0_object_name)
        physx = self.cfg.sim.physx
        robot_default_mass = getattr(self.robot.data, "default_mass", None)
        robot_default_inertia = getattr(self.robot.data, "default_inertia", None)
        env0_object = self._objects_by_name[env0_object_name]
        object_default_mass = getattr(env0_object.data, "default_mass", None)
        object_default_inertia = getattr(env0_object.data, "default_inertia", None)
        arm_actuator = self.cfg.robot_cfg.actuators.get("arm_joints", None)
        hand_actuator = self.cfg.robot_cfg.actuators.get("hand_joints", None)

        return {
            "backend": "isaac_lab",
            "sim": {
                "dt": float(self.cfg.sim.dt),
                "decimation": int(self.cfg.decimation),
                "control_hz": float(1.0 / (self.cfg.sim.dt * self.cfg.decimation)),
                "physx": {
                    "solver_type": int(physx.solver_type),
                    "position_iterations": int(physx.max_position_iteration_count),
                    "velocity_iterations": int(physx.max_velocity_iteration_count),
                    "bounce_threshold_velocity": float(physx.bounce_threshold_velocity),
                    "friction_offset_threshold": float(physx.friction_offset_threshold),
                    "friction_correlation_distance": float(physx.friction_correlation_distance),
                    "gpu_max_rigid_contact_count": int(physx.gpu_max_rigid_contact_count),
                    "gpu_max_rigid_patch_count": int(physx.gpu_max_rigid_patch_count),
                    "gpu_found_lost_pairs_capacity": int(physx.gpu_found_lost_pairs_capacity),
                    "gpu_heap_capacity": int(physx.gpu_heap_capacity),
                    "gpu_temp_buffer_capacity": int(physx.gpu_temp_buffer_capacity),
                },
            },
            "policy_interface": {
                "num_observations": int(self.cfg.num_observations),
                "num_actions": int(self.cfg.num_actions),
                "policy_joint_names": list(self.policy_joint_names),
                "policy_joint_lab_names": list(self.policy_joint_lab_names),
                "dof_lower_limits": _tensor_to_list(self.arm_hand_dof_lower_limits),
                "dof_upper_limits": _tensor_to_list(self.arm_hand_dof_upper_limits),
            },
            "materials": {
                "bind_counts": dict(getattr(self, "_debug_material_bind_counts", {})),
                "robot_friction": float(self.cfg.robot_friction),
                "fingertip_friction": float(self.cfg.fingertip_friction),
                "object_friction": float(self.cfg.object_friction),
                "table_friction": float(self.cfg.table_friction),
            },
            "robot": {
                "num_bodies": int(len(self.robot.data.body_names)),
                "num_joints": int(len(self.robot.data.joint_names)),
                "body_names": list(self.robot.data.body_names),
                "joint_names": list(self.robot.data.joint_names),
                "default_mass_env0": _tensor_to_list(
                    robot_default_mass[0] if isinstance(robot_default_mass, torch.Tensor) else None
                ),
                "default_inertia_env0": _tensor_to_list(
                    robot_default_inertia[0]
                    if isinstance(robot_default_inertia, torch.Tensor)
                    else None
                ),
                "arm_stiffness_cfg": dict(getattr(arm_actuator, "stiffness", {}) or {}),
                "arm_damping_cfg": dict(getattr(arm_actuator, "damping", {}) or {}),
                "hand_stiffness_cfg": dict(getattr(hand_actuator, "stiffness", {}) or {}),
                "hand_damping_cfg": dict(getattr(hand_actuator, "damping", {}) or {}),
            },
            "object": {
                "name": env0_object_name,
                "need_vhacd": bool(getattr(obj_info, "need_vhacd", False)) if obj_info else False,
                "object_scale_cfg": _tensor_to_list(self.object_scales[0]),
                "default_mass_env0": _tensor_to_list(
                    object_default_mass[0]
                    if isinstance(object_default_mass, torch.Tensor)
                    else None
                ),
                "default_inertia_env0": _tensor_to_list(
                    object_default_inertia[0]
                    if isinstance(object_default_inertia, torch.Tensor)
                    else None
                ),
            },
            "runtime_state_env0": {
                "object_pose": _tensor_to_list(
                    torch.cat([self.object_pos[0], self.object_rot[0]], dim=-1)
                ),
                "goal_pose": _tensor_to_list(self.goal_pose[0]),
                "object_linvel": _tensor_to_list(self.object_linvel[0]),
                "object_angvel": _tensor_to_list(self.object_angvel[0]),
                "fingertip_min_dist": float(self.curr_fingertip_distances[0].min().item()),
                "fingertip_max_dist": float(self.curr_fingertip_distances[0].max().item()),
                "has_interacted_with_object": bool(self.has_interacted_with_object[0].item()),
            },
        }

    # ------------------------------------------------------------------
    # Public extras (for rl_games)
    # ------------------------------------------------------------------

    def _get_extras(self) -> dict:
        """Return extra info dict for rl_games logging."""
        return {
            "successes": self.prev_episode_successes,
            "success_ratio": (
                self.prev_episode_successes.mean().item()
                / max(self.cfg.max_consecutive_successes, 1)
            ),
            "live_successes_mean": float(self.successes.mean().item()),
            "live_successes_max": float(self.successes.max().item()),
            "live_reward_mean": float(self.rew_buf.mean().item()),
            "live_reward_max": float(self.rew_buf.max().item()),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _legacy_keypoint_signs(device: torch.device) -> Tensor:
    """Return legacy 4-keypoint signs used by the pretrained SimToolReal policy."""
    signs = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return signs


def _set_legacy_keypoints(
    kp_buf: Tensor,
    kp_fixed_buf: Tensor,
    object_scales: Tensor,
    keypoint_scale: float,
    object_base_size: float,
    fixed_size: Tensor,
) -> None:
    """Fill keypoint offsets using legacy SimToolReal scaling conventions."""
    signs = _legacy_keypoint_signs(kp_buf.device)
    legacy_half = object_scales * object_base_size * keypoint_scale * 0.5
    fixed_half = fixed_size * keypoint_scale * 0.5
    kp_buf[:] = (signs * legacy_half.unsqueeze(0)).unsqueeze(0)
    kp_fixed_buf[:] = (signs * fixed_half.unsqueeze(0)).unsqueeze(0)


# Fill keypoint offsets for only the selected env ids while preserving the shared legacy convention.
def _set_legacy_keypoints_for_env_ids(
    kp_buf: Tensor,
    kp_fixed_buf: Tensor,
    env_ids: Tensor,
    object_scales: Tensor,
    keypoint_scale: float,
    object_base_size: float,
    fixed_size: Tensor,
) -> None:
    """Fill the keypoint buffers for one env subset."""
    signs = _legacy_keypoint_signs(kp_buf.device)
    legacy_half = object_scales * object_base_size * keypoint_scale * 0.5
    fixed_half = fixed_size * keypoint_scale * 0.5
    kp_values = (signs * legacy_half.unsqueeze(0)).unsqueeze(0).expand(len(env_ids), -1, -1)
    kp_fixed_values = (signs * fixed_half.unsqueeze(0)).unsqueeze(0).expand(len(env_ids), -1, -1)
    kp_buf[env_ids] = kp_values
    kp_fixed_buf[env_ids] = kp_fixed_values
