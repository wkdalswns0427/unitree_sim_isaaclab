# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tasks.common_config import CameraPresets, H12RobotPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.h12_27dof_state import get_robot_boy_joint_states
from tasks.common_observations.inspire_state import get_robot_inspire_joint_states


_DEFAULT_WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
_WAREHOUSE_USD = os.path.expanduser(os.environ.get("UNITREE_WAREHOUSE_USD", _DEFAULT_WAREHOUSE_USD))
_ROBOT_USD_OVERRIDE = os.environ.get("UNITREE_H12_USD")


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = H12RobotPresets.h12_27dof_inspire_wholebody_floating(
        init_pos=(0.0, 0.0, 1.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )
    if _ROBOT_USD_OVERRIDE:
        robot_usd = os.path.abspath(os.path.expanduser(_ROBOT_USD_OVERRIDE))
        if not os.path.exists(robot_usd):
            raise FileNotFoundError(f"UNITREE_H12_USD does not exist: {robot_usd}")
        robot_cfg = robot_cfg.replace(
            spawn=robot_cfg.spawn.replace(usd_path=robot_usd),
        )
    return robot_cfg


def zero_reward(env, *args, **kwargs):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    warehouse = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Warehouse",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_WAREHOUSE_USD,
        ),
    )

    robot: ArticulationCfg = _make_robot_cfg()

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.h12_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera_h12_floating()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera_h12_floating()
    robot_camera = CameraPresets.h12_world_camera()

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states, params={"enable_dds": False})
        robot_inspire_state = ObsTerm(func=get_robot_inspire_joint_states, params={"enable_dds": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.0})
    root_too_low = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class RewardsCfg:
    alive = RewTerm(func=zero_reward, weight=1.0)


@configclass
class WarehouseWalkH12InspireWholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: WarehouseSceneCfg = WarehouseSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None
    events = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )
