# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import torch
import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from . import mdp
from tasks.common_config import H12RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH

# Task target: pick the cylinder and place it 1 ft (0.3048 m) to the +x direction.
_OBJECT_INIT_X = -2.58514
_OBJECT_INIT_Y = -2.78975
_ONE_FOOT_M = 0.3048
_TARGET_X = _OBJECT_INIT_X + _ONE_FOOT_M
_TARGET_Y = _OBJECT_INIT_Y
_TARGET_Z = 0.855

# Workspace and goal-zone bounds for termination/reward.
_MIN_X = _OBJECT_INIT_X - 0.20
_MAX_X = _TARGET_X + 0.25
_MIN_Y = _OBJECT_INIT_Y - 0.25
_MAX_Y = _OBJECT_INIT_Y + 0.25
_MIN_H = 0.5
_POST_HALF_X = 0.08
_POST_HALF_Y = 0.10
_POST_MIN_X = _TARGET_X - _POST_HALF_X
_POST_MAX_X = _TARGET_X + _POST_HALF_X
_POST_MIN_Y = _TARGET_Y - _POST_HALF_Y
_POST_MAX_Y = _TARGET_Y + _POST_HALF_Y
_POST_MIN_H = 0.81
_POST_MAX_H = 0.9

@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfgWH):
    """Object-table scene config for H1-2 wholebody move task."""

    # Use stable base-fixed H1-2 for teleop/task debugging.
    # Floating-base H1-2 can collapse without a balancing controller/policy.
    robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_wholebody(
        init_pos=(-3.9, -2.81811, 1.00),
        init_rot=(1, 0, 0, 0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.h12_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()
    robot_camera = CameraPresets.h12_world_camera()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # Full-body control: expose all articulation joints to the policy.
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states, params={"enable_dds": False})
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states, params={"enable_dds": False})
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_workspace = DoneTerm(
        func=mdp.reset_object_estimate,
        params={
            "min_x": _MIN_X,
            "max_x": _MAX_X,
            "min_y": _MIN_Y,
            "max_y": _MAX_Y,
            "min_height": _MIN_H,
        },
    )


@configclass
class RewardsCfg:
    reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={
            # Disable DDS reward publishing for RL training runs.
            "enable_dds": False,
            # Goal: place 1 ft to the right (+x) from the object default location.
            "min_x": _MIN_X,
            "max_x": _MAX_X,
            "min_y": _MIN_Y,
            "max_y": _MAX_Y,
            "min_height": _MIN_H,
            "post_min_x": _POST_MIN_X,
            "post_max_x": _POST_MAX_X,
            "post_min_y": _POST_MIN_Y,
            "post_max_y": _POST_MAX_Y,
            "post_min_height": _POST_MIN_H,
            "post_max_height": _POST_MAX_H,
            "target_x": _TARGET_X,
            "target_y": _TARGET_Y,
            "target_z": _TARGET_Z,
            "dense_xy_weight": 0.4,
            "dense_z_weight": 0.2,
            "dense_xy_scale": 4.0,
            "dense_z_scale": 10.0,
        },
    )


@configclass
class EventCfg:
    pass


@configclass
class MoveCylinderH1227dofInspireWholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

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
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
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
