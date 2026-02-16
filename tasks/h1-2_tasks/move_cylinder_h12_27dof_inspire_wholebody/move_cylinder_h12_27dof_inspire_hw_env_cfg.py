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

@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfgWH):
    """Object-table scene config for H1-2 wholebody move task."""

    # Use stable base-fixed H1-2 for teleop/task debugging.
    # Floating-base H1-2 can collapse without a balancing controller/policy.
    robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_wholebody(
        init_pos=(-3.9, -2.81811, 0.80),
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_workspace = DoneTerm(
        func=mdp.reset_object_estimate,
        params={
            # Keep object inside a translated version of the default move-cylinder workspace.
            # The wholebody scene object starts near (-2.585, -2.790, 0.84).
            "min_x": -2.66,
            "max_x": -1.24,
            "min_y": -2.99,
            "max_y": -2.49,
            "min_height": 0.5,
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
            # Wholebody scene is translated far from origin, so align workspace and target bounds.
            "min_x": -2.66,
            "max_x": -1.24,
            "min_y": -2.99,
            "max_y": -2.49,
            "min_height": 0.5,
            "post_min_x": -1.96,
            "post_max_x": -1.28,
            "post_min_y": -2.95,
            "post_max_y": -2.62,
            "post_min_height": 0.81,
            "post_max_height": 0.9,
            "target_x": -1.62,
            "target_y": -2.785,
            "target_z": 0.855,
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
