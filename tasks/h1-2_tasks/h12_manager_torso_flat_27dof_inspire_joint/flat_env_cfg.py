# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, H12RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.h12_27dof_state import get_robot_boy_joint_states
from tasks.common_observations.inspire_state import get_robot_inspire_joint_states


@configclass
class H12ManagerTorsoFlatSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_base_fix(
        init_pos=(0.0, 0.0, 0.96),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )

    front_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/envs/env_.*/Robot/camera_link/front_cam",
    )
    left_wrist_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/envs/env_.*/Robot/left_hand.*base_link/left_wrist_camera",
        pos_offset=(-0.04012, -0.07441, 0.15711),
        rot_offset=(0.00539, 0.86024, 0.0424, 0.50809),
        focal_length=12.0,
    )
    right_wrist_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/envs/env_.*/Robot/right_hand.*base_link/right_wrist_camera",
        pos_offset=(-0.04012, 0.07441, 0.15711),
        rot_offset=(0.00539, 0.86024, 0.0424, 0.50809),
        focal_length=12.0,
    )
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(2.6, -2.2, 1.6),
        rot_offset=(-0.35641, 0.61117, 0.61117, -0.35641),
        focal_length=14.0,
        horizontal_aperture=24.0,
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


@configclass
class EventCfg:
    pass


@configclass
class CommandsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class H12ManagerTorsoFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: H12ManagerTorsoFlatSceneCfg = H12ManagerTorsoFlatSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0

        self.sim.dt = 0.005
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


@configclass
class H12ManagerTorsoFlatEnvCfg_PLAY(H12ManagerTorsoFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
