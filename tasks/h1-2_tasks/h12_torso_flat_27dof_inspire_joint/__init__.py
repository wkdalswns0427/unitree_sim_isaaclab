# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import torso_flat_h12_27dof_inspire_joint_env_cfg


gym.register(
    id="Isaac-Torso-Flat-H12-27dof-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": torso_flat_h12_27dof_inspire_joint_env_cfg.TorsoFlatH12InspireJointEnvCfg,
    },
    disable_env_checker=True,
)
