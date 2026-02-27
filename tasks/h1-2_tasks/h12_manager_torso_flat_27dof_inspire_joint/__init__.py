# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym


gym.register(
    id="Isaac-H12-Manager-Torso-Flat-27dof-Inspire-Joint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12ManagerTorsoFlatEnvCfg",
    },
)


gym.register(
    id="Isaac-H12-Manager-Torso-Flat-27dof-Inspire-Joint-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12ManagerTorsoFlatEnvCfg_PLAY",
    },
)
