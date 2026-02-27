# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import warehouse_walk_h12_27dof_inspire_hw_env_cfg


gym.register(
    id="Isaac-Warehouse-Walk-H12-27dof-Inspire-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": warehouse_walk_h12_27dof_inspire_hw_env_cfg.WarehouseWalkH12InspireWholebodyEnvCfg,
    },
    disable_env_checker=True,
)
