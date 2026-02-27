
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import agents
from . import move_cylinder_h12_27dof_inspire_hw_env_cfg


gym.register(
    id="Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_cylinder_h12_27dof_inspire_hw_env_cfg.MoveCylinderH1227dofInspireWholebodyEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveCylinderH12InspireWholebodyPPORunnerCfg",
    },
    disable_env_checker=True,
)
