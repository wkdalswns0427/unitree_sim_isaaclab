import gymnasium as gym

from . import agents


# Lower-body-only (legs + torso = 13 DOF) action subset.  Arms held at default
# by their actuator config; obs dim = 51.  Current recommended config.
gym.register(
    id="Isaac-H12-Velocity-Legonly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H12-Velocity-Legonly-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12FlatPPORunnerCfg",
    },
)


# 21-DOF wholebody action subset (legs + torso + 8 arms).  Matches the
# 2026-05-08 and 2026-05-15 training runs (h12_velocity_flat/) — preserved so
# those checkpoints continue to play and can be retrained.
gym.register(
    id="Isaac-H12-Velocity-Wholebody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wholebody_env_cfg:H12WholebodyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12WholebodyFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H12-Velocity-Wholebody-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wholebody_env_cfg:H12WholebodyFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12WholebodyFlatPPORunnerCfg",
    },
)
