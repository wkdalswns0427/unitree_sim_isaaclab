from isaaclab.utils import configclass

from .flat_env_cfg import H12FlatEnvCfg


@configclass
class H12VelocityEnvCfg(H12FlatEnvCfg):
    """Backward-compatible alias; use H12FlatEnvCfg from flat_env_cfg directly."""

    def __post_init__(self):
        super().__post_init__()
