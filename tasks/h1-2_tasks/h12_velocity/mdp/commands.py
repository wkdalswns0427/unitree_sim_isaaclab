from dataclasses import dataclass
from isaaclab.managers import CommandTerm

@dataclass
class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float]

class UniformVelocityCommand(CommandTerm):
    """Standard base velocity command for locomotion tasks."""
    cfg: object

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ranges = cfg.ranges