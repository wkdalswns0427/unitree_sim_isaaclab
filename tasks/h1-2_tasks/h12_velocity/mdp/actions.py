import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils.math import saturate

class JointVelocityAction(ActionTerm):
    """Applies joint velocity targets to selected joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.asset_name]
        self.joint_ids = self.asset.find_joints(cfg.joint_names)

        self.scale = float(getattr(cfg, "scale", 1.0))

    @property
    def action_dim(self) -> int:
        return len(self.joint_ids)

    def apply_actions(self):
        # Actions are in [-1, 1], scale to rad/s
        a = self._actions
        a = saturate(a, -1.0, 1.0) * self.scale

        # Build target tensor for all joints
        # Isaac Lab expects full joint target vectors, so we write into a full tensor.
        v_des = torch.zeros((self.env.num_envs, self.asset.num_joints), device=self.env.device)
        v_des[:, self.joint_ids] = a
        self.asset.set_joint_velocity_target(v_des)