import torch

def _exp_tracking(error: torch.Tensor, std: float):
    return torch.exp(-(error / std) ** 2)

def track_lin_vel_xy_exp(env, command_name: str, asset_name: str, std: float):
    asset = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    v_des = cmd[:, 0:2]
    v = asset.data.root_lin_vel_b[:, 0:2]
    err = torch.linalg.norm(v_des - v, dim=-1)
    return _exp_tracking(err, std)

def track_ang_vel_z_exp(env, command_name: str, asset_name: str, std: float):
    asset = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    wz_des = cmd[:, 2]
    wz = asset.data.root_ang_vel_b[:, 2]
    err = torch.abs(wz_des - wz)
    return _exp_tracking(err, std)

def action_rate_l2(env):
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)

def joint_acc_l2(env, asset_name: str):
    asset = env.scene[asset_name]
    return torch.sum(asset.data.joint_acc**2, dim=-1)

def joint_torques_l2(env, asset_name: str):
    asset = env.scene[asset_name]
    return torch.sum(asset.data.applied_joint_effort**2, dim=-1)

def upright_bonus(env, asset_name: str):
    asset = env.scene[asset_name]
    # projected gravity z close to -1 means upright, depending on convention
    # Using +z axis up in body frame, adapt if sign differs
    g = asset.data.projected_gravity_b
    return 1.0 - torch.abs(g[:, 2] + 1.0)

def base_height_l2(env, asset_name: str, target_height: float):
    asset = env.scene[asset_name]
    z = asset.data.root_pos_w[:, 2]
    return (z - target_height) ** 2

def foot_slip_l2(env, asset_name: str, feet_body_names: list[str]):
    asset = env.scene[asset_name]
    feet_ids = asset.find_bodies(feet_body_names)
    v = asset.data.body_lin_vel_w[:, feet_ids, 0:2]
    return torch.sum(v**2, dim=(-1, -2))