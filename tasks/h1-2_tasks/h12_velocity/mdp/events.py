import torch
from isaaclab.utils.math import quat_from_euler_xyz

def reset_root_state_uniform(env, asset_name: str, pose_range: dict, vel_range: dict):
    asset = env.scene[asset_name]
    n = env.num_envs
    device = env.device

    x = torch.empty(n, device=device).uniform_(*pose_range["x"])
    y = torch.empty(n, device=device).uniform_(*pose_range["y"])
    yaw = torch.empty(n, device=device).uniform_(*pose_range["yaw"])

    q = quat_from_euler_xyz(
        torch.zeros(n, device=device),
        torch.zeros(n, device=device),
        yaw,
    )

    root_pos = asset.data.default_root_state[:, 0:3].clone()
    root_pos[:, 0] += x
    root_pos[:, 1] += y

    root_vel = torch.zeros((n, 6), device=device)
    root_vel[:, 0] = torch.empty(n, device=device).uniform_(*vel_range["x"])
    root_vel[:, 1] = torch.empty(n, device=device).uniform_(*vel_range["y"])
    root_vel[:, 2] = torch.empty(n, device=device).uniform_(*vel_range["z"])
    root_vel[:, 5] = torch.empty(n, device=device).uniform_(*vel_range["yaw"])

    asset.write_root_pose_to_sim(torch.cat([root_pos, q], dim=-1))
    asset.write_root_velocity_to_sim(root_vel)

def reset_joints_by_offset(env, asset_name: str, position_offset_range: tuple, velocity_range: tuple):
    asset = env.scene[asset_name]
    n = env.num_envs
    device = env.device

    q = asset.data.default_joint_pos.clone()
    dq = torch.zeros_like(asset.data.default_joint_vel)

    q += torch.empty_like(q).uniform_(*position_offset_range)
    dq += torch.empty_like(dq).uniform_(*velocity_range)

    asset.write_joint_state_to_sim(q, dq)

def external_push(env, asset_name: str, force_range: tuple[float, float]):
    asset = env.scene[asset_name]
    n = env.num_envs
    device = env.device

    f = torch.zeros((n, 3), device=device)
    mag = torch.empty(n, device=device).uniform_(*force_range)
    angle = torch.empty(n, device=device).uniform_(-3.14159, 3.14159)
    f[:, 0] = mag * torch.cos(angle)
    f[:, 1] = mag * torch.sin(angle)

    asset.apply_external_force_and_torque(f, torch.zeros_like(f))