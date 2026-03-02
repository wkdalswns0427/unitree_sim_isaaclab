import math
import torch

def time_out(env):
    return env.episode_length_buf >= env.max_episode_length

def illegal_contact(env, asset_name: str, body_names: list[str], threshold: float):
    asset = env.scene[asset_name]
    ids = asset.find_bodies(body_names)
    # contact forces are in world frame
    f = asset.data.net_contact_forces_w[:, ids, :]
    mag = torch.linalg.norm(f, dim=-1)
    return torch.any(mag > threshold, dim=-1)

def bad_orientation(env, asset_name: str, max_tilt_deg: float):
    asset = env.scene[asset_name]
    # Use projected gravity to infer tilt
    g = asset.data.projected_gravity_b
    # When upright, gravity points down in body frame, usually near (0,0,-1)
    tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) * 180.0 / math.pi
    return tilt > max_tilt_deg