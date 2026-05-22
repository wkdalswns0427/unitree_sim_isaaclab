"""Reward functions for the squat task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_pelvis_height_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian-shaped reward for matching the target pelvis height.

    reward = exp(-(z - z_target)^2 / std^2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    z = asset.data.root_link_pos_w[:, 2]
    err = z - target
    return torch.exp(-err.pow(2) / (std * std))


def pelvis_height_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 distance |z - z_target|.  Use with a NEGATIVE weight as a penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    z = asset.data.root_link_pos_w[:, 2]
    return torch.abs(z - target)


def lin_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of body-frame horizontal linear velocity.  Penalty for drifting."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.root_lin_vel_b[:, :2].pow(2), dim=-1)


def ang_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of body-frame yaw angular velocity.  Penalty for spinning."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2].pow(2)


def both_feet_grounded_bonus(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0,
) -> torch.Tensor:
    """+1 reward when BOTH feet are loaded above ``force_threshold``; else 0.

    Designed to encourage two-foot stance during the squat — single-foot
    balance is the dominant failure mode for early policies.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    norm = torch.norm(forces, dim=-1).max(dim=1).values  # (num_envs, num_bodies)
    loaded = norm > force_threshold
    both_loaded = loaded.all(dim=-1).float()
    return both_loaded
