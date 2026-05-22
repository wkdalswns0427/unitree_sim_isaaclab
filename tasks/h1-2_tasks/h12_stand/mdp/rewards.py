"""Local reward terms for the balanced-standing task.

Base ``isaaclab.envs.mdp`` provides ``lin_vel_z_l2`` (vertical) and
``ang_vel_xy_l2`` (roll/pitch) but no horizontal-velocity or yaw-rate
penalty.  Standing requires *zero* motion in every axis, so we add the
two missing penalties here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of body-frame horizontal linear velocity (drift penalty)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.root_lin_vel_b[:, :2].pow(2), dim=-1)


def ang_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of body-frame yaw angular velocity (spinning penalty)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2].pow(2)


def both_feet_grounded_bonus(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0,
) -> torch.Tensor:
    """+1 reward when BOTH feet are loaded above ``force_threshold``; else 0.

    Pair with ``feet_slide`` so the policy cannot farm the bonus by skating.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    norm = torch.norm(forces, dim=-1).max(dim=1).values
    loaded = norm > force_threshold
    return loaded.all(dim=-1).float()
