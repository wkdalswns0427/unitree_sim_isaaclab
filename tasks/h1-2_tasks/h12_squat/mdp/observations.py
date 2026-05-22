"""Observation functions for the squat task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_pelvis_height(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Current target pelvis height from the command manager. Shape (num_envs, 1)."""
    return env.command_manager.get_command(command_name)


def pelvis_height(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current pelvis (root link) world-frame z. Shape (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]
