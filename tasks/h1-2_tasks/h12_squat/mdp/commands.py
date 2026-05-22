"""Custom command term: uniform target pelvis (root) height in metres.

Samples a target height in ``[height_range[0], height_range[1]]`` for each
environment, resampled every ``resampling_time_range[0..1]`` seconds.  The
policy observes this scalar and is rewarded for matching it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformPelvisHeightCommand(CommandTerm):
    """Pelvis-height target, resampled on a fixed cadence.

    Two modes:
      * ``mode="uniform"``    — sample a uniform random height in ``height_range``.
      * ``mode="alternating"`` — flip-flop between ``height_range[0]`` (squat)
        and ``height_range[1]`` (stand) on every resample event.  This produces
        a clean "down → hold → up → hold → ..." schedule that mirrors a squat
        rep.  Each env's phase is randomized at episode reset so different
        envs lead/lag, which improves PPO's experience diversity.
    """

    cfg: "UniformPelvisHeightCommandCfg"

    def __init__(self, cfg: "UniformPelvisHeightCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        # Per-env "current phase": 0 = stand high, 1 = squat low.  Initialized
        # randomly so different envs are out of sync.
        self._phase = torch.randint(
            0, 2, (self.num_envs,), device=self.device, dtype=torch.long
        )
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Target pelvis height. Shape is (num_envs, 1)."""
        return self.height_cmd

    def _update_metrics(self) -> None:
        h_target = self.height_cmd[:, 0]
        h_root = self.robot.data.root_link_pos_w[:, 2]
        max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["height_error"] += torch.abs(h_target - h_root) / max_steps

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self.cfg.mode == "alternating":
            # Flip phase, then map phase 0 → height_range[1] (stand high),
            # phase 1 → height_range[0] (squat low).
            self._phase[env_ids] = 1 - self._phase[env_ids]
            low, high = self.cfg.height_range
            target = torch.where(
                self._phase[env_ids] == 0,
                torch.full_like(self._phase[env_ids], high, dtype=torch.float32),
                torch.full_like(self._phase[env_ids], low, dtype=torch.float32),
            )
            self.height_cmd[env_ids, 0] = target
        else:  # uniform
            r = torch.empty(len(env_ids), device=self.device)
            self.height_cmd[env_ids, 0] = r.uniform_(*self.cfg.height_range)

    def _update_command(self) -> None:
        # Stateless within a hold; phase flips happen in _resample_command.
        return

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        return


@configclass
class UniformPelvisHeightCommandCfg(CommandTermCfg):
    """Cfg for :class:`UniformPelvisHeightCommand`."""

    class_type: type = UniformPelvisHeightCommand
    asset_name: str = MISSING
    """Name of the articulation in the scene whose root height we track."""
    height_range: tuple[float, float] = (0.50, 1.00)
    """Inclusive bounds on the height target (metres).

    In ``alternating`` mode this is interpreted as ``(squat_low, stand_high)``.
    """
    mode: str = "uniform"
    """Either ``"uniform"`` (random sample in range) or ``"alternating"``
    (flip-flop between the two bounds on each resample)."""
