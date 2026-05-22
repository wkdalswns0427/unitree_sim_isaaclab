"""Curriculum terms for the squat task.

The "stand-back-up from a deep squat" failure mode is the natural blocker for
``Isaac-H12-Squat-v0``: with a fixed wide ``(0.55, 1.00)`` height range the
policy can collect easy reward by sinking and parking near the squat target,
without ever learning the harder up-from-deep extension.

``progressive_squat_depth`` linearly widens the squat depth from a shallow
dip toward the full range as training episodes accumulate.  The shallow start
forces the policy to learn the down-and-up cycle on a *small* height delta
first, where the leg extension torque required is well within ankle/knee
limits; then the depth grows so the policy can ramp into the deeper squat
while keeping the up-phase competence it built up early.

Counted in **episode resets** (``sum(len(env_ids))`` across curriculum calls)
rather than wall time or iterations, so the schedule is independent of the
chosen ``num_envs`` / ``num_steps_per_env``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progressive_squat_depth(
    env: "ManagerBasedRLEnv",
    env_ids,
    start_low: float = 0.92,
    end_low: float = 0.55,
    stand_high: float = 1.00,
    progression_resets: int = 50000,
) -> torch.Tensor:
    """Widen ``commands.target_height.height_range`` from ``(start_low,
    stand_high)`` toward ``(end_low, stand_high)`` over ``progression_resets``
    cumulative episode resets.

    Returns the current squat-low value so it shows up in the TensorBoard
    "Curriculum/squat_depth" trace and you can see the progression live.
    """
    cmd = env.command_manager.get_term("target_height")

    # Lazy side-counter on the env (the curriculum manager has no built-in
    # global episode counter we can read).
    if not hasattr(env, "_squat_total_resets"):
        env._squat_total_resets = 0
    try:
        n_new = int(len(env_ids))
    except TypeError:
        n_new = 0
    env._squat_total_resets += n_new

    progress = float(env._squat_total_resets) / max(1.0, float(progression_resets))
    progress = min(1.0, max(0.0, progress))
    new_low = start_low + progress * (end_low - start_low)

    # `@configclass` is a dataclass under the hood — attribute assignment
    # works at runtime, and `_resample_command` re-reads `self.cfg.height_range`
    # on every flip, so the next squat target uses the new bound.
    cmd.cfg.height_range = (new_low, stand_high)

    return torch.tensor(new_low, device=env.device)
