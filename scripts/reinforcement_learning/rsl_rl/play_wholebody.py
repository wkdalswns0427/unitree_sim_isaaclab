#!/usr/bin/env python3
"""Play a legacy 21-DOF wholebody H1-2 velocity checkpoint.

Thin wrapper around ``play.py`` that pre-selects
``--task Isaac-H12-Velocity-Wholebody-Play-v0`` (the play variant of the
wholebody task — 50 envs, no observation corruption, deterministic command).
Pass ``--task Isaac-H12-Velocity-Wholebody-v0`` explicitly if you want the
training-environment variant instead.

All other CLI args (--num_envs, --checkpoint, --headless, --real-time, ...)
pass through unchanged.

Usage:
    python3 scripts/reinforcement_learning/rsl_rl/play_wholebody.py \
        --checkpoint logs/rsl_rl/h12_velocity_wholebody_flat/<run>/model_5000.pt

    # or to play the old (h12_velocity_flat) checkpoints against the wholebody env:
    python3 scripts/reinforcement_learning/rsl_rl/play_wholebody.py \
        --checkpoint logs/rsl_rl/h12_velocity_flat/2026-05-15_14-41-04/model_5000.pt
"""

import os
import sys

LEGACY_TASK = "Isaac-H12-Velocity-Wholebody-Play-v0"

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "play.py")

forwarded = sys.argv[1:]
if "--task" not in forwarded:
    forwarded = ["--task", LEGACY_TASK] + forwarded

os.execvp(sys.executable, [sys.executable, MAIN] + forwarded)
