#!/usr/bin/env python3
"""Train the legacy 21-DOF wholebody H1-2 velocity policy.

Thin wrapper around ``train.py`` that pre-selects
``--task Isaac-H12-Velocity-Wholebody-v0``.  All other CLI args (--num_envs,
--max_iterations, --headless, --device, etc.) pass through unchanged.

Logs land in ``logs/rsl_rl/h12_velocity_wholebody_flat/`` so they do not
collide with the new 13-DOF runs under ``logs/rsl_rl/h12_velocity_flat/``.

Usage:
    python3 scripts/reinforcement_learning/rsl_rl/train_wholebody.py \
        --num_envs 4096 --headless
"""

import os
import sys

LEGACY_TASK = "Isaac-H12-Velocity-Wholebody-v0"

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "train.py")

# Re-invoke train.py with --task <legacy> prepended (unless caller already set it).
forwarded = sys.argv[1:]
if "--task" not in forwarded:
    forwarded = ["--task", LEGACY_TASK] + forwarded

os.execvp(sys.executable, [sys.executable, MAIN] + forwarded)
