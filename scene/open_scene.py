#!/usr/bin/env python3

"""Launch Isaac Sim and open a USD scene."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a USD scene in Isaac Sim.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "warehouse_h1_sc.usd"),
        help="Path or URI to a USD file.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()

    # Keep GUI enabled by default for scene viewing.
    if "--headless" not in sys.argv:
        args_cli.headless = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    usd_ctx = None
    try:
        import omni.usd

        scene_path = os.path.abspath(os.path.expanduser(args_cli.scene))
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene USD not found: {scene_path}")

        usd_ctx = omni.usd.get_context()
        ok = usd_ctx.open_stage(scene_path)
        if not ok:
            raise RuntimeError(f"Failed to open scene: {scene_path}")

        print(f"[INFO] Opened scene: {scene_path}")
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        if usd_ctx is not None:
            usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
