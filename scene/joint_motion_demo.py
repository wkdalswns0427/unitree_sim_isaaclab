#!/usr/bin/env python3

"""Open scene USD and drive simple joint motions on both H1 robots."""

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)


def _set_attr(prim, name: str, value) -> None:
    attr = prim.GetAttribute(name)
    if attr.IsValid():
        attr.Set(value)


def _set_joint_target(stage, joint_path: str, target: float, kp: float | None = None, kd: float | None = None) -> None:
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        return
    _set_attr(prim, "drive:angular:physics:targetPosition", float(target))
    if kp is not None:
        _set_attr(prim, "drive:angular:physics:stiffness", float(kp))
    if kd is not None:
        _set_attr(prim, "drive:angular:physics:damping", float(kd))


def _disable_gravity_under(stage, root_path: str) -> int:
    from pxr import UsdPhysics

    count = 0
    for prim in stage.Traverse():
        spath = str(prim.GetPath())
        if not spath.startswith(root_path):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateDisableGravityAttr(True)
            count += 1
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H1/H1-2 joint motion demo in a scene USD.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "warehouse_h1_sc.usd"),
        help="Path to scene USD.",
    )
    parser.add_argument("--rate", type=float, default=120.0, help="Update rate (Hz).")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0=forever).")
    parser.add_argument("--torso_amp", type=float, default=0.20, help="Torso oscillation amplitude (rad).")
    parser.add_argument("--torso_freq", type=float, default=0.35, help="Torso oscillation frequency (Hz).")
    parser.add_argument(
        "--keep-upright",
        action="store_true",
        default=True,
        help="Disable gravity on robot rigid bodies to prevent falling.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()

    if "--headless" not in sys.argv:
        args_cli.headless = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    usd_ctx = None
    try:
        import omni.timeline
        import omni.usd

        scene_path = os.path.abspath(os.path.expanduser(args_cli.scene))
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene USD not found: {scene_path}")

        usd_ctx = omni.usd.get_context()
        if not usd_ctx.open_stage(scene_path):
            raise RuntimeError(f"Failed to open scene: {scene_path}")
        stage = usd_ctx.get_stage()

        # Robot A: Isaac H1 in /World
        h1_base = "/World/h1/Joints"
        h1_targets = {
            "left_hip_pitch": -0.28,
            "right_hip_pitch": -0.28,
            "left_knee": 0.79,
            "right_knee": 0.79,
            "left_ankle": -0.52,
            "right_ankle": -0.52,
        }
        h1_torso = f"{h1_base}/torso"

        # Robot B: H1-2 FTP hand imported at root
        h12_base = "/h1_2_with_FTP_hand/joints"
        h12_targets = {
            "left_hip_pitch_joint": -0.20,
            "right_hip_pitch_joint": -0.20,
            "left_knee_joint": 0.42,
            "right_knee_joint": 0.42,
            "left_ankle_pitch_joint": -0.23,
            "right_ankle_pitch_joint": -0.23,
        }
        h12_torso = f"{h12_base}/torso_joint"

        if args_cli.keep_upright:
            h1_count = _disable_gravity_under(stage, "/World/h1")
            h12_count = _disable_gravity_under(stage, "/h1_2_with_FTP_hand")
            print(f"[INFO] Disabled gravity on rigid bodies: /World/h1={h1_count}, /h1_2_with_FTP_hand={h12_count}")

        # Start sim.
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        dt = 1.0 / max(args_cli.rate, 1e-3)
        t0 = time.time()
        print(f"[INFO] Running joint motion demo in: {scene_path}")

        while simulation_app.is_running():
            now = time.time()
            t = now - t0
            if args_cli.duration > 0.0 and t >= args_cli.duration:
                break

            torso_offset = args_cli.torso_amp * math.sin(2.0 * math.pi * args_cli.torso_freq * t)

            # Hold stable stand targets each frame (prevents collapse in free-run scene).
            for joint_name, q in h1_targets.items():
                _set_joint_target(stage, f"{h1_base}/{joint_name}", q, kp=140.0, kd=10.0)
            _set_joint_target(stage, h1_torso, torso_offset, kp=120.0, kd=10.0)

            for joint_name, q in h12_targets.items():
                _set_joint_target(stage, f"{h12_base}/{joint_name}", q, kp=140.0, kd=10.0)
            _set_joint_target(stage, h12_torso, torso_offset, kp=120.0, kd=10.0)

            simulation_app.update()

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

        timeline.stop()
    finally:
        if usd_ctx is not None:
            usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
