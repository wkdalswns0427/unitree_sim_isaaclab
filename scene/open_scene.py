#!/usr/bin/env python3

"""Launch Isaac Sim and open a USD scene with H1-2 held in balanced standing stance."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)


# ── Balanced standing stance (C++ kReady, validated on hardware) ─────────────
# Balance condition: hip_pitch + knee + ankle_pitch ≈ 0  (-0.4 + 0.8 - 0.42 ≈ 0)
H12_STANCE = {
    "left_hip_yaw_joint":           0.0,
    "left_hip_pitch_joint":        -0.4,
    "left_hip_roll_joint":          0.0,
    "left_knee_joint":              0.8,
    "left_ankle_pitch_joint":      -0.42,
    "left_ankle_roll_joint":        0.0,
    "right_hip_yaw_joint":          0.0,
    "right_hip_pitch_joint":       -0.4,
    "right_hip_roll_joint":         0.0,
    "right_knee_joint":             0.8,
    "right_ankle_pitch_joint":     -0.42,
    "right_ankle_roll_joint":       0.0,
    "torso_joint":                  0.0,
    "left_shoulder_pitch_joint":   -0.3,
    "left_shoulder_roll_joint":     0.2,
    "left_shoulder_yaw_joint":      0.0,
    "left_elbow_joint":             0.3,
    "left_wrist_roll_joint":        0.0,
    "left_wrist_pitch_joint":       0.0,
    "left_wrist_yaw_joint":         0.0,
    "right_shoulder_pitch_joint":  -0.3,
    "right_shoulder_roll_joint":   -0.2,
    "right_shoulder_yaw_joint":     0.0,
    "right_elbow_joint":            0.3,
    "right_wrist_roll_joint":       0.0,
    "right_wrist_pitch_joint":      0.0,
    "right_wrist_yaw_joint":        0.0,
}

# Per-joint PD gains — ankles need high stiffness (400) to resist gravity
_KP = {
    "left_hip_yaw_joint": 200,    "left_hip_pitch_joint": 200,   "left_hip_roll_joint": 200,
    "left_knee_joint": 300,       "left_ankle_pitch_joint": 400,  "left_ankle_roll_joint": 200,
    "right_hip_yaw_joint": 200,   "right_hip_pitch_joint": 200,  "right_hip_roll_joint": 200,
    "right_knee_joint": 300,      "right_ankle_pitch_joint": 400, "right_ankle_roll_joint": 200,
    "torso_joint": 200,
    "left_shoulder_pitch_joint": 120, "left_shoulder_roll_joint": 120,
    "left_shoulder_yaw_joint": 80,    "left_elbow_joint": 80,
    "left_wrist_roll_joint": 40,      "left_wrist_pitch_joint": 40,  "left_wrist_yaw_joint": 40,
    "right_shoulder_pitch_joint": 120,"right_shoulder_roll_joint": 120,
    "right_shoulder_yaw_joint": 80,   "right_elbow_joint": 80,
    "right_wrist_roll_joint": 40,     "right_wrist_pitch_joint": 40, "right_wrist_yaw_joint": 40,
}
_KD = {
    "left_hip_yaw_joint": 5,     "left_hip_pitch_joint": 5,    "left_hip_roll_joint": 5,
    "left_knee_joint": 8,        "left_ankle_pitch_joint": 10,  "left_ankle_roll_joint": 5,
    "right_hip_yaw_joint": 5,    "right_hip_pitch_joint": 5,   "right_hip_roll_joint": 5,
    "right_knee_joint": 8,       "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 5,
    "torso_joint": 5,
    "left_shoulder_pitch_joint": 3, "left_shoulder_roll_joint": 3,
    "left_shoulder_yaw_joint": 2,   "left_elbow_joint": 2,
    "left_wrist_roll_joint": 1,     "left_wrist_pitch_joint": 1,  "left_wrist_yaw_joint": 1,
    "right_shoulder_pitch_joint": 3,"right_shoulder_roll_joint": 3,
    "right_shoulder_yaw_joint": 2,  "right_elbow_joint": 2,
    "right_wrist_roll_joint": 1,    "right_wrist_pitch_joint": 1, "right_wrist_yaw_joint": 1,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a USD scene in Isaac Sim.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "h1-2_cones.usd"),
        help="Path or URI to a USD file.",
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
        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import UsdPhysics

        scene_path = os.path.abspath(os.path.expanduser(args_cli.scene))
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene USD not found: {scene_path}")

        usd_ctx = omni.usd.get_context()
        if not usd_ctx.open_stage(scene_path):
            raise RuntimeError(f"Failed to open scene: {scene_path}")

        simulation_app.update()
        simulation_app.update()

        stage = usd_ctx.get_stage()

        # ── Pre-seed drive gains + targets into USD BEFORE play() ────────────
        # PhysX reads drive attributes at articulation creation (first play frame).
        # Without this, stiffness is 0 on frame 0 and the robot collapses instantly.
        joints_base = "/World/h1_2_with_FTP_hand/joints"
        for joint_name, q in H12_STANCE.items():
            prim = stage.GetPrimAtPath(f"{joints_base}/{joint_name}")
            if not prim.IsValid():
                continue
            for attr_name, val in [
                ("drive:angular:physics:targetPosition", float(q)),
                ("drive:angular:physics:stiffness",      float(_KP.get(joint_name, 100.0))),
                ("drive:angular:physics:damping",        float(_KD.get(joint_name,   5.0))),
            ]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(val)

        omni.timeline.get_timeline_interface().play()
        simulation_app.update()  # one frame — PhysX creates the articulation

        dc = _dynamic_control.acquire_dynamic_control_interface()

        # Find articulation root
        art_root = None
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/h1_2_with_FTP_hand") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = str(prim.GetPath())
                break

        if art_root is None:
            print("[ERROR] ArticulationRootAPI not found under /World/h1_2_with_FTP_hand")
            return

        art = dc.get_articulation(art_root)
        if art == _dynamic_control.INVALID_HANDLE:
            print(f"[ERROR] Invalid articulation handle at {art_root}")
            return

        n = dc.get_articulation_dof_count(art)
        print(f"[INFO] {art_root}  DOFs: {n}")

        # Apply per-joint gains and build name→dof map
        name_to_dof = {}
        for i in range(n):
            dof  = dc.get_articulation_dof(art, i)
            name = dc.get_dof_name(dof)
            name_to_dof[name] = dof
            props = dc.get_dof_properties(dof)
            props.stiffness = float(_KP.get(name, 100.0))
            props.damping   = float(_KD.get(name,   5.0))
            dc.set_dof_properties(dof, props)

        # Teleport all joints to stance immediately
        dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_POS)
        for idx, name in enumerate(name_to_dof.keys()):
            if name in H12_STANCE:
                dof_states["pos"][idx] = H12_STANCE[name]
        dc.set_articulation_dof_states(art, dof_states, _dynamic_control.STATE_POS)

        # Set position targets so drives hold the stance
        for name, q in H12_STANCE.items():
            if name in name_to_dof:
                dc.set_dof_position_target(name_to_dof[name], q)

        print(f"[INFO] Stance applied. Opened scene: {scene_path}")

        while simulation_app.is_running():
            # Re-send targets every frame to resist any external disturbance
            for name, q in H12_STANCE.items():
                if name in name_to_dof:
                    dc.set_dof_position_target(name_to_dof[name], q)
            simulation_app.update()

    finally:
        if usd_ctx is not None:
            usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
