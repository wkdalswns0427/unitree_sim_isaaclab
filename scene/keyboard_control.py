#!/usr/bin/env python3
"""H1-2 keyboard joint controller for the cones scene.

Launches Isaac Sim with ``scene/h1-2_cones.usd`` (default) or any ``--scene``
USD, authors per-joint PD gains matching the Krumi pick_cone_palms.py SDK
(via ``scene/joint_motion.py``'s ``_KP_SDK`` / ``_KD_SDK``), then drives the
H1-2 joints from the keyboard.  Pelvis is pinned by default so the robot
holds station while you move joints.

Key bindings
------------
Arms (left)                    Arms (right)
    q / a   shoulder pitch ±       u / j   shoulder pitch ±
    w / s   shoulder roll  ±       i / k   shoulder roll  ±
    e / d   shoulder yaw   ±       o / l   shoulder yaw   ±
    r / f   elbow          ±       p / ;   elbow          ±

Torso                          Legs (coordinated)
    t / g   torso yaw ±             6   squat down (knees + ankles)
                                    5   stand up (extend)
                                    8   lean forward (hip pitch −)
                                    7   lean back    (hip pitch +)
                                    b   bend over   (hip flex + knee + ankle)
                                    n   un-bend     (reverse of b)

Misc
    0   snap every joint back to default pose
    -   slower step rate         (multiplier ×0.5)
    =   faster step rate         (multiplier ×2.0)
    ESC quit

Usage
-----
    python3 scene/keyboard_control.py
    python3 scene/keyboard_control.py --no_pin_pelvis     # let the legs balance
    python3 scene/keyboard_control.py --scene other.usd
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

from scene.joint_motion import (
    POLICY_JOINT_DEFAULTS,
    _KD_SDK,
    _KP_SDK,
    _STANCE_ANKLE_OVERRIDES,
    _dump_stage_diagnostic,
    _find_articulation_root,
    _pin_articulation_base,
    _to_numpy,
)


DEFAULT_SCENE = os.path.join(PROJECT_ROOT, "scene", "h1-2_cones.usd")

# Foot link substrings to pin when --pin_feet is set.
FOOT_LINK_TOKENS = ("left_ankle_roll_link", "right_ankle_roll_link")


def _pin_feet_to_world(stage, root_path: str) -> int:
    """UsdPhysics.FixedJoint between world and each *_ankle_roll_link foot.

    Returns the number of feet pinned (0/1/2).  Reads each foot's current
    world transform from USD and freezes it there, so the robot stays
    glued to its current standing pose.
    """
    from pxr import UsdPhysics, UsdGeom, Sdf, Gf

    if not stage.GetPrimAtPath(root_path).IsValid():
        print(f"[WARN] Articulation root prim invalid at {root_path}; cannot pin feet.")
        return 0

    n_pinned = 0
    for token in FOOT_LINK_TOKENS:
        target = next(
            (prim for prim in stage.Traverse()
             if str(prim.GetPath()).startswith(root_path)
             and prim.HasAPI(UsdPhysics.RigidBodyAPI)
             and token in str(prim.GetPath()).lower()),
            None,
        )
        if target is None:
            print(f"[WARN] No rigid-body prim matching '{token}' under {root_path}.")
            continue

        world_mat = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(0)
        world_pos = world_mat.ExtractTranslation()
        world_rot = world_mat.ExtractRotationQuat()
        quat_f = Gf.Quatf(
            float(world_rot.GetReal()),
            Gf.Vec3f(*[float(c) for c in world_rot.GetImaginary()]),
        )

        pin_path = Sdf.Path(f"/World/_{token}_pin")
        if stage.GetPrimAtPath(pin_path).IsValid():
            stage.RemovePrim(pin_path)

        joint = UsdPhysics.FixedJoint.Define(stage, pin_path)
        joint.CreateBody1Rel().SetTargets([target.GetPath()])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
        )
        joint.CreateLocalRot0Attr().Set(quat_f)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        print(f"[INFO] Pinned {target.GetPath()} to world at "
              f"({world_pos[0]:.3f},{world_pos[1]:.3f},{world_pos[2]:.3f}).")
        n_pinned += 1

    return n_pinned


# Per-joint bindings: char → (joint_name, sign).  Each iteration we add
# ``sign * step_per_iter`` to the joint's target while the key is held.
KEY_BINDINGS_SINGLE_JOINT = {
    # ── LEFT ARM ────────────────────────────────────────────────────────────
    "q": ("left_shoulder_pitch_joint",  +1.0),
    "a": ("left_shoulder_pitch_joint",  -1.0),
    "w": ("left_shoulder_roll_joint",   +1.0),
    "s": ("left_shoulder_roll_joint",   -1.0),
    "e": ("left_shoulder_yaw_joint",    +1.0),
    "d": ("left_shoulder_yaw_joint",    -1.0),
    "r": ("left_elbow_joint",           +1.0),
    "f": ("left_elbow_joint",           -1.0),
    # ── RIGHT ARM ───────────────────────────────────────────────────────────
    "u": ("right_shoulder_pitch_joint", +1.0),
    "j": ("right_shoulder_pitch_joint", -1.0),
    "i": ("right_shoulder_roll_joint",  +1.0),
    "k": ("right_shoulder_roll_joint",  -1.0),
    "o": ("right_shoulder_yaw_joint",   +1.0),
    "l": ("right_shoulder_yaw_joint",   -1.0),
    "p": ("right_elbow_joint",          +1.0),
    ";": ("right_elbow_joint",          -1.0),
    # ── TORSO ───────────────────────────────────────────────────────────────
    "t": ("torso_joint", +1.0),
    "g": ("torso_joint", -1.0),
}

# Coordinated leg motions — single key moves multiple joints together.
KEY_BINDINGS_LEG_GROUP = {
    # Squat down: bend knees (+), plantarflex ankles (+), pitch hips back (−).
    "6": {
        "left_knee_joint":         +2.0,
        "right_knee_joint":        +2.0,
        "left_hip_pitch_joint":    -1.0,
        "right_hip_pitch_joint":   -1.0,
        "left_ankle_pitch_joint":  +1.0,
        "right_ankle_pitch_joint": +1.0,
    },
    # Stand up: reverse.
    "5": {
        "left_knee_joint":         -2.0,
        "right_knee_joint":        -2.0,
        "left_hip_pitch_joint":    +1.0,
        "right_hip_pitch_joint":   +1.0,
        "left_ankle_pitch_joint":  -1.0,
        "right_ankle_pitch_joint": -1.0,
    },
    # Lean forward: dorsiflex ankles (−ankle_pitch), pitch hips forward (+).
    "8": {
        "left_hip_pitch_joint":    +1.0,
        "right_hip_pitch_joint":   +1.0,
        "left_ankle_pitch_joint":  -1.0,
        "right_ankle_pitch_joint": -1.0,
    },
    # Lean back.
    "7": {
        "left_hip_pitch_joint":    -1.0,
        "right_hip_pitch_joint":   -1.0,
        "left_ankle_pitch_joint":  +1.0,
        "right_ankle_pitch_joint": +1.0,
    },
    # Bend over: deep hip flexion (fold torso forward), plus moderate knee
    # bend + ankle dorsiflexion so the shins track the forward CoM shift.
    "b": {
        "left_hip_pitch_joint":    -2.0,
        "right_hip_pitch_joint":   -2.0,
        "left_knee_joint":         +1.0,
        "right_knee_joint":        +1.0,
        "left_ankle_pitch_joint":  -1.0,
        "right_ankle_pitch_joint": -1.0,
    },
    # Un-bend (return from bend-over): reverse of "b".
    "n": {
        "left_hip_pitch_joint":    +2.0,
        "right_hip_pitch_joint":   +2.0,
        "left_knee_joint":         -1.0,
        "right_knee_joint":        -1.0,
        "left_ankle_pitch_joint":  +1.0,
        "right_ankle_pitch_joint": +1.0,
    },
}


# Per-joint hard clip so we don't drive a joint past sane limits.
JOINT_LIMITS = {
    # Hip
    "left_hip_yaw_joint":   (-0.5, 0.5),  "right_hip_yaw_joint":   (-0.5, 0.5),
    "left_hip_roll_joint":  (-0.4, 0.4),  "right_hip_roll_joint":  (-0.4, 0.4),
    "left_hip_pitch_joint": (-2.5, 0.5),  "right_hip_pitch_joint": (-2.5, 0.5),
    # Knee
    "left_knee_joint":      ( 0.0, 1.8),  "right_knee_joint":      ( 0.0, 1.8),
    # Ankle
    "left_ankle_pitch_joint":(-0.7, 0.5), "right_ankle_pitch_joint":(-0.7, 0.5),
    "left_ankle_roll_joint": (-0.3, 0.3), "right_ankle_roll_joint": (-0.3, 0.3),
    # Torso
    "torso_joint":          (-1.5, 1.5),
    # Shoulders
    "left_shoulder_pitch_joint":  (-2.5, 1.5),  "right_shoulder_pitch_joint": (-2.5, 1.5),
    "left_shoulder_roll_joint":   (-0.8, 1.6),  "right_shoulder_roll_joint":  (-1.6, 0.8),
    "left_shoulder_yaw_joint":    (-1.6, 1.6),  "right_shoulder_yaw_joint":   (-1.6, 1.6),
    # Elbow
    "left_elbow_joint":           ( 0.0, 2.2),  "right_elbow_joint":          ( 0.0, 2.2),
    # Wrists (not bound to keys; clip just in case)
    "left_wrist_roll_joint":      (-1.5, 1.5),  "right_wrist_roll_joint":     (-1.5, 1.5),
    "left_wrist_pitch_joint":     (-0.5, 0.5),  "right_wrist_pitch_joint":    (-0.5, 0.5),
    "left_wrist_yaw_joint":       (-1.0, 1.0),  "right_wrist_yaw_joint":      (-1.0, 1.0),
}


class _KeyState:
    """Thread-safe key tracker + special action flags (reset / step-rate)."""

    def __init__(self):
        self._held: set[str] = set()
        self._reset_flag = False
        self._rate_multiplier_change = 1.0
        self._quit_flag = False
        self._lock = threading.Lock()

    def on_press(self, key) -> Optional[bool]:
        # Special keys
        from pynput import keyboard as kb
        if isinstance(key, kb.Key) and key == kb.Key.esc:
            with self._lock:
                self._quit_flag = True
            return False  # stop listener
        c = getattr(key, "char", None)
        if c is None:
            return
        with self._lock:
            if c == "0":
                self._reset_flag = True
                return
            if c == "-":
                self._rate_multiplier_change *= 0.5
                print(f"[KB] step rate ×0.5 -> ×{self._rate_multiplier_change:.3f}")
                return
            if c == "=":
                self._rate_multiplier_change *= 2.0
                print(f"[KB] step rate ×2.0 -> ×{self._rate_multiplier_change:.3f}")
                return
            self._held.add(c)

    def on_release(self, key):
        c = getattr(key, "char", None)
        if c is None:
            return
        with self._lock:
            self._held.discard(c)

    def held(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def pop_reset(self) -> bool:
        with self._lock:
            r = self._reset_flag
            self._reset_flag = False
            return r

    def consume_rate_change(self) -> float:
        with self._lock:
            m = self._rate_multiplier_change
            self._rate_multiplier_change = 1.0
            return m

    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit_flag


def _print_keymap():
    print("=" * 72)
    print(" H1-2 keyboard joint controller — bindings")
    print("=" * 72)
    print(" ARMS (LEFT)              ARMS (RIGHT)")
    print("   q/a shoulder pitch ±     u/j shoulder pitch ±")
    print("   w/s shoulder roll  ±     i/k shoulder roll  ±")
    print("   e/d shoulder yaw   ±     o/l shoulder yaw   ±")
    print("   r/f elbow          ±     p/; elbow          ±")
    print(" TORSO                    LEGS (coordinated)")
    print("   t/g torso yaw ±          6 squat down    5 stand up")
    print("                            8 lean forward  7 lean back")
    print("                            b bend over     n un-bend")
    print(" MISC")
    print("   0 reset all to default pose")
    print("   - slower (×0.5)   = faster (×2.0)   ESC quit")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE,
                        help=f"Scene USD (default: {DEFAULT_SCENE}).")
    parser.add_argument("--robot_root", type=str, default="/World/h1_2_physics",
                        help="USD path of the articulation root prim.")
    parser.add_argument("--physics_dt", type=float, default=1.0 / 200.0,
                        help="Physics step (s).")
    parser.add_argument("--decimation", type=int, default=4,
                        help="Physics steps per control update.")
    parser.add_argument("--step_rate", type=float, default=0.6,
                        help="Joint speed in rad/s while a key is held.")
    parser.add_argument("--pin_pelvis", dest="pin_pelvis", action="store_true", default=True,
                        help="Pin the pelvis to world (default ON for stable arm work).")
    parser.add_argument("--no_pin_pelvis", dest="pin_pelvis", action="store_false",
                        help="Let the legs balance the body.  PD only.")
    parser.add_argument("--pin_feet", dest="pin_feet", action="store_true", default=False,
                        help="Pin both *_ankle_roll_link feet to world (default OFF).")
    parser.add_argument("--no_pin_feet", dest="pin_feet", action="store_false",
                        help="Do not pin feet.")
    parser.add_argument("--stance_ankle", dest="stance_ankle", action="store_true", default=True,
                        help="Stiffen ankle PD (kp=200 pitch / 150 roll, max_force=200 Nm) "
                             "for passive balance.  Default ON.")
    parser.add_argument("--no_stance_ankle", dest="stance_ankle", action="store_false",
                        help="Use the soft SDK ankle gains.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if "--headless" not in sys.argv:
        args.headless = False

    _print_keymap()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pynput import keyboard as kb

    scene_path = os.path.abspath(os.path.expanduser(args.scene))
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    usd_ctx = omni.usd.get_context()
    if not usd_ctx.open_stage(scene_path):
        raise RuntimeError(f"Failed to open: {scene_path}")
    simulation_app.update(); simulation_app.update()
    stage = usd_ctx.get_stage()

    # Pre-author PD gains + initial drive targets on every joint, BEFORE
    # physics initializes.  Matches the joint_motion.py / walk_policy.py
    # pattern: USD attributes are the reliable "send kp/kd" path in this
    # Isaac Sim build.
    joints_base = f"{args.robot_root}/joints"
    seeded = 0
    for jname, q in POLICY_JOINT_DEFAULTS.items():
        prim = stage.GetPrimAtPath(f"{joints_base}/{jname}")
        if not prim.IsValid():
            continue
        for attr_name, val in [
            ("drive:angular:physics:targetPosition", float(np.degrees(q))),
            ("drive:angular:physics:stiffness",      float(_KP_SDK.get(jname, 100.0))),
            ("drive:angular:physics:damping",        float(_KD_SDK.get(jname,   5.0))),
        ]:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(val)
        seeded += 1
    print(f"[INFO] Authored kp/kd/target on {seeded}/{len(POLICY_JOINT_DEFAULTS)} joint prims.")

    if args.stance_ankle:
        from pxr import Sdf
        stiffened = 0
        for joint_name, overrides in _STANCE_ANKLE_OVERRIDES.items():
            prim = stage.GetPrimAtPath(f"{joints_base}/{joint_name}")
            if not prim.IsValid():
                continue
            for attr_name, val, type_ in [
                ("drive:angular:physics:stiffness", overrides["kp"],        Sdf.ValueTypeNames.Float),
                ("drive:angular:physics:damping",   overrides["kd"],        Sdf.ValueTypeNames.Float),
                ("drive:angular:physics:maxForce",  overrides["max_force"], Sdf.ValueTypeNames.Float),
            ]:
                attr = prim.GetAttribute(attr_name)
                if not attr.IsValid():
                    attr = prim.CreateAttribute(attr_name, type_)
                attr.Set(float(val))
            stiffened += 1
        print(f"[INFO] Stance mode: stiffened {stiffened}/4 ankle joints "
              f"(kp=200 pitch / 150 roll, max_force=200 Nm).")

    if args.pin_pelvis:
        _pin_articulation_base(stage, args.robot_root)
    if args.pin_feet:
        _pin_feet_to_world(stage, args.robot_root)

    sim = SimulationContext(
        physics_dt=args.physics_dt,
        rendering_dt=args.physics_dt * max(args.decimation, 1),
        stage_units_in_meters=1.0,
        backend="numpy",
    )
    sim.initialize_physics()
    sim.play()
    simulation_app.update(); simulation_app.update()

    art_root = _find_articulation_root(stage, args.robot_root)
    if art_root is None:
        print(f"[ERROR] ArticulationRootAPI not found under {args.robot_root}")
        _dump_stage_diagnostic(stage, args.robot_root)
        simulation_app.close()
        return
    art = SingleArticulation(prim_path=art_root, name="h1_2")
    art.initialize()
    dof_names = list(art.dof_names)
    print(f"[INFO] Articulation: {art_root}  DOFs: {len(dof_names)}")

    # Snap joints to defaults before any physics has acted.
    qpos = _to_numpy(art.get_joint_positions()).astype(np.float32, copy=True)
    for jname, q in POLICY_JOINT_DEFAULTS.items():
        idx = dof_names.index(jname) if jname in dof_names else None
        if idx is not None:
            qpos[idx] = q
    art.set_joint_positions(qpos)
    try:
        art.set_joint_velocities(np.zeros_like(qpos))
    except Exception:
        pass
    art.apply_action(ArticulationAction(joint_positions=qpos.copy()))

    # Build the live target dict (joint_name -> target rad).  Anything not in
    # POLICY_JOINT_DEFAULTS (e.g., hand finger joints if present) stays at the
    # current articulation pose so we don't snap them.
    joint_targets: dict[str, float] = {}
    for i, jname in enumerate(dof_names):
        if jname in POLICY_JOINT_DEFAULTS:
            joint_targets[jname] = float(POLICY_JOINT_DEFAULTS[jname])
        else:
            joint_targets[jname] = float(qpos[i])
    defaults_snapshot = dict(joint_targets)

    state = _KeyState()
    listener = kb.Listener(on_press=state.on_press, on_release=state.on_release, daemon=True)
    listener.start()
    print("[INFO] Keyboard listener started.  Focus this terminal or the sim window.")

    control_dt = args.physics_dt * args.decimation  # ~0.02 s @ 200 Hz / decim 4
    step_per_iter_base = args.step_rate * control_dt
    rate_multiplier = 1.0

    try:
        while simulation_app.is_running() and not state.quit_requested():
            # Apply step-rate changes.
            rate_multiplier *= state.consume_rate_change()
            step_per_iter = step_per_iter_base * rate_multiplier

            # Reset?
            if state.pop_reset():
                joint_targets.update(defaults_snapshot)
                print("[INFO] Reset all joint targets to default pose.")

            # Per-joint key updates.
            held = state.held()
            for key, (jname, sign) in KEY_BINDINGS_SINGLE_JOINT.items():
                if key in held:
                    lo, hi = JOINT_LIMITS.get(jname, (-3.14, 3.14))
                    joint_targets[jname] = float(np.clip(
                        joint_targets[jname] + sign * step_per_iter, lo, hi
                    ))
            # Coordinated leg-group updates.
            for key, group in KEY_BINDINGS_LEG_GROUP.items():
                if key in held:
                    for jname, sign in group.items():
                        lo, hi = JOINT_LIMITS.get(jname, (-3.14, 3.14))
                        joint_targets[jname] = float(np.clip(
                            joint_targets[jname] + sign * step_per_iter, lo, hi
                        ))

            # Build full target vector and apply.
            target_vec = np.array(
                [joint_targets[n] for n in dof_names], dtype=np.float32
            )
            art.apply_action(ArticulationAction(joint_positions=target_vec))

            for _ in range(args.decimation):
                sim.step(render=False)
            sim.render()

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C — stopping.")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        try:
            sim.stop()
        except Exception:
            pass
        usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
