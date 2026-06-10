#!/usr/bin/env python3
"""H1-2 stand-still + custom-motion sandbox for the warehouse scene.

Loads the warehouse USD, wraps the H1-2 articulation with the modern
``isaacsim.core.prims.SingleArticulation`` API, and authors per-joint PD
gains (``_KP_SDK`` / ``_KD_SDK``, matching ``Krumi_python_sdk_unitree/
example/h1_2/pick_cone_palms.py``) on the joint drive USD attributes
*before* physics initializes — that's how kp/kd are "sent" to PhysX in this
Isaac Sim build (``SingleArticulation.set_gains()`` is not exposed).

Run modes (mutually compatible):

* default — pure PD stand-still.  Every joint is pulled toward its
  ``POLICY_JOINT_DEFAULTS`` value at the SDK-matched gains.  No RL policy
  loaded; no scripted motion.  Use this as a base when authoring custom
  arm motions.
* ``--grab`` — overlay the scripted arm-reach + Surface Gripper sequence
  on top of the held leg/torso pose.  Mirrors the C++/Python SDK
  ``pick_cone_palms`` choreography.
* ``--policy <policy.pt>`` — load a JIT-traced RSL-RL checkpoint and
  drive the 21-DOF locomotion action subset.  Combine with ``--cmd_vx``
  to send a non-zero velocity command.  ``--grab`` still works on top.

Usage:
    python3 scene/joint_motion.py                           # stand still
    python3 scene/joint_motion.py --grab                    # cone-grab demo
    python3 scene/joint_motion.py --policy logs/.../policy.pt --cmd_vx 0.4
"""

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


DEFAULT_POLICY_PATH = os.path.join(
    PROJECT_ROOT, "logs", "rsl_rl", "h12_velocity_legonly",
    "2026-05-19_11-14-40", "exported", "policy.pt",
)


# Joint pose the policy was trained around (matches
# `_H12_FTP_FLOATING_DEFAULT_JOINT_POS` in `robots/unitree.py`).  The policy
# observes `joint_pos - default`, so the runtime defaults must match these
# exactly or the policy will start in a permanently "leaning" state.
POLICY_JOINT_DEFAULTS = {
    "left_hip_yaw_joint":         0.0,
    "left_hip_roll_joint":        0.0,
    "left_hip_pitch_joint":      -0.20,
    "left_knee_joint":            0.42,
    "left_ankle_pitch_joint":    -0.23,
    "left_ankle_roll_joint":      0.0,
    "right_hip_yaw_joint":        0.0,
    "right_hip_roll_joint":       0.0,
    "right_hip_pitch_joint":     -0.20,
    "right_knee_joint":           0.42,
    "right_ankle_pitch_joint":   -0.23,
    "right_ankle_roll_joint":     0.0,
    "torso_joint":                0.0,
    "left_shoulder_pitch_joint":  0.35,
    "left_shoulder_roll_joint":   0.18,
    "left_shoulder_yaw_joint":    0.0,
    "left_elbow_joint":           0.87,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_joint":          0.87,
    "left_wrist_roll_joint":      0.0,
    "left_wrist_pitch_joint":     0.0,
    "left_wrist_yaw_joint":       0.0,
    "right_wrist_roll_joint":     0.0,
    "right_wrist_pitch_joint":    0.0,
    "right_wrist_yaw_joint":      0.0,
}

# Action subset, in the order the policy emits.  Matches the legonly task
# (Isaac-H12-Velocity-Legonly-v0): 12 leg DOF + 1 torso DOF = 13 actions and
# 13 joint_pos / joint_vel observations.
LOCOMOTION_JOINT_NAMES = [
    "left_hip_yaw_joint",   "left_hip_roll_joint",   "left_hip_pitch_joint",
    "left_knee_joint",      "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_yaw_joint",  "right_hip_roll_joint",  "right_hip_pitch_joint",
    "right_knee_joint",     "right_ankle_pitch_joint","right_ankle_roll_joint",
    "torso_joint",
]

# Arm joints are NOT in the policy action subset (legonly).  They are held at
# default by their actuator config; the --grab overlay drives them via a
# separate ArticulationAction with `arm_dof_indices` resolved at runtime.
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
]
ARM_INDICES_IN_LOCOMOTION: list[int] = []  # empty: arms not in policy subset

POLICY_ACTION_SCALE = 0.5  # matches `self.actions.joint_pos.scale = 0.5`

# Match training-time timestepping exactly so the policy sees the same
# closed-loop dynamics it was rewarded on:
#   physics_dt   = 1/200 s   (env_cfg.sim.dt)
#   decimation   = 4         (env_cfg.decimation)
#   policy rate  = 50 Hz     (= 1 / (physics_dt * decimation))
PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
POLICY_HZ  = 1.0 / (PHYSICS_DT * DECIMATION)


# Per-joint PD gains used during PPO training (H12_CFG_WITH_INSPIRE_WHOLEBODY
# actuators + H12RoughEnvCfg feet override).  Used by scene/walk_policy.py at
# policy deployment time so motion looks the same as in sim.
#
# How gains are "sent" to Isaac Sim
# ---------------------------------
# SingleArticulation.set_gains() is not exposed in our Isaac Sim build, so we
# author the joint drive properties directly on the USD stage *before* the
# articulation initializes — PhysX reads them at creation time:
#
#     drive:angular:physics:stiffness    -> kp
#     drive:angular:physics:damping      -> kd
#     drive:angular:physics:targetPosition -> initial target
#     drive:angular:physics:maxForce     -> effort limit
#
# Search for "Pre-seed drive gains" in main() to see this in action.
_KP = {
    "left_hip_yaw_joint":  150.0, "left_hip_roll_joint":  150.0,
    "left_hip_pitch_joint":200.0, "left_knee_joint":      200.0,
    "right_hip_yaw_joint": 150.0, "right_hip_roll_joint": 150.0,
    "right_hip_pitch_joint":200.0,"right_knee_joint":     200.0,
    "left_ankle_pitch_joint": 20.0, "left_ankle_roll_joint": 20.0,
    "right_ankle_pitch_joint":20.0, "right_ankle_roll_joint":20.0,
    "torso_joint": 200.0,
    "left_shoulder_pitch_joint": 100.0, "left_shoulder_roll_joint": 100.0,
    "right_shoulder_pitch_joint":100.0, "right_shoulder_roll_joint":100.0,
    "left_shoulder_yaw_joint":   50.0,  "left_elbow_joint":   50.0,
    "right_shoulder_yaw_joint":  50.0,  "right_elbow_joint":  50.0,
    "left_wrist_roll_joint":  40.0, "left_wrist_pitch_joint": 40.0, "left_wrist_yaw_joint": 40.0,
    "right_wrist_roll_joint": 40.0, "right_wrist_pitch_joint":40.0, "right_wrist_yaw_joint":40.0,
}
_KD = {
    "left_hip_yaw_joint":  5.0, "left_hip_roll_joint":  5.0,
    "left_hip_pitch_joint":5.0, "left_knee_joint":      5.0,
    "right_hip_yaw_joint": 5.0, "right_hip_roll_joint": 5.0,
    "right_hip_pitch_joint":5.0,"right_knee_joint":     5.0,
    "left_ankle_pitch_joint": 2.5, "left_ankle_roll_joint": 2.5,
    "right_ankle_pitch_joint":2.5, "right_ankle_roll_joint":2.5,
    "torso_joint": 5.0,
    "left_shoulder_pitch_joint": 2.0, "left_shoulder_roll_joint": 2.0,
    "right_shoulder_pitch_joint":2.0, "right_shoulder_roll_joint":2.0,
    "left_shoulder_yaw_joint":   2.0,  "left_elbow_joint":   2.0,
    "right_shoulder_yaw_joint":  2.0,  "right_elbow_joint":  2.0,
    "left_wrist_roll_joint":  2.0, "left_wrist_pitch_joint": 2.0, "left_wrist_yaw_joint": 2.0,
    "right_wrist_roll_joint": 2.0, "right_wrist_pitch_joint":2.0, "right_wrist_yaw_joint":2.0,
}


# Hardware-matched PD gains for the arms + waist, copied from
# Krumi_python_sdk_unitree/example/h1_2/pick_cone_palms.py (ARM_KP / ARM_KD).
# Motion scripts that should transfer 1:1 to the real H1-2 over rt/arm_sdk
# (e.g. the cone-grab demo in this file) author these so the joint response
# matches the on-robot control loop without re-tuning.
#
# Legs inherit the training-time values from _KP / _KD since the Krumi SDK
# leaves leg control to FSM 204 (LocoClient) and never sets per-joint gains.
_KP_SDK = {
    **_KP,
    "torso_joint": 150.0,
    "left_shoulder_pitch_joint":  80.0, "right_shoulder_pitch_joint":  80.0,
    "left_shoulder_roll_joint":   80.0, "right_shoulder_roll_joint":   80.0,
    "left_shoulder_yaw_joint":    60.0, "right_shoulder_yaw_joint":    60.0,
    "left_elbow_joint":           60.0, "right_elbow_joint":           60.0,
    "left_wrist_roll_joint":      30.0, "right_wrist_roll_joint":      30.0,
    "left_wrist_pitch_joint":     30.0, "right_wrist_pitch_joint":     30.0,
    "left_wrist_yaw_joint":       30.0, "right_wrist_yaw_joint":       30.0,
}
_KD_SDK = {
    **_KD,
    "torso_joint": 2.0,
    "left_shoulder_pitch_joint":  2.0, "right_shoulder_pitch_joint":  2.0,
    "left_shoulder_roll_joint":   2.0, "right_shoulder_roll_joint":   2.0,
    "left_shoulder_yaw_joint":    1.5, "right_shoulder_yaw_joint":    1.5,
    "left_elbow_joint":           1.5, "right_elbow_joint":           1.5,
    "left_wrist_roll_joint":      1.0, "right_wrist_roll_joint":      1.0,
    "left_wrist_pitch_joint":     1.0, "right_wrist_pitch_joint":     1.0,
    "left_wrist_yaw_joint":       1.0, "right_wrist_yaw_joint":       1.0,
}

# Training-time joint torque limits (N·m).  Source: H12_CFG_WITH_INSPIRE_WHOLEBODY
# in robots/unitree.py:1340+ (effort_limit_sim), with the env-cfg override
# at tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py:237-243 applied for the
# ankle joints (raw 35 → 45 N·m).  When the deploy USD authors a lower
# `drive:angular:physics:maxForce` than these values, the policy's leg-swing
# torques saturate → tracking error grows → policy diverges → fall.
_TAU_MAX = {
    "left_hip_yaw_joint":     88.0,  "right_hip_yaw_joint":    88.0,
    "left_hip_roll_joint":   139.0,  "right_hip_roll_joint":  139.0,
    "left_hip_pitch_joint":   88.0,  "right_hip_pitch_joint":  88.0,
    "left_knee_joint":       139.0,  "right_knee_joint":      139.0,
    "left_ankle_pitch_joint": 45.0,  "right_ankle_pitch_joint":45.0,
    "left_ankle_roll_joint":  45.0,  "right_ankle_roll_joint": 45.0,
    "torso_joint":            88.0,
    "left_shoulder_pitch_joint": 25.0, "right_shoulder_pitch_joint":25.0,
    "left_shoulder_roll_joint":  25.0, "right_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint":   25.0, "right_shoulder_yaw_joint":  25.0,
    "left_elbow_joint":          25.0, "right_elbow_joint":         25.0,
    "left_wrist_roll_joint":     25.0, "right_wrist_roll_joint":    25.0,
    "left_wrist_pitch_joint":     5.0, "right_wrist_pitch_joint":    5.0,
    "left_wrist_yaw_joint":       5.0, "right_wrist_yaw_joint":      5.0,
}

# Training-time joint armature (rotor inertia reflected to joint side).
# Source: robots/unitree.py:1371,1379,1393, etc.  Uniform 0.01 across all
# robot joints in the wholebody actuator config.
_ARMATURE_DEFAULT = 0.01


# Stance-mode ankle stiffening.  The default pose has bent knees and a slight
# forward ankle pitch (~-13°); the training-time ankle gains (kp=20, effort
# limit 45 Nm) are far too soft to hold that statically — the body tips
# forward at the ankle before any policy can react.  These overrides apply
# only in the no-policy (stand-still / scripted-motion) path so a deployed
# RL policy still sees the gains it was trained with.
_STANCE_ANKLE_OVERRIDES = {
    "left_ankle_pitch_joint":  {"kp": 200.0, "kd": 8.0, "max_force": 200.0},
    "right_ankle_pitch_joint": {"kp": 200.0, "kd": 8.0, "max_force": 200.0},
    "left_ankle_roll_joint":   {"kp": 150.0, "kd": 6.0, "max_force": 200.0},
    "right_ankle_roll_joint":  {"kp": 150.0, "kd": 6.0, "max_force": 200.0},
}


# ── Cone-grab arm overlay (only the 8 arm joints) ────────────────────────────
# Each entry: (hold_seconds, target_arm_pose, gripper_command).
# gripper_command ∈ [-1, 1]:  < -0.3 = open, > +0.3 = close, else idle.
_REST_ARMS = {
    "left_shoulder_pitch_joint":  POLICY_JOINT_DEFAULTS["left_shoulder_pitch_joint"],
    "left_shoulder_roll_joint":   POLICY_JOINT_DEFAULTS["left_shoulder_roll_joint"],
    "left_shoulder_yaw_joint":    0.0,
    "left_elbow_joint":           POLICY_JOINT_DEFAULTS["left_elbow_joint"],
    "right_shoulder_pitch_joint": POLICY_JOINT_DEFAULTS["right_shoulder_pitch_joint"],
    "right_shoulder_roll_joint":  POLICY_JOINT_DEFAULTS["right_shoulder_roll_joint"],
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_joint":          POLICY_JOINT_DEFAULTS["right_elbow_joint"],
}

_REACH_ARMS = {
    "left_shoulder_pitch_joint":  -1.6,
    "left_shoulder_roll_joint":    0.12,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.25,
    "right_shoulder_pitch_joint": -1.6,
    "right_shoulder_roll_joint":  -0.12,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.25,
}

GRAB_SEQUENCE = [
    (2.0, _REST_ARMS,  -1.0),   # arms at rest, gripper open
    (2.0, _REACH_ARMS, -1.0),   # reach forward, still open
    (1.0, _REACH_ARMS,  1.0),   # close gripper while reaching
    (2.0, _REST_ARMS,   1.0),   # pull back while gripper holds
    (1.5, _REST_ARMS,  -1.0),   # release
]


# ── Math + USD helpers ───────────────────────────────────────────────────────

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_pose(a: dict, b: dict, t: float) -> dict:
    return {k: a[k] * (1.0 - t) + b[k] * t for k in a}


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _quat_rotate_inverse(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """Rotate v by the inverse of unit quaternion q = (w, x, y, z)."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    qv = np.array([-x, -y, -z], dtype=np.float32)
    v = v_xyz.astype(np.float32, copy=False)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _find_articulation_root(stage, search_path: str):
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(search_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None


def _dump_stage_diagnostic(stage, search_path: str) -> None:
    from pxr import UsdPhysics
    print(f"[DIAG] Searched for ArticulationRootAPI under: {search_path}")
    print("[DIAG] All ArticulationRoots in stage:")
    found_any = False
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  - {prim.GetPath()}  (type={prim.GetTypeName()})")
            found_any = True
    if not found_any:
        print("  (none)")
    world = stage.GetPrimAtPath("/World")
    print("[DIAG] Top-level prims under /World:")
    if not world.IsValid():
        print("  (/World invalid)")
        return
    for child in world.GetChildren():
        marker = "  [payload/ref]" if (child.GetPayloads() or child.GetReferences()) else ""
        print(f"  - {child.GetPath()}  (type={child.GetTypeName()}){marker}")


def _log_omnigraphs(stage) -> int:
    """Print every OmniGraph / ActionGraph prim found in the stage.

    Catches:
      * Prims with type ``OmniGraph``, ``ComputeGraph`` or any type whose
        name contains ``Graph`` (covers ``OmniGraphNode`` and friends).
      * Prims with the applied schema ``OmniGraphAPI`` or ``ComputeGraphAPI``
        (action graphs authored as ``Scope`` with the schema applied).
      * Prims whose *name* (not type) contains ``Graph`` — a last-ditch catch
        for ``Scope`` prims that humans typed as ``ActionGraph`` etc.
    """
    found = []
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        name = prim.GetName()
        applied = []
        try:
            applied = list(prim.GetAppliedSchemas())
        except Exception:
            pass
        type_hit  = type_name in ("OmniGraph", "ComputeGraph") or "Graph" in type_name
        sch_hit   = any("Graph" in s for s in applied)
        name_hit  = "Graph" in name and not type_hit
        if type_hit or sch_hit or name_hit:
            tag = ""
            if sch_hit and not type_hit:
                tag = f"  +schemas={[s for s in applied if 'Graph' in s]}"
            elif name_hit:
                tag = "  (matched by name only)"
            found.append(f"  - {prim.GetPath()}  type={type_name or '<none>'}{tag}")
    if found:
        print(f"[INFO] OmniGraph / ActionGraph prims in stage ({len(found)}):")
        for line in found:
            print(line)
    else:
        print("[INFO] No OmniGraph / ActionGraph prims found in the loaded stage.")
        print("       If you authored one in the GUI: confirm it's on the root "
              "layer of the saved USD and re-run.  See `python3 -c \"from pxr "
              "import Usd; ...\"` one-liner in chat for an on-disk check.")
    return len(found)


def _enable_omnigraph_extensions() -> None:
    """Load the OmniGraph runtime extensions explicitly.

    ``AppLauncher`` boots a minimal extension set; if your action graph uses
    nodes from ``omni.graph.action``, ``omni.graph.nodes``, or
    ``omni.graph.scriptnode``, those need to be enabled or the graph stays
    inert (the prims exist in the stage but never tick).
    """
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        for ext in (
            "omni.graph",
            "omni.graph.core",
            "omni.graph.action",
            "omni.graph.nodes",
            "omni.graph.scriptnode",
            "omni.graph.window.action",  # editor UI; harmless if headless
        ):
            try:
                mgr.set_extension_enabled_immediate(ext, True)
            except Exception:
                pass
        print("[INFO] OmniGraph extensions enabled (omni.graph[, .action, .nodes, .scriptnode]).")
    except Exception as exc:
        print(f"[WARN] Could not enable OmniGraph extensions: {exc}")


def _log_prim_world_pose(stage, prim_path: str, label: str) -> None:
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] {label} prim not found at {prim_path}")
        return
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    print(f"[INFO] {label} ({prim_path}) world pos: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")


def _pin_articulation_base(stage, root_path: str) -> bool:
    """UsdPhysics.FixedJoint between world and the articulation's pelvis."""
    from pxr import UsdPhysics, UsdGeom, Sdf, Gf
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"[WARN] Articulation root prim invalid at {root_path}; cannot pin.")
        return False

    candidates = [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(root_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not candidates:
        print(f"[WARN] No RigidBodyAPI prims under {root_path}; cannot pin.")
        return False

    target = next((p for p in candidates if "pelvis" in str(p.GetPath()).lower()), None)
    if target is None:
        target = min(candidates, key=lambda p: len(str(p.GetPath()).split("/")))

    world_mat = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(0)
    world_pos = world_mat.ExtractTranslation()
    world_rot = world_mat.ExtractRotationQuat()
    quat_f = Gf.Quatf(
        float(world_rot.GetReal()),
        Gf.Vec3f(*[float(c) for c in world_rot.GetImaginary()]),
    )

    pin_path = Sdf.Path("/World/_pelvis_pin")
    if stage.GetPrimAtPath(pin_path).IsValid():
        stage.RemovePrim(pin_path)

    joint = UsdPhysics.FixedJoint.Define(stage, pin_path)
    joint.CreateBody1Rel().SetTargets([target.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
    joint.CreateLocalRot0Attr().Set(quat_f)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    print(f"[INFO] Pinned {target.GetPath()} to world at "
          f"({world_pos[0]:.3f},{world_pos[1]:.3f},{world_pos[2]:.3f}).")
    return True


def _create_surface_gripper(stage, parent_link_path: str, name: str, *,
                             forward_offset: float = 0.05,
                             max_grip_distance: float = 0.15,
                             coaxial_force_limit: float = 250.0,
                             shear_force_limit: float = 250.0,
                             retry_interval: float = 0.1):
    """Author a SurfaceGripper prim attached to ``parent_link_path``.

    Creates one D6 attachment-point joint with ``body1`` set to the hand link;
    ``body0`` is left unset so the SDK can dynamically bind a gripped object.
    Returns the gripper prim path, or None if ``parent_link_path`` doesn't exist.
    """
    from pxr import UsdPhysics, UsdGeom, Sdf, Gf

    try:
        from usd.schema.isaac import robot_schema
    except ImportError as exc:
        print(f"[WARN] usd.schema.isaac unavailable ({exc}); skipping Surface Gripper.")
        return None

    parent = stage.GetPrimAtPath(parent_link_path)
    if not parent.IsValid():
        print(f"[WARN] Surface gripper parent link not found at {parent_link_path}; skipped.")
        return None

    joints_scope_path = Sdf.Path(f"{parent_link_path}/{name}_joints")
    if not stage.GetPrimAtPath(joints_scope_path).IsValid():
        UsdGeom.Scope.Define(stage, joints_scope_path)

    joint_path = Sdf.Path(f"{joints_scope_path}/attachment_d6")
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint_prim = joint.GetPrim()
    joint_prim.AddAppliedSchema("IsaacAttachmentPointAPI")

    joint.CreateBody1Rel().SetTargets([parent_link_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(forward_offset), 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint_prim.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)
    joint_prim.CreateAttribute("isaac:forwardAxis", Sdf.ValueTypeNames.Token).Set("X")
    joint_prim.CreateAttribute("isaac:clearanceOffset", Sdf.ValueTypeNames.Float).Set(0.005)

    gripper_path = Sdf.Path(f"{parent_link_path}/{name}")
    if stage.GetPrimAtPath(gripper_path).IsValid():
        stage.RemovePrim(gripper_path)
    gripper_prim = robot_schema.CreateSurfaceGripper(stage, str(gripper_path))

    rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)
    if rel:
        rel.SetTargets([joint_path])

    for attr_name, value in [
        (robot_schema.Attributes.MAX_GRIP_DISTANCE.name,   max_grip_distance),
        (robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name, coaxial_force_limit),
        (robot_schema.Attributes.SHEAR_FORCE_LIMIT.name,   shear_force_limit),
        (robot_schema.Attributes.RETRY_INTERVAL.name,      retry_interval),
    ]:
        attr = gripper_prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(value)

    print(f"[INFO] Authored SurfaceGripper at {gripper_path} (max_grip={max_grip_distance} m)")
    return str(gripper_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="H1-2 walking-policy + cone-grab demo.")
    parser.add_argument("--scene", type=str,
                        default=os.path.join(PROJECT_ROOT, "scene", "h1-2_cones.usd"),
                        help="Path to scene USD.")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY_PATH,
                        help="Path to a policy file.  Accepts either a JIT-traced policy.pt "
                             "(from `play.py`'s exported/ subdir) OR a raw RSL-RL checkpoint "
                             "(model_N.pt) — for the raw case we reconstruct the actor MLP "
                             "in-place.  Pass --policy \"\" to disable the policy and just "
                             "hold the default pose via PD (stand-still).")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Outer-loop tick rate in Hz (policy step rate).  "
                             "Default matches training (50 Hz = 1 / (physics_dt * decimation)).  "
                             "Capped internally at POLICY_HZ — going faster than 50 Hz would "
                             "drift from the training-time control contract.")
    parser.add_argument("--cmd_vx", type=float, default=1.0,
                        help="Forward velocity command (m/s) sent to the policy.")
    parser.add_argument("--cmd_vy", type=float, default=0.0,
                        help="Lateral velocity command (m/s).")
    parser.add_argument("--cmd_wz", type=float, default=0.0,
                        help="Yaw-rate command (rad/s).")
    parser.add_argument("--task", type=str, default="auto",
                        choices=("auto", "legonly", "squat"),
                        help="Which observation layout to feed the policy.  "
                             "'legonly' = 51 dims (base_lin_vel, base_ang_vel, "
                             "proj_g, velocity_commands, joint_pos_rel, "
                             "joint_vel_rel, last_action).  'squat' = 50 dims "
                             "(... target_height, pelvis_height ... in place "
                             "of velocity_commands).  'auto' = probe the "
                             "policy's input dim and pick whichever fits.")
    parser.add_argument("--target_height", type=float, default=0.78,
                        help="Squat-task target pelvis height (m).  Only used "
                             "when --task=squat (or auto-detected as squat).")
    parser.add_argument("--action_smoothing", type=float, default=0.0,
                        help="EMA smoothing on the PD target_q (NOT on the "
                             "obs `last_action` slot — that still gets the "
                             "raw policy output, matching training).  "
                             "0.0=passthrough (default).  0.5=half-and-half.  "
                             "0.9=heavily smoothed, very damped.  Useful for "
                             "the Legonly policy at low cmd_vx, where it "
                             "wasn't trained to stand still and naturally "
                             "twitches.  Trade-off: smoothed targets lag the "
                             "policy → slower recovery from disturbances.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Run duration seconds (0 = forever).")
    parser.add_argument("--grab", action="store_true",
                        help="Overlay the scripted arm reach + gripper sequence on top of the policy.")
    parser.add_argument("--cone_path", type=str,
                        default="/World/heavydutytrafficcone_a04_01",
                        help="Cone USD path (logged at startup for reach tuning).")
    parser.add_argument("--robot_root", type=str,
                        default="/World/h1_2_physics",
                        help="USD path of the articulation root prim.")
    parser.add_argument("--left_gripper_link", type=str,
                        default="/World/h1_2_physics/left_hand_yaw_link/L_hand_base_link",
                        help="USD path of the left hand link to attach a Surface Gripper to.")
    parser.add_argument("--right_gripper_link", type=str,
                        default="/World/h1_2_physics/right_hand_yaw_link/R_hand_base_link",
                        help="USD path of the right hand link to attach a Surface Gripper to.")
    parser.add_argument("--no_gripper", action="store_true",
                        help="Skip authoring Surface Gripper prims and sending gripper commands.")
    parser.add_argument("--pin_pelvis", dest="pin_pelvis", action="store_true", default=False,
                        help="Pin the pelvis to world (debug aid; the policy will not balance).")
    parser.add_argument("--no_pin_pelvis", dest="pin_pelvis", action="store_false",
                        help="Do not pin the pelvis (default).")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if "--headless" not in sys.argv:
        args.headless = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Enable OmniGraph extensions BEFORE we open the stage so any Action Graph
    # prims saved in the USD start ticking on stage load.
    _enable_omnigraph_extensions()

    import omni.usd
    import torch
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    scene_path = os.path.abspath(os.path.expanduser(args.scene))
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    usd_ctx = omni.usd.get_context()
    if not usd_ctx.open_stage(scene_path):
        raise RuntimeError(f"Failed to open: {scene_path}")

    simulation_app.update()
    simulation_app.update()
    stage = usd_ctx.get_stage()

    _log_prim_world_pose(stage, args.cone_path, "Middle cone")
    _log_omnigraphs(stage)

    if args.pin_pelvis:
        _pin_articulation_base(stage, args.robot_root)

    gripper_paths = []
    if not args.no_gripper:
        for link_path, name in [
            (args.left_gripper_link,  "surface_gripper"),
            (args.right_gripper_link, "surface_gripper"),
        ]:
            p = _create_surface_gripper(stage, link_path, name, forward_offset=0.05)
            if p:
                gripper_paths.append(p)

    # Pre-seed drive gains + initial position targets so PhysX picks them up
    # at articulation creation (first physics step).  Arm + waist gains come
    # from the Krumi hardware SDK (pick_cone_palms.py) so motions transfer
    # 1:1 to the real H1-2; leg gains stay at the training-time values.
    joints_base = f"{args.robot_root}/joints"
    kp_map, kd_map = _KP_SDK, _KD_SDK
    seeded = 0
    # Also author maxForce (training-time effort_limit_sim) and armature so
    # PhysX joint dynamics match training exactly.  Without maxForce, the USD
    # may impose a lower torque cap than training → leg-swing torques saturate
    # → tracking error grows → policy diverges → stable-then-collapse fall.
    from pxr import Sdf as _Sdf_seed
    for joint_name, q in POLICY_JOINT_DEFAULTS.items():
        prim = stage.GetPrimAtPath(f"{joints_base}/{joint_name}")
        if not prim.IsValid():
            continue
        for attr_name, val in [
            ("drive:angular:physics:targetPosition", float(q)),
            ("drive:angular:physics:stiffness",      float(kp_map.get(joint_name, 100.0))),
            ("drive:angular:physics:damping",        float(kd_map.get(joint_name,   5.0))),
            ("drive:angular:physics:maxForce",       float(_TAU_MAX.get(joint_name, 200.0))),
            ("physxJoint:armature",                  float(_ARMATURE_DEFAULT)),
        ]:
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                attr = prim.CreateAttribute(attr_name, _Sdf_seed.ValueTypeNames.Float)
            if attr.IsValid():
                attr.Set(val)
        seeded += 1
    print(f"[INFO] Authored kp/kd/target/maxForce/armature on {seeded}/"
          f"{len(POLICY_JOINT_DEFAULTS)} joint prims "
          f"(maxForce + armature match training-time ImplicitActuatorCfg).")

    # Stand-still / scripted-motion path: stiffen ankles so the bent-knee
    # default pose doesn't tip forward.  Skipped when a policy is provided —
    # the trained RL policy expects the softer training-time ankle gains.
    if not args.policy:
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

    # Author training-time friction on the robot feet AND ensure the ground
    # has matching friction.  Training applies:
    #   robot rigid bodies: randomize_rigid_body_material  → static=0.6, dynamic=0.6
    #   ground (flat plane):  RigidBodyMaterialCfg default → static=0.5, dynamic=0.5
    # The PhysX combine mode is "average", so effective foot/ground friction
    # ~ 0.55.  If the deploy scene's robot USD authored lower foot friction
    # (or none), feet slip and the policy can't propel → wobble + premature
    # fall, exactly the failure mode here.  Force-set both sides.
    from pxr import UsdGeom, UsdPhysics, UsdShade, Sdf
    _MATERIAL_PATH = "/World/_joint_motion_physics_material"
    if not stage.GetPrimAtPath(_MATERIAL_PATH).IsValid():
        mat_prim = UsdShade.Material.Define(stage, _MATERIAL_PATH).GetPrim()
        UsdPhysics.MaterialAPI.Apply(mat_prim)
        for attr_name, val in [
            ("physics:staticFriction",  0.6),
            ("physics:dynamicFriction", 0.6),
            ("physics:restitution",     0.0),
        ]:
            a = mat_prim.GetAttribute(attr_name)
            if not a.IsValid():
                a = mat_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            a.Set(float(val))
        print(f"[INFO] Authored {_MATERIAL_PATH} (static=0.6 dynamic=0.6 restitution=0).")

    # Bind it to ankle_roll_link colliders (the actual foot links) and to any
    # ground/floor-named prim in the stage.
    bound_feet, bound_ground = 0, 0
    target_prims: list = []
    for prim in stage.Traverse():
        path = str(prim.GetPath()).lower()
        if "ankle_roll_link" in path and prim.IsA(UsdGeom.Imageable):
            target_prims.append((prim, "foot"))
        if any(tok in path for tok in ("flat_plane", "ground", "floor", "terrain")) \
                and prim.IsA(UsdGeom.Imageable):
            target_prims.append((prim, "ground"))
    mat = UsdShade.Material(stage.GetPrimAtPath(_MATERIAL_PATH))
    for prim, kind in target_prims:
        try:
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(mat, materialPurpose="physics")
            if kind == "foot":
                bound_feet += 1
            else:
                bound_ground += 1
        except Exception as exc:
            print(f"[WARN] Could not bind physics material to {prim.GetPath()}: {exc}")
    print(f"[INFO] Bound physics material to {bound_feet} foot prim(s), "
          f"{bound_ground} ground prim(s).")

    # Build a proper SimulationContext so physics_dt + decimation match the
    # training-time env config exactly.  Without this, the USD stage's default
    # timestep drives physics (often 60 Hz) and the policy sees dynamics it
    # was never rewarded against — manifests as wobble, weak propulsion, or
    # premature falls even though play.py behaves correctly with the same
    # checkpoint.
    sim = SimulationContext(
        physics_dt=PHYSICS_DT,
        rendering_dt=PHYSICS_DT * DECIMATION,
        stage_units_in_meters=1.0,
        backend="numpy",
    )
    sim.initialize_physics()
    sim.play()
    simulation_app.update()
    simulation_app.update()

    art_root = _find_articulation_root(stage, args.robot_root)
    if art_root is None:
        print(f"[ERROR] ArticulationRootAPI not found under {args.robot_root}")
        _dump_stage_diagnostic(stage, args.robot_root)
        simulation_app.close()
        return

    art = SingleArticulation(prim_path=art_root, name="h1_2")
    art.initialize()

    dof_names = list(art.dof_names)
    name_to_dof = {n: i for i, n in enumerate(dof_names)}
    print(f"[INFO] Articulation: {art_root}  DOFs: {len(dof_names)}")

    try:
        locomotion_dof_indices = np.array(
            [name_to_dof[n] for n in LOCOMOTION_JOINT_NAMES], dtype=np.int64
        )
    except KeyError as exc:
        print(f"[ERROR] Locomotion joint missing from articulation: {exc}")
        simulation_app.close()
        return

    locomotion_defaults = np.array(
        [POLICY_JOINT_DEFAULTS[n] for n in LOCOMOTION_JOINT_NAMES], dtype=np.float32
    )

    # Resolve arm DOF indices for the --grab overlay.  When ARM_INDICES_IN_LOCOMOTION
    # is empty (legonly: arms not in policy subset), the grab block applies arm
    # targets through this separate ArticulationAction so the legs can keep
    # following the policy output without interference.
    arm_dof_indices_np = None
    arm_present_names: list[str] = []
    if not ARM_INDICES_IN_LOCOMOTION:
        resolved = []
        for n in ARM_JOINT_NAMES:
            idx = name_to_dof.get(n)
            if idx is not None:
                resolved.append(idx)
                arm_present_names.append(n)
        if resolved:
            arm_dof_indices_np = np.array(resolved, dtype=np.int64)

    # Teleport joints to the policy default pose before gravity acts.
    qpos = _to_numpy(art.get_joint_positions()).astype(np.float32, copy=True)
    for name, q in POLICY_JOINT_DEFAULTS.items():
        idx = name_to_dof.get(name)
        if idx is not None:
            qpos[idx] = q
    art.set_joint_positions(qpos)
    # Zero joint + base velocities so the first obs doesn't carry over a
    # transient from physics warmup / teleport.  IsaacLab's env.reset() does
    # the equivalent every episode; without it the first policy query sees
    # nonzero `dq` and `base_lin_vel` even though the pose looks right.
    try:
        art.set_joint_velocities(np.zeros_like(qpos))
    except Exception:
        pass
    try:
        art.set_linear_velocity(np.zeros(3, dtype=np.float32))
        art.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:
        pass
    art.apply_action(ArticulationAction(joint_positions=qpos.copy()))

    # Log spawn pose so a mismatched USD init (e.g. robot pelvis at z=0.5
    # instead of training's 1.0) is visible — that alone can produce the
    # "fall + jitter" failure mode even with a perfect policy.
    try:
        spawn_pos, spawn_quat = art.get_world_pose()
        sp = _to_numpy(spawn_pos)
        sq = _to_numpy(spawn_quat)
        print(f"[INFO] Spawn pose: pos=({sp[0]:+.3f}, {sp[1]:+.3f}, "
              f"{sp[2]:+.3f}) quat_wxyz=({sq[0]:+.3f}, {sq[1]:+.3f}, "
              f"{sq[2]:+.3f}, {sq[3]:+.3f})  (training trained from "
              f"pos=(0,0,1.000) quat=(1,0,0,0)).")
    except Exception:
        pass

    # Settle for ~0.2 s at the default pose so the obs the policy sees on
    # its first query is from a stationary robot, matching the reset-state
    # observation distribution training was rewarded against.
    settle_iters = int(0.2 * POLICY_HZ)   # 10 outer iters at 50 Hz
    for _ in range(settle_iters):
        art.apply_action(ArticulationAction(
            joint_positions=qpos.copy(),
            joint_indices=None,
        ))
        for _ in range(DECIMATION):
            sim.step(render=False)
    sim.render()
    print(f"[INFO] Settled robot for {settle_iters} control steps "
          f"({settle_iters / POLICY_HZ:.2f}s) before first policy query.")

    def _build_actor_mlp(actor_state_dict, obs_dim: int, action_dim: int):
        """Instantiate the **exact same** MLP class the trainer used
        (``rsl_rl.networks.mlp.MLP``) with H12FlatPPORunnerCfg's hidden sizes
        and load the saved actor weights into it.  No hand-rolled
        ``nn.Sequential`` — we keep the training actor MLP as-is.
        """
        from rsl_rl.networks.mlp import MLP
        actor = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=[512, 256, 128],
            activation="elu",
        )
        actor.load_state_dict(actor_state_dict)
        actor.eval()
        return actor

    policy = None
    policy_task = None   # "legonly" or "squat" — resolved below
    if args.policy:
        policy_path = os.path.abspath(os.path.expanduser(args.policy))
        if not os.path.exists(policy_path):
            print(f"[WARN] Policy not found at {policy_path}; falling back to PD stand-still.")
        else:
            obs_dim_expected = 3 + 3 + 3 + 3 + 3 * len(LOCOMOTION_JOINT_NAMES)
            action_dim_expected = len(LOCOMOTION_JOINT_NAMES)
            # 1) Try JIT (the canonical `play.py` export at `exported/policy.pt`)
            try:
                policy = torch.jit.load(policy_path, map_location="cpu")
                policy.eval()
                print(f"[INFO] Loaded JIT policy: {policy_path}")
            except Exception as jit_exc:
                # 2) Raw RSL-RL checkpoint (model_N.pt) — rebuild the actor MLP.
                try:
                    raw = torch.load(policy_path, map_location="cpu", weights_only=False)
                    full_sd = raw.get("model_state_dict", raw)
                    actor_sd = {
                        k[len("actor."):]: v for k, v in full_sd.items()
                        if k.startswith("actor.")
                    }
                    if not actor_sd:
                        raise RuntimeError("no 'actor.*' keys in state dict")
                    policy = _build_actor_mlp(actor_sd, obs_dim_expected, action_dim_expected)
                    print(f"[INFO] Loaded raw RSL-RL checkpoint: {policy_path}")
                    print(f"       Rebuilt actor MLP {obs_dim_expected}->512->256->128->{action_dim_expected}.")
                except Exception as raw_exc:
                    print(f"[WARN] Could not load {policy_path} as JIT ({jit_exc}) "
                          f"or as raw checkpoint ({raw_exc}); falling back to PD stand-still.")
                    policy = None

    # Resolve --task: always probe the loaded policy's input dim (51 → legonly,
    # 50 → squat) and either honor a matching explicit --task or hard-error
    # with a clear message.  Silent shape-mismatch crashes inside Isaac Sim's
    # update loop produce a clean "Simulation App Shutting Down" with no
    # traceback, which is impossible to debug from the user's seat.
    if policy is not None:
        detected_task: str | None = None
        for cand, dim in (("legonly", 51), ("squat", 50)):
            try:
                with torch.no_grad():
                    policy(torch.zeros(1, dim))
                detected_task = cand
                break
            except Exception:
                continue
        if detected_task is None:
            print("[ERROR] Could not detect policy task — neither 51-dim "
                  "(legonly) nor 50-dim (squat) forward pass succeeded.  "
                  "Is this a 13-action H1-2 policy?  Disabling policy.")
            policy = None
        elif args.task == "auto":
            policy_task = detected_task
        elif args.task == detected_task:
            policy_task = detected_task
        else:
            print(f"[ERROR] --task {args.task} requested, but the loaded "
                  f"policy expects the '{detected_task}' obs layout "
                  f"({'51' if detected_task == 'legonly' else '50'} dims).  "
                  f"Either drop --task (auto-detect picks the right one) or "
                  f"pass --policy <path-to-{args.task}-policy.pt>.  "
                  f"Disabling policy.")
            policy = None
        if policy is not None:
            print(f"[INFO] Policy task resolved to '{policy_task}' "
                  f"({'51-dim, velocity_commands' if policy_task == 'legonly' else '50-dim, target_height + pelvis_height'} obs).")

    gripper_view = None
    if gripper_paths:
        try:
            from isaacsim.robot.surface_gripper import GripperView
            gripper_view = GripperView(paths=gripper_paths)
            print(f"[INFO] GripperView wraps {len(gripper_paths)} gripper(s).")
        except Exception as exc:
            print(f"[WARN] GripperView init failed ({exc}); gripper commands will be skipped.")
            gripper_view = None

    velocity_cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    last_action = np.zeros(len(LOCOMOTION_JOINT_NAMES), dtype=np.float32)
    # EMA buffer for the smoothed PD target.  Initialized to the default pose
    # so the very first smoothed step doesn't pull joints toward zero.
    smoothed_target = locomotion_defaults.copy()

    grab_active = bool(args.grab)
    seg_idx = 0
    seg_start = time.time()
    current_gripper_cmd = -1.0
    if grab_active and policy is not None:
        print(f"[INFO] Running grab overlay ({len(GRAB_SEQUENCE)} segments) + policy "
              f"cmd = {velocity_cmd.tolist()}.")
    elif grab_active:
        print(f"[INFO] Running grab overlay ({len(GRAB_SEQUENCE)} segments); legs/torso held at default by PD.")
    elif policy is not None:
        if policy_task == "squat":
            print(f"[INFO] Running {len(LOCOMOTION_JOINT_NAMES)}-DOF squat policy with "
                  f"target_height = {args.target_height:.3f} m.")
        else:
            print(f"[INFO] Running {len(LOCOMOTION_JOINT_NAMES)}-DOF legonly policy with "
                  f"cmd = {velocity_cmd.tolist()}.")
    else:
        print(f"[INFO] PD stand-still — holding default pose.  Pass --grab or --policy <path> for motion.")

    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()

    try:
        while simulation_app.is_running():
            now = time.time()
            if args.duration > 0.0 and (now - t0) >= args.duration:
                break

            if policy is not None:
                pos_w, quat_w = art.get_world_pose()
                pos_np    = _to_numpy(pos_w)
                quat_np   = _to_numpy(quat_w)
                lin_vel_w = _to_numpy(art.get_linear_velocity())
                ang_vel_w = _to_numpy(art.get_angular_velocity())

                lin_vel_b = _quat_rotate_inverse(quat_np, lin_vel_w)
                ang_vel_b = _quat_rotate_inverse(quat_np, ang_vel_w)
                grav_b    = _quat_rotate_inverse(quat_np, np.array([0.0, 0.0, -1.0], dtype=np.float32))

                q_all  = _to_numpy(art.get_joint_positions())
                qd_all = _to_numpy(art.get_joint_velocities())
                q_subset  = q_all[locomotion_dof_indices].astype(np.float32)
                qd_subset = qd_all[locomotion_dof_indices].astype(np.float32)
                joint_pos_rel = q_subset - locomotion_defaults
                joint_vel_rel = qd_subset

                if policy_task == "squat":
                    # Squat obs: target_height + pelvis_height instead of vel cmd.
                    target_h = np.array([args.target_height], dtype=np.float32)
                    pelvis_h = np.array([pos_np[2]], dtype=np.float32)
                    middle = np.concatenate([target_h, pelvis_h], axis=0)
                else:
                    # Legonly obs: 3-D velocity command.
                    middle = velocity_cmd

                obs = np.concatenate([
                    lin_vel_b.astype(np.float32),
                    ang_vel_b.astype(np.float32),
                    grav_b.astype(np.float32),
                    middle,
                    joint_pos_rel,
                    joint_vel_rel,
                    last_action,
                ], axis=0)

                with torch.no_grad():
                    action_t = policy(torch.from_numpy(obs).unsqueeze(0))
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
                # Feed RAW policy output back into the next obs.  Smoothing
                # only the PD target leaves the policy's own internal feedback
                # loop (last_action -> obs -> next action) matching training.
                last_action = action.copy()

                raw_target = action * POLICY_ACTION_SCALE + locomotion_defaults
                if args.action_smoothing > 0.0:
                    alpha = float(args.action_smoothing)
                    smoothed_target[:] = alpha * smoothed_target + (1.0 - alpha) * raw_target
                    targets = smoothed_target
                else:
                    targets = raw_target
            else:
                # PD stand-still: hold the default pose; legs balance via gain.
                targets = locomotion_defaults.copy()

            arm_pose = None
            if grab_active:
                elapsed = now - seg_start
                dur, target_arms, gripper_cmd = GRAB_SEQUENCE[seg_idx]
                if elapsed >= dur:
                    seg_idx = (seg_idx + 1) % len(GRAB_SEQUENCE)
                    seg_start = now
                    elapsed = 0.0
                    dur, target_arms, gripper_cmd = GRAB_SEQUENCE[seg_idx]
                    print(f"[INFO] Grab segment {seg_idx}/{len(GRAB_SEQUENCE)-1} ({dur:.1f}s)")

                prev_arms = GRAB_SEQUENCE[(seg_idx - 1) % len(GRAB_SEQUENCE)][1]
                t_lerp = _smoothstep(elapsed / max(dur, 1e-3))
                arm_pose = _lerp_pose(prev_arms, target_arms, t_lerp)
                current_gripper_cmd = gripper_cmd

                # Legacy 21-DOF path: arms are inside the policy action subset,
                # so overwrite the relevant slots in `targets` and ride the
                # single apply_action below.
                if ARM_INDICES_IN_LOCOMOTION:
                    for arm_idx, name in zip(ARM_INDICES_IN_LOCOMOTION, ARM_JOINT_NAMES):
                        targets[arm_idx] = arm_pose[name]

            art.apply_action(ArticulationAction(
                joint_positions=targets,
                joint_indices=locomotion_dof_indices,
            ))

            # Legonly path: arms aren't in the policy subset, so drive them via
            # a separate ArticulationAction routed at the arm DOF indices.
            if (arm_pose is not None
                    and not ARM_INDICES_IN_LOCOMOTION
                    and arm_dof_indices_np is not None):
                arm_targets = np.array(
                    [arm_pose[n] for n in arm_present_names], dtype=np.float32
                )
                art.apply_action(ArticulationAction(
                    joint_positions=arm_targets,
                    joint_indices=arm_dof_indices_np,
                ))

            if gripper_view is not None:
                try:
                    gripper_view.apply_gripper_action(
                        [float(current_gripper_cmd)] * gripper_view.count
                    )
                except Exception as exc:
                    print(f"[WARN] gripper.apply_gripper_action failed ({exc}); disabling.")
                    gripper_view = None

            # Advance physics DECIMATION times per policy query — matches the
            # env_cfg's `physics_dt / decimation` contract.  Render once per
            # outer iter to keep the viewport fluid.
            for _ in range(DECIMATION):
                sim.step(render=False)
            sim.render()

            # Per-second diagnostic: pelvis z + horizontal speed estimate.
            # If z drops fast → falling through floor / friction too low.
            # If z stays near spawn but horizontal speed << cmd_vx → policy
            # is producing motion but feet are slipping → friction issue.
            if int(now - t0) != int(now - t0 - dt):   # every wall second
                try:
                    pos_dbg, _ = art.get_world_pose()
                    lv_dbg = _to_numpy(art.get_linear_velocity())
                    pz = float(_to_numpy(pos_dbg)[2])
                    vxy = float(np.hypot(lv_dbg[0], lv_dbg[1]))
                    print(f"[DBG] t={now - t0:5.1f}s  pelvis_z={pz:+.3f}  "
                          f"|v_xy|={vxy:+.3f} m/s  cmd_vx={args.cmd_vx:+.2f}")
                except Exception:
                    pass

            # Real-time pace.  sim.step() does NOT block on wall clock, so we
            # cap the outer loop at 1/POLICY_HZ.  --rate lets you slow it down
            # (rate < 50) but never go faster than POLICY_HZ since that would
            # drift away from the training contract.
            target_dt = max(dt, 1.0 / POLICY_HZ)
            sleep_time = target_dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[INFO] Stopped.")
    finally:
        try:
            sim.stop()
        except Exception:
            pass
        usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
