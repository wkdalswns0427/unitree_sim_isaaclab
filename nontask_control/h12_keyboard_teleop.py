#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Keyboard teleop for H1-2 in joint DDS tasks.

Target sim command:
  python3 sim_main.py --device cuda --enable_cameras --action_source dds \
    --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint \
    --enable_inspire_dds --robot_type h1_2

Run this in another terminal:
  python3 nontask_control/h12_keyboard_teleop.py --channel 1
"""

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.utils.crc import CRC


# H1-2 slots expected by action_provider_dds.py (robot_type=h1_2)
L_HIP_PITCH = 0
L_HIP_ROLL = 1
L_HIP_YAW = 2
L_KNEE = 3
L_ANKLE_PITCH = 4
L_ANKLE_ROLL = 5
R_HIP_PITCH = 6
R_HIP_ROLL = 7
R_HIP_YAW = 8
R_KNEE = 9
R_ANKLE_PITCH = 10
R_ANKLE_ROLL = 11
TORSO = 12
L_SHOULDER_PITCH = 13
L_SHOULDER_ROLL = 14
L_SHOULDER_YAW = 15
L_ELBOW = 16
L_WRIST_ROLL = 17
L_WRIST_PITCH = 18
L_WRIST_YAW = 19
R_SHOULDER_PITCH = 20
R_SHOULDER_ROLL = 21
R_SHOULDER_YAW = 22
R_ELBOW = 23
R_WRIST_ROLL = 24
R_WRIST_PITCH = 25
R_WRIST_YAW = 26

LEG_INDICES = [
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_ANKLE_PITCH,
    L_ANKLE_ROLL,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
    R_ANKLE_PITCH,
    R_ANKLE_ROLL,
    TORSO,
]
ARM_INDICES = [
    L_SHOULDER_PITCH,
    L_SHOULDER_ROLL,
    L_SHOULDER_YAW,
    L_ELBOW,
    L_WRIST_ROLL,
    L_WRIST_PITCH,
    L_WRIST_YAW,
    R_SHOULDER_PITCH,
    R_SHOULDER_ROLL,
    R_SHOULDER_YAW,
    R_ELBOW,
    R_WRIST_ROLL,
    R_WRIST_PITCH,
    R_WRIST_YAW,
]


@dataclass
class JointLimit:
    lo: float
    hi: float


LIMITS = {
    L_HIP_PITCH: JointLimit(-1.0, 1.0),
    R_HIP_PITCH: JointLimit(-1.0, 1.0),
    L_KNEE: JointLimit(-1.6, 1.6),
    R_KNEE: JointLimit(-1.6, 1.6),
    L_ANKLE_PITCH: JointLimit(-1.0, 1.0),
    R_ANKLE_PITCH: JointLimit(-1.0, 1.0),
    TORSO: JointLimit(-0.8, 0.8),
    L_SHOULDER_PITCH: JointLimit(-2.0, 2.0),
    R_SHOULDER_PITCH: JointLimit(-2.0, 2.0),
    L_ELBOW: JointLimit(-2.4, 2.4),
    R_ELBOW: JointLimit(-2.4, 2.4),
}

STARTUP_STAND_Q = {
    L_HIP_PITCH: 0.28,
    R_HIP_PITCH: -0.28,
    L_KNEE: -0.62,
    R_KNEE: 0.62,
    L_ANKLE_PITCH: -0.32,
    R_ANKLE_PITCH: 0.32,
    TORSO: 0.0,
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class RawStdin:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key_nonblocking() -> str:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H1-2 keyboard teleop over DDS")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--interface", type=str, default="", help="Optional network interface/IP")
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument("--step", type=float, default=0.05, help="Joint increment per keypress (rad)")
    parser.add_argument("--hand_step", type=float, default=0.05, help="Inspire increment per keypress [0..1]")
    parser.add_argument("--kp_body", type=float, default=120.0, help="PD kp for leg/torso")
    parser.add_argument("--kd_body", type=float, default=5.0, help="PD kd for leg/torso")
    parser.add_argument("--kp_arm", type=float, default=60.0, help="PD kp for arm")
    parser.add_argument("--kd_arm", type=float, default=2.0, help="PD kd for arm")
    return parser


def print_help() -> None:
    print("")
    print("Key bindings:")
    print("  q: quit")
    print("  h: print help")
    print("  space: reset to standing pose")
    print("  i/k: squat down/up (coupled leg motion)")
    print("  j/l: torso -/+")
    print("  w/s: left shoulder pitch +/-, e/d: right shoulder pitch +/-")
    print("  r/f: left elbow +/-, t/g: right elbow +/-")
    print("  z/x: close/open both Inspire hands")
    print("  u: reset object (DDS reset category=1)")
    print("  p: reset all (DDS reset category=2)")
    print("")


def make_hand_msg(hand_value: float) -> MotorCmds_:
    msg = MotorCmds_()
    for _ in range(12):
        cmd = unitree_go_msg_dds__MotorCmd_()
        cmd.q = hand_value
        cmd.dq = 0.0
        cmd.tau = 0.0
        cmd.kp = 1.0
        cmd.kd = 0.1
        msg.cmds.append(cmd)
    return msg


def apply_squat_step(q, step):
    # mirrored signs for H1-2 lowcmd mapping
    q[L_HIP_PITCH] = clamp(q[L_HIP_PITCH] + step, -1.0, 1.0)
    q[R_HIP_PITCH] = clamp(q[R_HIP_PITCH] - step, -1.0, 1.0)
    q[L_KNEE] = clamp(q[L_KNEE] - 1.8 * step, -1.6, 1.6)
    q[R_KNEE] = clamp(q[R_KNEE] + 1.8 * step, -1.6, 1.6)
    q[L_ANKLE_PITCH] = clamp(q[L_ANKLE_PITCH] - 0.9 * step, -1.0, 1.0)
    q[R_ANKLE_PITCH] = clamp(q[R_ANKLE_PITCH] + 0.9 * step, -1.0, 1.0)


def publish_reset(reset_pub, category: int) -> None:
    msg = String_(data=str(category))
    reset_pub.Write(msg)
    print(f"\n[teleop] sent reset category={category}")


def main() -> None:
    args = build_parser().parse_args()
    if args.interface:
        ChannelFactoryInitialize(args.channel, args.interface)
    else:
        ChannelFactoryInitialize(args.channel)

    low_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_pub.Init()
    hand_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    hand_pub.Init()
    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()

    crc = CRC()
    low = unitree_hg_msg_dds__LowCmd_()
    low.mode_pr = 0
    low.mode_machine = 0

    q = [0.0] * 29
    for idx, val in STARTUP_STAND_Q.items():
        q[idx] = val
    hand_value = 0.7

    for i in range(29):
        m = low.motor_cmd[i]
        m.mode = 1
        m.q = 0.0
        m.dq = 0.0
        m.tau = 0.0
        m.kp = 0.0
        m.kd = 0.0
    for idx in LEG_INDICES:
        low.motor_cmd[idx].kp = args.kp_body
        low.motor_cmd[idx].kd = args.kd_body
    for idx in ARM_INDICES:
        low.motor_cmd[idx].kp = args.kp_arm
        low.motor_cmd[idx].kd = args.kd_arm

    print_help()
    print("Keyboard teleop started. Focus this terminal to send keys.")
    print(f"DDS channel={args.channel}, rate={args.rate:.1f} Hz")

    dt = 1.0 / max(args.rate, 1e-3)
    last_print = 0.0
    stand_hold_until = 0.0

    try:
        with RawStdin():
            while True:
                key = read_key_nonblocking()
                if key:
                    if key == "q":
                        break
                    if key == "h":
                        print_help()
                    elif key == " ":
                        for i in range(29):
                            q[i] = 0.0
                        for idx, val in STARTUP_STAND_Q.items():
                            q[idx] = val
                    elif key == "i":
                        apply_squat_step(q, +args.step)
                    elif key == "k":
                        apply_squat_step(q, -args.step)
                    elif key == "j":
                        q[TORSO] = clamp(q[TORSO] - args.step, LIMITS[TORSO].lo, LIMITS[TORSO].hi)
                    elif key == "l":
                        q[TORSO] = clamp(q[TORSO] + args.step, LIMITS[TORSO].lo, LIMITS[TORSO].hi)
                    elif key == "w":
                        q[L_SHOULDER_PITCH] = clamp(q[L_SHOULDER_PITCH] + args.step, LIMITS[L_SHOULDER_PITCH].lo, LIMITS[L_SHOULDER_PITCH].hi)
                    elif key == "s":
                        q[L_SHOULDER_PITCH] = clamp(q[L_SHOULDER_PITCH] - args.step, LIMITS[L_SHOULDER_PITCH].lo, LIMITS[L_SHOULDER_PITCH].hi)
                    elif key == "e":
                        q[R_SHOULDER_PITCH] = clamp(q[R_SHOULDER_PITCH] + args.step, LIMITS[R_SHOULDER_PITCH].lo, LIMITS[R_SHOULDER_PITCH].hi)
                    elif key == "d":
                        q[R_SHOULDER_PITCH] = clamp(q[R_SHOULDER_PITCH] - args.step, LIMITS[R_SHOULDER_PITCH].lo, LIMITS[R_SHOULDER_PITCH].hi)
                    elif key == "r":
                        q[L_ELBOW] = clamp(q[L_ELBOW] + args.step, LIMITS[L_ELBOW].lo, LIMITS[L_ELBOW].hi)
                    elif key == "f":
                        q[L_ELBOW] = clamp(q[L_ELBOW] - args.step, LIMITS[L_ELBOW].lo, LIMITS[L_ELBOW].hi)
                    elif key == "t":
                        q[R_ELBOW] = clamp(q[R_ELBOW] + args.step, LIMITS[R_ELBOW].lo, LIMITS[R_ELBOW].hi)
                    elif key == "g":
                        q[R_ELBOW] = clamp(q[R_ELBOW] - args.step, LIMITS[R_ELBOW].lo, LIMITS[R_ELBOW].hi)
                    elif key == "z":
                        hand_value = clamp(hand_value - args.hand_step, 0.0, 1.0)
                    elif key == "x":
                        hand_value = clamp(hand_value + args.hand_step, 0.0, 1.0)
                    elif key == "u":
                        publish_reset(reset_pub, 1)
                        for i in range(29):
                            q[i] = 0.0
                        for idx, val in STARTUP_STAND_Q.items():
                            q[idx] = val
                        stand_hold_until = time.time() + 1.0
                    elif key == "p":
                        publish_reset(reset_pub, 2)
                        for i in range(29):
                            q[i] = 0.0
                        for idx, val in STARTUP_STAND_Q.items():
                            q[idx] = val
                        stand_hold_until = time.time() + 1.0

                if time.time() < stand_hold_until:
                    for i in range(29):
                        q[i] = 0.0
                    for idx, val in STARTUP_STAND_Q.items():
                        q[idx] = val

                for i in range(29):
                    low.motor_cmd[i].q = q[i]
                low.crc = crc.Crc(low)
                low_pub.Write(low)
                hand_pub.Write(make_hand_msg(hand_value))

                now = time.time()
                if now - last_print > 0.5:
                    last_print = now
                    print(
                        f"\rhip(L/R)=({q[L_HIP_PITCH]:+.2f},{q[R_HIP_PITCH]:+.2f}) "
                        f"knee(L/R)=({q[L_KNEE]:+.2f},{q[R_KNEE]:+.2f}) "
                        f"torso={q[TORSO]:+.2f} hand={hand_value:.2f}      ",
                        end="",
                        flush=True,
                    )
                time.sleep(dt)
    except KeyboardInterrupt:
        pass

    # soft neutral tail
    for _ in range(20):
        for i in range(29):
            low.motor_cmd[i].q = 0.0
        low.crc = crc.Crc(low)
        low_pub.Write(low)
        hand_pub.Write(make_hand_msg(hand_value))
        time.sleep(dt)
    print("\nStopped.")


if __name__ == "__main__":
    main()
