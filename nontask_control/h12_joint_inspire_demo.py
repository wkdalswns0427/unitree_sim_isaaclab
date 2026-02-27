#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Publish simple joint commands for H1-2 Inspire joint tasks.

This script is intended for non-wholebody tasks such as:
  Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint

It sends:
  - rt/lowcmd      (29 motor command slots; arm indices 13..26 are used by sim bridge)
  - rt/inspire/cmd (12 hand commands)
"""

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC


# Indices expected by action_provider/action_provider_dds.py for robot_type=h1_2.
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

BODY_INDICES = [
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint DDS demo sender for H1-2 Inspire tasks (non-wholebody)."
    )
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id (must match sim)")
    parser.add_argument("--interface", type=str, default="", help="Optional network interface/IP")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate in Hz")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = forever)")
    parser.add_argument("--arm_amp", type=float, default=0.35, help="Arm sinusoid amplitude (rad)")
    parser.add_argument("--arm_freq", type=float, default=0.5, help="Arm sinusoid frequency (Hz)")
    parser.add_argument("--elbow_bias", type=float, default=0.6, help="Nominal elbow bend (rad)")
    parser.add_argument("--full_body", action="store_true", help="Enable lower-body + torso motion")
    parser.add_argument("--leg_amp", type=float, default=0.08, help="Leg/torso sinusoid amplitude (rad)")
    parser.add_argument("--hand_open", type=float, default=0.85, help="Inspire normalized open value [0, 1]")
    parser.add_argument("--hand_close", type=float, default=0.25, help="Inspire normalized close value [0, 1]")
    parser.add_argument("--hand_freq", type=float, default=0.35, help="Hand open/close frequency (Hz)")
    return parser


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
    crc = CRC()

    low = unitree_hg_msg_dds__LowCmd_()
    low.mode_pr = 0
    low.mode_machine = 0

    # Initialize all 29 slots to zero.
    for i in range(29):
        m = low.motor_cmd[i]
        m.mode = 1
        m.q = 0.0
        m.dq = 0.0
        m.tau = 0.0
        m.kp = 0.0
        m.kd = 0.0

    # Give gains only on arm-related slots so we do not force lower-body joints.
    for idx in ARM_INDICES:
        low.motor_cmd[idx].kp = 40.0
        low.motor_cmd[idx].kd = 1.5
    if args.full_body:
        for idx in BODY_INDICES:
            low.motor_cmd[idx].kp = 60.0
            low.motor_cmd[idx].kd = 2.0

    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()

    hand_open = clamp(args.hand_open, 0.0, 1.0)
    hand_close = clamp(args.hand_close, 0.0, 1.0)
    hand_mid = 0.5 * (hand_open + hand_close)
    hand_amp = 0.5 * (hand_open - hand_close)

    print("Publishing rt/lowcmd + rt/inspire/cmd. Press Ctrl+C to stop.")
    print(f"channel={args.channel}, rate={args.rate:.1f}Hz")

    try:
        while True:
            now = time.time()
            t = now - t0
            if args.duration > 0.0 and t >= args.duration:
                break

            arm_w = 2.0 * math.pi * args.arm_freq * t
            hand_w = 2.0 * math.pi * args.hand_freq * t

            if args.full_body:
                leg_s = math.sin(0.8 * arm_w)
                leg_c = math.cos(0.8 * arm_w)

                low.motor_cmd[L_HIP_PITCH].q = 0.5 * args.leg_amp * leg_s
                low.motor_cmd[R_HIP_PITCH].q = -0.5 * args.leg_amp * leg_s
                low.motor_cmd[L_HIP_ROLL].q = 0.25 * args.leg_amp * leg_c
                low.motor_cmd[R_HIP_ROLL].q = -0.25 * args.leg_amp * leg_c
                low.motor_cmd[L_HIP_YAW].q = 0.2 * args.leg_amp * leg_s
                low.motor_cmd[R_HIP_YAW].q = -0.2 * args.leg_amp * leg_s

                low.motor_cmd[L_KNEE].q = -0.35 * args.leg_amp * leg_s
                low.motor_cmd[R_KNEE].q = 0.35 * args.leg_amp * leg_s
                low.motor_cmd[L_ANKLE_PITCH].q = -0.25 * args.leg_amp * leg_s
                low.motor_cmd[R_ANKLE_PITCH].q = 0.25 * args.leg_amp * leg_s
                low.motor_cmd[L_ANKLE_ROLL].q = 0.2 * args.leg_amp * leg_c
                low.motor_cmd[R_ANKLE_ROLL].q = -0.2 * args.leg_amp * leg_c
                low.motor_cmd[TORSO].q = 0.25 * args.leg_amp * leg_s

            # A simple mirrored arm motion.
            low.motor_cmd[L_SHOULDER_PITCH].q = args.arm_amp * math.sin(arm_w)
            low.motor_cmd[R_SHOULDER_PITCH].q = -args.arm_amp * math.sin(arm_w)

            low.motor_cmd[L_SHOULDER_ROLL].q = 0.5 * args.arm_amp * math.sin(arm_w + 0.5 * math.pi)
            low.motor_cmd[R_SHOULDER_ROLL].q = -0.5 * args.arm_amp * math.sin(arm_w + 0.5 * math.pi)

            low.motor_cmd[L_SHOULDER_YAW].q = 0.4 * args.arm_amp * math.sin(arm_w + 0.25 * math.pi)
            low.motor_cmd[R_SHOULDER_YAW].q = -0.4 * args.arm_amp * math.sin(arm_w + 0.25 * math.pi)

            low.motor_cmd[L_ELBOW].q = args.elbow_bias + 0.35 * args.arm_amp * math.sin(arm_w)
            low.motor_cmd[R_ELBOW].q = -args.elbow_bias - 0.35 * args.arm_amp * math.sin(arm_w)

            low.motor_cmd[L_WRIST_ROLL].q = 0.4 * args.arm_amp * math.sin(arm_w + 1.2)
            low.motor_cmd[R_WRIST_ROLL].q = -0.4 * args.arm_amp * math.sin(arm_w + 1.2)

            low.motor_cmd[L_WRIST_PITCH].q = 0.25 * args.arm_amp * math.sin(arm_w + 0.8)
            low.motor_cmd[R_WRIST_PITCH].q = -0.25 * args.arm_amp * math.sin(arm_w + 0.8)

            low.motor_cmd[L_WRIST_YAW].q = 0.2 * args.arm_amp * math.sin(arm_w + 0.3)
            low.motor_cmd[R_WRIST_YAW].q = -0.2 * args.arm_amp * math.sin(arm_w + 0.3)

            low.crc = crc.Crc(low)
            low_pub.Write(low)

            hand_val = clamp(hand_mid + hand_amp * math.sin(hand_w), 0.0, 1.0)
            hand_msg = MotorCmds_()
            for _ in range(12):
                c = unitree_go_msg_dds__MotorCmd_()
                c.q = hand_val
                c.dq = 0.0
                c.tau = 0.0
                c.kp = 1.0
                c.kd = 0.1
                hand_msg.cmds.append(c)
            hand_pub.Write(hand_msg)

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass

    print("Stopped.")


if __name__ == "__main__":
    main()
