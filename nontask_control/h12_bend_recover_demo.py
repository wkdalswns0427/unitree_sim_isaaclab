#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Small H1-2 full-body motion demo:
- stand
- squat down
- stand back up

Publishes DDS low-level commands on `rt/lowcmd`.
"""

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC


# H1-2 command slots consumed by action_provider/action_provider_dds.py
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
R_SHOULDER_PITCH = 20


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
ARM_TEST_INDICES = [L_SHOULDER_PITCH, R_SHOULDER_PITCH]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H1-2 squat-and-stand DDS motion demo.")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id (must match sim)")
    parser.add_argument("--interface", type=str, default="", help="Optional network interface/IP")
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate in Hz")
    parser.add_argument("--cycles", type=int, default=0, help="Number of squat/stand cycles (0 = forever)")
    parser.add_argument("--duration", type=float, default=0.0, help="Max run duration in seconds (0 = ignore)")

    parser.add_argument("--stand_before", type=float, default=1.0, help="Initial stand time per cycle (s)")
    parser.add_argument("--squat_time", type=float, default=1.2, help="Time to squat down (s)")
    parser.add_argument("--hold_squat", type=float, default=0.8, help="Hold time at squat pose (s)")
    parser.add_argument("--rise_time", type=float, default=1.2, help="Time to stand back up (s)")
    parser.add_argument("--hold_stand", type=float, default=1.0, help="Hold time after standing (s)")

    parser.add_argument("--sign", type=float, default=1.0, help="Global sign for squat direction (try -1.0 if needed)")
    parser.add_argument("--hip_pitch", type=float, default=0.26, help="Hip pitch command at full squat (rad)")
    parser.add_argument("--knee", type=float, default=0.70, help="Knee command at full squat (rad)")
    parser.add_argument("--ankle_pitch", type=float, default=-0.32, help="Ankle pitch command at full squat (rad)")
    parser.add_argument("--torso", type=float, default=0.0, help="Torso command at full squat (rad)")
    parser.add_argument("--disable_torso", action="store_true", help="Do not command torso slot (use legs only)")

    parser.add_argument("--kp_body", type=float, default=110.0, help="Position gain for body joints")
    parser.add_argument("--kd_body", type=float, default=4.0, help="Velocity gain for body joints")
    parser.add_argument("--arm_test_amp", type=float, default=0.0, help="Shoulder pitch test amplitude (rad); set 0 to disable")
    parser.add_argument("--arm_test_freq", type=float, default=0.35, help="Shoulder pitch test frequency (Hz)")
    parser.add_argument("--mirror_legs", action="store_true", default=True, help="Use mirrored left/right signs (recommended for H1-2)")
    return parser


def compute_squat_alpha(t_cycle: float, stand_before: float, squat_time: float, hold_squat: float, rise_time: float) -> float:
    t0 = stand_before
    t1 = t0 + squat_time
    t2 = t1 + hold_squat
    t3 = t2 + rise_time

    if t_cycle < t0:
        return 0.0
    if t_cycle < t1:
        return (t_cycle - t0) / max(squat_time, 1e-6)
    if t_cycle < t2:
        return 1.0
    if t_cycle < t3:
        return 1.0 - (t_cycle - t2) / max(rise_time, 1e-6)
    return 0.0


def main() -> None:
    args = build_parser().parse_args()
    if args.interface:
        ChannelFactoryInitialize(args.channel, args.interface)
    else:
        ChannelFactoryInitialize(args.channel)

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()

    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = 0
    msg.mode_machine = 0

    for i in range(29):
        m = msg.motor_cmd[i]
        m.mode = 1
        m.q = 0.0
        m.dq = 0.0
        m.tau = 0.0
        m.kp = 0.0
        m.kd = 0.0

    for idx in BODY_INDICES:
        msg.motor_cmd[idx].kp = args.kp_body
        msg.motor_cmd[idx].kd = args.kd_body
    for idx in ARM_TEST_INDICES:
        msg.motor_cmd[idx].kp = 50.0
        msg.motor_cmd[idx].kd = 2.0

    dt = 1.0 / max(args.rate, 1e-3)
    cycle_period = args.stand_before + args.squat_time + args.hold_squat + args.rise_time + args.hold_stand
    t_start = time.time()
    cycle_count = 0

    print("Publishing H1-2 squat/stand DDS commands. Press Ctrl+C to stop.")
    print(f"channel={args.channel}, rate={args.rate:.1f}Hz, cycle_period={cycle_period:.2f}s")

    try:
        while True:
            now = time.time()
            elapsed = now - t_start
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            t_cycle = elapsed % cycle_period
            alpha = compute_squat_alpha(
                t_cycle,
                args.stand_before,
                args.squat_time,
                args.hold_squat,
                args.rise_time,
            )

            if t_cycle < dt:
                cycle_count += 1
                if args.cycles > 0 and cycle_count > args.cycles:
                    break

            s = args.sign
            hip = s * args.hip_pitch * alpha
            knee = s * args.knee * alpha
            ankle = s * args.ankle_pitch * alpha
            torso = s * args.torso * alpha

            # Reset all body slots each cycle then apply squat-only sagittal commands.
            for idx in BODY_INDICES:
                msg.motor_cmd[idx].q = 0.0

            if args.mirror_legs:
                # H1-2 uses mirrored joint axes between left/right legs.
                msg.motor_cmd[L_HIP_PITCH].q = hip
                msg.motor_cmd[R_HIP_PITCH].q = -hip
                msg.motor_cmd[L_KNEE].q = -knee
                msg.motor_cmd[R_KNEE].q = knee
                msg.motor_cmd[L_ANKLE_PITCH].q = ankle
                msg.motor_cmd[R_ANKLE_PITCH].q = -ankle
            else:
                msg.motor_cmd[L_HIP_PITCH].q = hip
                msg.motor_cmd[R_HIP_PITCH].q = hip
                msg.motor_cmd[L_KNEE].q = knee
                msg.motor_cmd[R_KNEE].q = knee
                msg.motor_cmd[L_ANKLE_PITCH].q = ankle
                msg.motor_cmd[R_ANKLE_PITCH].q = ankle
            if not args.disable_torso:
                msg.motor_cmd[TORSO].q = torso
            if args.arm_test_amp > 0.0:
                arm = args.arm_test_amp * math.sin(2.0 * math.pi * args.arm_test_freq * elapsed)
                msg.motor_cmd[L_SHOULDER_PITCH].q = arm
                msg.motor_cmd[R_SHOULDER_PITCH].q = -arm

            msg.crc = crc.Crc(msg)
            pub.Write(msg)

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass

    # Send a short neutral tail so robot returns to stand.
    for _ in range(20):
        msg.motor_cmd[L_HIP_PITCH].q = 0.0
        msg.motor_cmd[R_HIP_PITCH].q = 0.0
        msg.motor_cmd[L_KNEE].q = 0.0
        msg.motor_cmd[R_KNEE].q = 0.0
        msg.motor_cmd[L_ANKLE_PITCH].q = 0.0
        msg.motor_cmd[R_ANKLE_PITCH].q = 0.0
        if not args.disable_torso:
            msg.motor_cmd[TORSO].q = 0.0
        msg.motor_cmd[L_SHOULDER_PITCH].q = 0.0
        msg.motor_cmd[R_SHOULDER_PITCH].q = 0.0
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        time.sleep(dt)

    print("Stopped.")


if __name__ == "__main__":
    main()
