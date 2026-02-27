#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
H1-2 squat demo over DDS lowcmd.

Recommended sim command (joint task):
  python3 sim_main.py --device cuda --enable_cameras --action_source dds \
    --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint \
    --enable_inspire_dds --robot_type h1_2

Then run:
  python3 nontask_control/h12_squat_demo.py --channel 1

Optional wholebody task command (requires lower-body DDS override):
  python3 sim_main.py --device cuda --enable_cameras --action_source dds \
    --task Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody \
    --enable_inspire_dds --robot_type h1_2 --wholebody_dds_lower_body
"""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC


# H1-2 lowcmd slots used by unitree_sim_isaaclab
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
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H1-2 squat/get-up DDS demo")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--interface", type=str, default="", help="Optional network interface/IP")
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument("--cycles", type=int, default=0, help="Squat cycles (0 = forever)")
    parser.add_argument("--duration", type=float, default=0.0, help="Max runtime seconds (0 = ignore)")

    parser.add_argument("--stand_before", type=float, default=1.2, help="Initial stand time (s)")
    parser.add_argument("--squat_time", type=float, default=1.2, help="Squat-down duration (s)")
    parser.add_argument("--hold_squat", type=float, default=1.0, help="Hold at squat (s)")
    parser.add_argument("--rise_time", type=float, default=1.2, help="Stand-up duration (s)")
    parser.add_argument("--hold_stand", type=float, default=1.0, help="Hold at stand (s)")

    parser.add_argument("--hip_pitch", type=float, default=0.45, help="Hip target at full squat (rad)")
    parser.add_argument("--knee", type=float, default=1.05, help="Knee target at full squat (rad)")
    parser.add_argument("--ankle_pitch", type=float, default=-0.52, help="Ankle target at full squat (rad)")
    parser.add_argument("--sign", type=float, default=1.0, help="Global sign (try -1.0 if reversed)")
    parser.add_argument("--mirror_legs", action=argparse.BooleanOptionalAction, default=True, help="Mirror left/right leg signs")

    parser.add_argument("--kp", type=float, default=150.0, help="PD stiffness for leg slots")
    parser.add_argument("--kd", type=float, default=6.0, help="PD damping for leg slots")
    return parser


def compute_alpha(t_cycle: float, stand_before: float, squat_time: float, hold_squat: float, rise_time: float) -> float:
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

    for idx in LEG_INDICES:
        msg.motor_cmd[idx].kp = args.kp
        msg.motor_cmd[idx].kd = args.kd

    dt = 1.0 / max(args.rate, 1e-3)
    cycle_period = args.stand_before + args.squat_time + args.hold_squat + args.rise_time + args.hold_stand
    t_start = time.time()
    cycle_count = 0

    print("Publishing H1-2 squat/get-up lowcmd. Ctrl+C to stop.")
    print(f"channel={args.channel}, rate={args.rate:.1f}Hz, cycle_period={cycle_period:.2f}s")

    try:
        while True:
            now = time.time()
            elapsed = now - t_start
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            t_cycle = elapsed % cycle_period
            alpha = compute_alpha(t_cycle, args.stand_before, args.squat_time, args.hold_squat, args.rise_time)
            if t_cycle < dt:
                cycle_count += 1
                if args.cycles > 0 and cycle_count > args.cycles:
                    break

            for idx in LEG_INDICES:
                msg.motor_cmd[idx].q = 0.0

            hip = args.sign * args.hip_pitch * alpha
            knee = args.sign * args.knee * alpha
            ankle = args.sign * args.ankle_pitch * alpha

            if args.mirror_legs:
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

            msg.crc = crc.Crc(msg)
            pub.Write(msg)

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass

    for _ in range(20):
        for idx in LEG_INDICES:
            msg.motor_cmd[idx].q = 0.0
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        time.sleep(dt)

    print("Stopped.")


if __name__ == "__main__":
    main()
