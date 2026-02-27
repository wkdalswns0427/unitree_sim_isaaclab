#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Publish a standing H1 torso oscillation for the IsaacLab H1 minimal asset."""

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC


L_HIP_PITCH = 0
L_HIP_ROLL = 1
L_HIP_YAW = 2
L_KNEE = 3
L_ANKLE = 4
R_HIP_PITCH = 6
R_HIP_ROLL = 7
R_HIP_YAW = 8
R_KNEE = 9
R_ANKLE = 10
TORSO = 12

BODY_INDICES = [
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_ANKLE,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
    R_ANKLE,
    TORSO,
]

STARTUP_STAND_Q = {
    L_HIP_PITCH: -0.28,
    R_HIP_PITCH: -0.28,
    L_KNEE: 0.79,
    R_KNEE: 0.79,
    L_ANKLE: -0.52,
    R_ANKLE: -0.52,
    TORSO: 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H1 minimal torso rotation DDS demo")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--interface", type=str, default="", help="Optional network interface/IP")
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument("--amp", type=float, default=0.15, help="Torso amplitude (rad)")
    parser.add_argument("--freq", type=float, default=0.35, help="Torso frequency (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = forever)")
    parser.add_argument("--kp_body", type=float, default=120.0, help="PD kp for leg/torso joints")
    parser.add_argument("--kd_body", type=float, default=5.0, help="PD kd for leg/torso joints")
    return parser


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

    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()

    print("Publishing H1 minimal torso oscillation on rt/lowcmd. Press Ctrl+C to stop.")
    print(f"channel={args.channel}, rate={args.rate:.1f}Hz, amp={args.amp:.3f}rad, freq={args.freq:.3f}Hz")

    try:
        while True:
            now = time.time()
            elapsed = now - t0
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            for idx in BODY_INDICES:
                msg.motor_cmd[idx].q = 0.0
            for idx, q in STARTUP_STAND_Q.items():
                msg.motor_cmd[idx].q = q

            msg.motor_cmd[TORSO].q = args.amp * math.sin(2.0 * math.pi * args.freq * elapsed)

            msg.crc = crc.Crc(msg)
            pub.Write(msg)

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass

    for _ in range(20):
        for idx in BODY_INDICES:
            msg.motor_cmd[idx].q = 0.0
        for idx, q in STARTUP_STAND_Q.items():
            msg.motor_cmd[idx].q = q
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        time.sleep(dt)

    print("Stopped.")


if __name__ == "__main__":
    main()
