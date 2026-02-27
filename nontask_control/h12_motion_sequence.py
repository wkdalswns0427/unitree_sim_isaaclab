#!/usr/bin/env python3
"""
Open-loop high-level motion sequencer for H1-2 wholebody tasks.

This does not generate joint commands directly. It publishes the same
`[x_vel, y_vel, yaw_vel, body_height]` run-command vector consumed by the
existing wholebody policy in `action_provider/action_provider_wh_dds.py`.

Example:
  python3 sim_main.py \
    --device cuda \
    --enable_cameras \
    --task Isaac-Warehouse-Walk-H12-27dof-Inspire-Wholebody \
    --enable_inspire_dds \
    --robot_type h1_2 \
    --model_path logs/rsl_rl/h12_move_cylinder_wholebody/<run>/exported/policy.onnx

  python3 nontask_control/h12_motion_sequence.py --preset square --loop
"""

import argparse
import json
import time
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


@dataclass
class Segment:
    duration: float
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_vel: float = 0.0
    height: float = 0.8
    label: str = ""


def publish_string(pub: ChannelPublisher, value: str):
    pub.Write(String_(data=value))


def _preset_square(base_height: float) -> list[Segment]:
    return [
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "settle"),
        Segment(3.0, 0.45, 0.0, 0.0, base_height, "forward-1"),
        Segment(1.6, 0.0, 0.0, 0.65, base_height, "turn-1"),
        Segment(3.0, 0.45, 0.0, 0.0, base_height, "forward-2"),
        Segment(1.6, 0.0, 0.0, 0.65, base_height, "turn-2"),
        Segment(3.0, 0.45, 0.0, 0.0, base_height, "forward-3"),
        Segment(1.6, 0.0, 0.0, 0.65, base_height, "turn-3"),
        Segment(3.0, 0.45, 0.0, 0.0, base_height, "forward-4"),
        Segment(1.6, 0.0, 0.0, 0.65, base_height, "turn-4"),
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "stop"),
    ]


def _preset_figure8(base_height: float) -> list[Segment]:
    return [
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "settle"),
        Segment(7.0, 0.45, 0.0, 0.45, base_height, "left-loop"),
        Segment(7.0, 0.45, 0.0, -0.45, base_height, "right-loop"),
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "stop"),
    ]


def _preset_strafe_scan(base_height: float) -> list[Segment]:
    return [
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "settle"),
        Segment(3.0, 0.2, 0.25, 0.0, base_height, "diag-left"),
        Segment(2.0, 0.0, 0.0, -0.5, base_height, "scan-right"),
        Segment(3.0, 0.2, -0.25, 0.0, base_height, "diag-right"),
        Segment(2.0, 0.0, 0.0, 0.5, base_height, "scan-left"),
        Segment(1.0, 0.0, 0.0, 0.0, base_height, "stop"),
    ]


def load_segments(args) -> list[Segment]:
    if args.sequence_json:
        with open(args.sequence_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        segments = []
        for idx, item in enumerate(raw):
            segments.append(
                Segment(
                    duration=float(item["duration"]),
                    x_vel=float(item.get("x_vel", 0.0)),
                    y_vel=float(item.get("y_vel", 0.0)),
                    yaw_vel=float(item.get("yaw_vel", 0.0)),
                    height=float(item.get("height", args.base_height)),
                    label=str(item.get("label", f"segment-{idx}")),
                )
            )
        return segments

    if args.preset == "square":
        return _preset_square(args.base_height)
    if args.preset == "figure8":
        return _preset_figure8(args.base_height)
    if args.preset == "strafe_scan":
        return _preset_strafe_scan(args.base_height)
    raise ValueError(f"unsupported preset: {args.preset}")


def main():
    parser = argparse.ArgumentParser(description="H1-2 high-level motion sequence publisher")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--rate", type=float, default=50.0, help="publish rate in Hz")
    parser.add_argument("--base_height", type=float, default=0.8, help="neutral commanded body height")
    parser.add_argument(
        "--preset",
        choices=["square", "figure8", "strafe_scan"],
        default="square",
        help="built-in motion sequence",
    )
    parser.add_argument("--sequence_json", type=str, default=None, help="path to a JSON sequence override")
    parser.add_argument("--loop", action="store_true", default=False, help="repeat the sequence")
    parser.add_argument(
        "--reset",
        choices=["none", "object", "all"],
        default="all",
        help="optional reset command before starting motion",
    )
    args = parser.parse_args()

    segments = load_segments(args)
    period = 1.0 / max(args.rate, 1.0)

    ChannelFactoryInitialize(args.channel)
    run_pub = ChannelPublisher("rt/run_command/cmd", String_)
    run_pub.Init()
    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()

    if args.reset == "object":
        publish_string(reset_pub, "1")
    elif args.reset == "all":
        publish_string(reset_pub, "2")

    print("=" * 64)
    print("H1-2 MOTION SEQUENCE")
    print(f"preset/json: {args.preset if not args.sequence_json else args.sequence_json}")
    print(f"segments: {len(segments)}  loop: {args.loop}  rate: {args.rate:.1f} Hz")
    print("=" * 64)

    try:
        while True:
            for segment in segments:
                cmd = [segment.x_vel, segment.y_vel, segment.yaw_vel, segment.height]
                print(f"[motion-seq] {segment.label or 'segment'} -> {cmd} for {segment.duration:.2f}s")
                end_time = time.time() + segment.duration
                while time.time() < end_time:
                    publish_string(run_pub, str(cmd))
                    time.sleep(period)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        publish_string(run_pub, str([0.0, 0.0, 0.0, float(args.base_height)]))
        print("[motion-seq] stopped")


if __name__ == "__main__":
    main()
