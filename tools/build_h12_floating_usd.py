#!/usr/bin/env python3

# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Build a floating-base H1-2 USD from URDF for local tasks.

Example:
python3 tools/build_h12_floating_usd.py \
  --urdf /absolute/path/to/h1_2.urdf \
  --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build floating-base H1-2 USD from URDF.")
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Absolute path to H1-2 URDF file. If omitted, uses --variant under assets/robots/h1_2-wholebody-asset-urdf.",
)
parser.add_argument(
    "--variant",
    type=str,
    choices=["inspire", "ftp", "handless"],
    default="inspire",
    help="Built-in H1-2 URDF variant used when --urdf is not provided.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output USD path (relative to repo root or absolute path). If omitted, variant-specific default is used.",
)
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="Joint drive stiffness.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="Joint drive damping.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="Joint drive target type.",
)
parser.add_argument(
    "--force",
    action="store_true",
    default=True,
    help="Force USD reconversion.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    default_urdf_by_variant = {
        "inspire": "assets/robots/h1_2-wholebody-asset-urdf/h1_2.urdf",
        "ftp": "assets/robots/h1_2-wholebody-asset-urdf/h1_2_with_FTP_hand.urdf",
        "handless": "assets/robots/h1_2-wholebody-asset-urdf/h1_2_handless.urdf",
    }
    default_output_by_variant = {
        "inspire": "assets/robots/h1_2-26dof-inspire-floating-usd/h1_2_26dof_with_inspire_floating.usd",
        "ftp": "assets/robots/h1_2-26dof-ftp-floating-usd/h1_2_26dof_with_ftp_floating.usd",
        "handless": "assets/robots/h1_2-26dof-handless-floating-usd/h1_2_26dof_handless_floating.usd",
    }
    chosen_urdf = args_cli.urdf or default_urdf_by_variant[args_cli.variant]
    chosen_output = args_cli.output or default_output_by_variant[args_cli.variant]
    urdf_path = chosen_urdf if os.path.isabs(chosen_urdf) else os.path.abspath(os.path.join(repo_root, chosen_urdf))
    output_path = chosen_output if os.path.isabs(chosen_output) else os.path.abspath(os.path.join(repo_root, chosen_output))

    if not check_file_path(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(output_path),
        usd_file_name=os.path.basename(output_path),
        fix_base=False,
        make_instanceable=False,
        merge_fixed_joints=False,
        force_usd_conversion=bool(args_cli.force),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    print(f"[h12-floating] variant: {args_cli.variant}")
    print(f"[h12-floating] urdf: {urdf_path}")
    print("[h12-floating] converter config:")
    print_dict(cfg.to_dict(), nesting=0)
    converter = UrdfConverter(cfg)
    print(f"[h12-floating] done: {converter.usd_path}")
    print("[h12-floating] expected fix_base in generated config.yaml should be false.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
