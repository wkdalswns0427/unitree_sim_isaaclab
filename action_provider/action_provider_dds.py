# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
from typing import Optional
import torch
import time
from dds.dds_master import dds_manager


class DDSActionProvider(ActionProvider):
    """Action provider based on DDS"""
    
    def __init__(self,env, args_cli):
        super().__init__("DDSActionProvider")
        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.wait_for_first_robot_cmd = True
        self._received_first_robot_cmd = False
        self._hold_notice_printed = False
        self._post_reset_hold_until = 0.0
        self.env = env
        # Initialize DDS communication
        self.robot_dds = None
        self.gripper_dds = None
        self.dex3_dds = None
        self.inspire_dds = None
        self._setup_dds()
        self._setup_joint_mapping()
    
    def _setup_dds(self):
        """Setup DDS communication"""
        print(f"enable_robot: {self.enable_robot}")
        print(f"enable_gripper: {self.enable_gripper}")
        print(f"enable_dex3: {self.enable_dex3}")
        try:
            if self.enable_robot == "g129" or self.enable_robot == "h1_2":
                self.robot_dds = dds_manager.get_object("g129")
            if self.enable_gripper:
                self.gripper_dds = dds_manager.get_object("dex1")
            elif self.enable_dex3:
                self.dex3_dds = dds_manager.get_object("dex3")
            elif self.enable_inspire:
                self.inspire_dds = dds_manager.get_object("inspire")
            print(f"[{self.name}] DDS communication initialized")
        except Exception as e:
            print(f"[{self.name}] DDS initialization failed: {e}")
    
    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        self._robot_target_indices = []
        self._robot_source_indices = []

        arm_joint_names = []

        if self.enable_robot == "g129":
            body_joint_names = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ]
            arm_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            arm_source_offset = 15
        elif self.enable_robot == "h1_2":
            if "torso" in self.joint_to_index:
                body_joint_names = [
                    "left_hip_pitch",
                    "left_hip_roll",
                    "left_hip_yaw",
                    "left_knee",
                    "left_ankle",
                    None,
                    "right_hip_pitch",
                    "right_hip_roll",
                    "right_hip_yaw",
                    "right_knee",
                    "right_ankle",
                    None,
                    "torso",
                ]
                arm_joint_names = [
                    "left_shoulder_pitch",
                    "left_shoulder_roll",
                    "left_shoulder_yaw",
                    "left_elbow",
                ]
            else:
                body_joint_names = [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "torso_joint",
                ]
                arm_joint_names = [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ]
            arm_source_offset = 13
        else:
            body_joint_names = []
            arm_source_offset = 0

        for source_idx, name in enumerate(body_joint_names):
            if name is not None and name in self.joint_to_index:
                self._robot_target_indices.append(self.joint_to_index[name])
                self._robot_source_indices.append(source_idx)

        self.arm_joint_mapping = {name: i for i, name in enumerate(arm_joint_names)}
        self.arm_action_pose = []
        self.arm_action_pose_indices = []
        for arm_idx, name in enumerate(arm_joint_names):
            if name in self.joint_to_index:
                self.arm_action_pose.append(self.joint_to_index[name])
                self.arm_action_pose_indices.append(arm_idx)
                self._robot_target_indices.append(self.joint_to_index[name])
                self._robot_source_indices.append(arm_source_offset + arm_idx)

        if self.enable_gripper:
            self.gripper_joint_mapping = {
                "left_hand_Joint1_1": 1,
                "left_hand_Joint2_1": 1,
                "right_hand_Joint1_1": 0,
                "right_hand_Joint2_1": 0,
            }
        if self.enable_dex3:
            self.left_hand_joint_mapping = {
                "left_hand_thumb_0_joint":0,
                "left_hand_thumb_1_joint":1,
                "left_hand_thumb_2_joint":2,
                "left_hand_middle_0_joint":3,
                "left_hand_middle_1_joint":4,
                "left_hand_index_0_joint":5,
                "left_hand_index_1_joint":6}
            self.right_hand_joint_mapping = {
                "right_hand_thumb_0_joint":0,     
                "right_hand_thumb_1_joint":1,
                "right_hand_thumb_2_joint":2,
                "right_hand_middle_0_joint":3,
                "right_hand_middle_1_joint":4,
                "right_hand_index_0_joint":5,
                "right_hand_index_1_joint":6}
        if self.enable_inspire:
            self.inspire_hand_joint_mapping = {
                "R_pinky_proximal_joint":0,
                "R_ring_proximal_joint":1,
                "R_middle_proximal_joint":2,
                "R_index_proximal_joint":3,
                "R_thumb_proximal_pitch_joint":4,
                "R_thumb_proximal_yaw_joint":5,
                "L_pinky_proximal_joint":6,
                "L_ring_proximal_joint":7,
                "L_middle_proximal_joint":8,
                "L_index_proximal_joint":9,
                "L_thumb_proximal_pitch_joint":10,
                "L_thumb_proximal_yaw_joint":11,
            }
            self.special_joint_mapping = {
                "L_index_intermediate_joint":[9,1],
                "L_middle_intermediate_joint":[8,1],
                "L_pinky_intermediate_joint":[6,1],
                "L_ring_intermediate_joint":[7,1],
                "L_thumb_intermediate_joint":[10,1.5],
                "L_thumb_distal_joint":[10,2.4],

                "R_index_intermediate_joint":[3,1],
                "R_middle_intermediate_joint":[2,1],
                "R_pinky_intermediate_joint":[0,1],
                "R_ring_intermediate_joint":[1,1],
                "R_thumb_intermediate_joint":[4,1.5],
                "R_thumb_distal_joint":[4,2.4],
            }

        
        # precompute indices (for vectorization)

        if self.enable_gripper:
            self._gripper_target_indices = []
            self._gripper_source_indices = []
            for name, source_idx in self.gripper_joint_mapping.items():
                if name in self.joint_to_index:
                    self._gripper_target_indices.append(self.joint_to_index[name])
                    self._gripper_source_indices.append(source_idx)
        if self.enable_dex3:
            self._left_hand_target_indices = []
            self._left_hand_source_indices = []
            for name, source_idx in self.left_hand_joint_mapping.items():
                if name in self.joint_to_index:
                    self._left_hand_target_indices.append(self.joint_to_index[name])
                    self._left_hand_source_indices.append(source_idx)
            self._right_hand_target_indices = []
            self._right_hand_source_indices = []
            for name, source_idx in self.right_hand_joint_mapping.items():
                if name in self.joint_to_index:
                    self._right_hand_target_indices.append(self.joint_to_index[name])
                    self._right_hand_source_indices.append(source_idx)
        if self.enable_inspire:
            self._inspire_target_indices = []
            self._inspire_source_indices = []
            for name, source_idx in self.inspire_hand_joint_mapping.items():
                if name in self.joint_to_index:
                    self._inspire_target_indices.append(self.joint_to_index[name])
                    self._inspire_source_indices.append(source_idx)
            self._inspire_special_target_indices = []
            self._inspire_special_source_indices = []
            self._inspire_special_scales = []
            for name, spec in self.special_joint_mapping.items():
                if name in self.joint_to_index:
                    self._inspire_special_target_indices.append(self.joint_to_index[name])
                    self._inspire_special_source_indices.append(spec[0])
                    self._inspire_special_scales.append(spec[1])
            self._inspire_special_scales = torch.tensor(self._inspire_special_scales, dtype=torch.float32)
        
        device = self.env.device
        self._robot_target_idx_t = torch.tensor(self._robot_target_indices, dtype=torch.long, device=device)
        self._robot_source_idx_t = torch.tensor(self._robot_source_indices, dtype=torch.long, device=device)
        if self.enable_gripper:
            self._gripper_target_idx_t = torch.tensor(self._gripper_target_indices, dtype=torch.long, device=device)
            self._gripper_source_idx_t = torch.tensor(self._gripper_source_indices, dtype=torch.long, device=device)
        if self.enable_dex3:
            self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_indices, dtype=torch.long, device=device)
            self._left_hand_source_idx_t = torch.tensor(self._left_hand_source_indices, dtype=torch.long, device=device)
            self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_indices, dtype=torch.long, device=device)
            self._right_hand_source_idx_t = torch.tensor(self._right_hand_source_indices, dtype=torch.long, device=device)
        if self.enable_inspire:
            self._inspire_target_idx_t = torch.tensor(self._inspire_target_indices, dtype=torch.long, device=device)
            self._inspire_source_idx_t = torch.tensor(self._inspire_source_indices, dtype=torch.long, device=device)
            self._inspire_special_target_idx_t = torch.tensor(self._inspire_special_target_indices, dtype=torch.long, device=device)
            self._inspire_special_source_idx_t = torch.tensor(self._inspire_special_source_indices, dtype=torch.long, device=device)
            self._inspire_special_scales_t = self._inspire_special_scales.to(device)
        
        self._full_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)
        robot_cmd_size = max(self._robot_source_indices) + 1 if self._robot_source_indices else 0
        self._positions_buf = torch.empty(robot_cmd_size, device=device, dtype=torch.float32)
        self._startup_stand_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)
        if self.enable_robot == "h1_2":
            if "torso" in self.joint_to_index:
                startup_pose = {
                    "left_hip_pitch": -0.28,
                    "right_hip_pitch": -0.28,
                    "left_knee": 0.79,
                    "right_knee": 0.79,
                    "left_ankle": -0.52,
                    "right_ankle": -0.52,
                    "torso": 0.0,
                }
            else:
                startup_pose = {
                    # mirrored sagittal stance for H1-2
                    "left_hip_pitch_joint": 0.28,
                    "right_hip_pitch_joint": -0.28,
                    "left_knee_joint": -0.62,
                    "right_knee_joint": 0.62,
                    "left_ankle_pitch_joint": -0.32,
                    "right_ankle_pitch_joint": 0.32,
                    "torso_joint": 0.0,
                }
            for joint_name, joint_value in startup_pose.items():
                joint_idx = self.joint_to_index.get(joint_name, None)
                if joint_idx is not None:
                    self._startup_stand_action_buf[joint_idx] = joint_value
        if self.enable_gripper:
            self._gripper_buf = torch.empty(2, device=device, dtype=torch.float32)
        if self.enable_dex3:
            self._left_hand_buf = torch.empty(len(self._left_hand_source_indices), device=device, dtype=torch.float32)
            self._right_hand_buf = torch.empty(len(self._right_hand_source_indices), device=device, dtype=torch.float32)
        if self.enable_inspire:
            self._inspire_buf = torch.empty(12, device=device, dtype=torch.float32)
    
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        try:

            full_action = self._full_action_buf
            full_action.zero_()
            robot_cmd_active = False
            if self.enable_robot in ("g129", "h1_2") and self.robot_dds and self._positions_buf.numel() > 0:
                cmd_data = self.robot_dds.get_robot_command()
                if cmd_data and "motor_cmd" in cmd_data:
                    positions = cmd_data["motor_cmd"].get("positions", [])
                    if len(positions) >= self._positions_buf.numel():
                        pos_tensor = torch.as_tensor(
                            positions[: self._positions_buf.numel()],
                            dtype=torch.float32,
                            device=self.env.device,
                        )
                        self._positions_buf.copy_(pos_tensor)
                        robot_vals = self._positions_buf.index_select(0, self._robot_source_idx_t)
                        full_action.index_copy_(0, self._robot_target_idx_t, robot_vals)
                        if torch.max(torch.abs(robot_vals)).item() > 1e-4:
                            robot_cmd_active = True
                            self._received_first_robot_cmd = True

            now = time.time()
            in_post_reset_hold = self.enable_robot == "h1_2" and now < self._post_reset_hold_until
            if in_post_reset_hold:
                full_action.copy_(self._startup_stand_action_buf)
            elif self.wait_for_first_robot_cmd and self.enable_robot == "h1_2" and not self._received_first_robot_cmd:
                full_action.copy_(self._startup_stand_action_buf)
                if not self._hold_notice_printed:
                    print(f"[{self.name}] holding H1-2 standing pose until first DDS robot command")
                    self._hold_notice_printed = True
            elif robot_cmd_active and self._hold_notice_printed:
                print(f"[{self.name}] first DDS robot command received; releasing standing hold")
                self._hold_notice_printed = False
            # Get gripper command
            if self.gripper_dds:
                gripper_cmd = self.gripper_dds.get_gripper_command()
                if gripper_cmd:
                    left_gripper_cmd = gripper_cmd.get('left_gripper_cmd', {})
                    right_gripper_cmd = gripper_cmd.get('right_gripper_cmd', {})
                    left_gripper_positions = left_gripper_cmd.get('positions', [])
                    right_gripper_positions = right_gripper_cmd.get('positions', [])
                    gripper_positions = right_gripper_positions + left_gripper_positions
                    if len(gripper_positions) >= 2:
                        self._gripper_buf.copy_(torch.tensor(gripper_positions[:2], dtype=torch.float32, device=self.env.device))
                        gp_vals = self._gripper_buf.index_select(0, self._gripper_source_idx_t)
                        full_action.index_copy_(0, self._gripper_target_idx_t, gp_vals)
             
            elif self.dex3_dds:
                hand_cmds = self.dex3_dds.get_hand_commands()
                if hand_cmds:
                    left_hand_cmd = hand_cmds.get('left_hand_cmd', {})
                    right_hand_cmd = hand_cmds.get('right_hand_cmd', {})
                    if left_hand_cmd and right_hand_cmd:
                        left_positions = left_hand_cmd.get('positions', [])
                        right_positions = right_hand_cmd.get('positions', [])
                        if len(left_positions) >= len(self._left_hand_buf) and len(right_positions) >= len(self._right_hand_buf):
                            self._left_hand_buf.copy_(torch.tensor(left_positions[:len(self._left_hand_buf)], dtype=torch.float32, device=self.env.device))
                            self._right_hand_buf.copy_(torch.tensor(right_positions[:len(self._right_hand_buf)], dtype=torch.float32, device=self.env.device))
                            l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                            r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                            full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                            full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
            elif self.inspire_dds:
                inspire_cmds = self.inspire_dds.get_inspire_hand_command()
                if inspire_cmds and 'positions' in inspire_cmds:
                        inspire_cmds_positions = inspire_cmds['positions']
                        if len(inspire_cmds_positions) >= 12:
                            self._inspire_buf.copy_(torch.tensor(inspire_cmds_positions[:12], dtype=torch.float32, device=self.env.device))
                            base_vals = self._inspire_buf.index_select(0, self._inspire_source_idx_t)
                            full_action.index_copy_(0, self._inspire_target_idx_t, base_vals)
                            special_vals = self._inspire_buf.index_select(0, self._inspire_special_source_idx_t) * self._inspire_special_scales_t
                            full_action.index_copy_(0, self._inspire_special_target_idx_t, special_vals)
            return full_action.unsqueeze(0)
            
        except Exception as e:
            print(f"[{self.name}] Get DDS action failed: {e}")
            return None

    def arm_post_reset_stand_hold(self, duration_s: float = 0.8) -> None:
        """Hold H1-2 at stand pose after reset to avoid immediate collapse."""
        if self.enable_robot != "h1_2":
            return
        self._post_reset_hold_until = max(self._post_reset_hold_until, time.time() + max(0.0, float(duration_s)))
        print(f"[{self.name}] post-reset standing hold armed for {duration_s:.2f}s")
    
    def _convert_to_joint_range(self, value):
        """Convert gripper control value to joint angle"""
        input_min, input_max = 0.0, 5.6
        output_min, output_max = 0.03, -0.02
        value = max(input_min, min(input_max, value))
        return output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)
    
    def cleanup(self):
        """Clean up DDS resources"""
        try:
            if self.robot_dds:
                self.robot_dds.stop_communication()
            if self.gripper_dds:
                self.gripper_dds.stop_communication()
            if self.dex3_dds:
                self.dex3_dds.stop_communication()
            if self.inspire_dds:
                self.inspire_dds.stop_communication()
        except Exception as e:
            print(f"[{self.name}] Clean up DDS resources failed: {e}")
