import torch
from isaaclab.managers import SceneEntityCfg

def _exp_tracking(error: torch.Tensor, std: float):
    return torch.exp(-(error / std) ** 2)

def track_lin_vel_xy_exp(env, command_name: str, asset_name: str, std: float):
    asset = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    v_des = cmd[:, 0:2]
    v = asset.data.root_lin_vel_b[:, 0:2]
    err = torch.linalg.norm(v_des - v, dim=-1)
    return _exp_tracking(err, std)

def track_ang_vel_z_exp(env, command_name: str, asset_name: str, std: float):
    asset = env.scene[asset_name]
    cmd = env.command_manager.get_command(command_name)
    wz_des = cmd[:, 2]
    wz = asset.data.root_ang_vel_b[:, 2]
    err = torch.abs(wz_des - wz)
    return _exp_tracking(err, std)

def action_rate_l2(env):
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)

def joint_acc_l2(env, asset_name: str):
    asset = env.scene[asset_name]
    return torch.sum(asset.data.joint_acc**2, dim=-1)

def joint_torques_l2(env, asset_name: str):
    asset = env.scene[asset_name]
    return torch.sum(asset.data.applied_joint_effort**2, dim=-1)

def upright_bonus(env, asset_name: str):
    asset = env.scene[asset_name]
    # projected gravity z close to -1 means upright, depending on convention
    # Using +z axis up in body frame, adapt if sign differs
    g = asset.data.projected_gravity_b
    return 1.0 - torch.abs(g[:, 2] + 1.0)

def base_height_l2(env, asset_name: str, target_height: float):
    asset = env.scene[asset_name]
    z = asset.data.root_pos_w[:, 2]
    return (z - target_height) ** 2

def foot_slip_l2(env, asset_name: str, feet_body_names: list[str]):
    asset = env.scene[asset_name]
    feet_ids = asset.find_bodies(feet_body_names)
    v = asset.data.body_lin_vel_w[:, feet_ids, 0:2]
    return torch.sum(v**2, dim=(-1, -2))

def sustained_single_foot_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    max_single_stance_time: float = 0.45,
    force_threshold: float = 20.0,
):
    """Penalty for parking on one loaded foot instead of alternating steps."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)

    net_forces = contact_sensor.data.net_forces_w[:, body_ids, :]
    in_contact = torch.linalg.norm(net_forces, dim=-1) > force_threshold
    contact_count = torch.sum(in_contact, dim=-1)

    contact_time = contact_sensor.data.current_contact_time[:, body_ids]
    longest_contact = torch.max(torch.where(in_contact, contact_time, torch.zeros_like(contact_time)), dim=-1).values
    excess_single_stance = torch.clamp(longest_contact - max_single_stance_time, min=0.0)

    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return (contact_count == 1).float() * moving_command.float() * excess_single_stance

def swing_knee_flexion(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_flexion: float = 0.55,
    target_flexion: float = 0.9,
    force_threshold: float = 20.0,
):
    """Reward knee bend on the swing leg so the learned gait does not stay straight-legged."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    knee_pos = asset.data.joint_pos[:, joint_ids].squeeze(-1)

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    foot_force = torch.linalg.norm(contact_sensor.data.net_forces_w[:, body_ids, :], dim=-1).squeeze(-1)
    is_swing = foot_force < force_threshold

    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.norm(command[:, :2], dim=-1) > 0.1

    flexion_reward = torch.clamp((knee_pos - min_flexion) / (target_flexion - min_flexion), min=0.0, max=1.0)
    return flexion_reward * is_swing.float() * moving_command.float()

def knee_extension_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_flexion: float = 0.25,
):
    """Penalize locked knees while moving."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    knee_pos = asset.data.joint_pos[:, joint_ids]
    locked_knee = torch.clamp(min_flexion - knee_pos, min=0.0)

    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return torch.sum(locked_knee, dim=-1) * moving_command.float()

def alternating_leg_swing_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_joint_speed: float = 0.25,
    target_joint_speed: float = 1.2,
    anti_phase_std: float = 1.0,
):
    """Reward left/right leg swing with similar magnitude and opposite phase."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_vel = asset.data.joint_vel[:, joint_ids]
    left_hip_vel, left_knee_vel, right_hip_vel, right_knee_vel = joint_vel.unbind(dim=-1)

    hip_speed = 0.5 * (torch.abs(left_hip_vel) + torch.abs(right_hip_vel))
    hip_amp = torch.clamp((hip_speed - min_joint_speed) / (target_joint_speed - min_joint_speed), min=0.0, max=1.0)
    hip_anti_phase = torch.exp(-((left_hip_vel + right_hip_vel) / anti_phase_std) ** 2)
    hip_balance = torch.exp(-((torch.abs(left_hip_vel) - torch.abs(right_hip_vel)) / anti_phase_std) ** 2)

    knee_speed = 0.5 * (torch.abs(left_knee_vel) + torch.abs(right_knee_vel))
    knee_amp = torch.clamp((knee_speed - min_joint_speed) / (target_joint_speed - min_joint_speed), min=0.0, max=1.0)
    knee_balance = torch.exp(-((torch.abs(left_knee_vel) - torch.abs(right_knee_vel)) / anti_phase_std) ** 2)

    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return (0.7 * hip_amp * hip_anti_phase * hip_balance + 0.3 * knee_amp * knee_balance) * moving_command.float()

def flight_phase_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    force_threshold: float = 20.0,
):
    """Penalize having no foot in contact with the ground (i.e. a flight phase).

    Walking always keeps at least one foot loaded; jumping/hopping briefly
    leaves both feet airborne.  This penalty fires whenever both ankle-roll
    contact forces are below the threshold while the robot is commanded to
    move, directly discouraging bunny-hop gaits.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)

    forces = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, body_ids, :], dim=-1
    )
    in_contact = forces > force_threshold
    no_contact = ~torch.any(in_contact, dim=-1)

    command = env.command_manager.get_command(command_name)
    moving  = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return no_contact.float() * moving.float()


def leg_antisymmetry_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
):
    """Penalize in-phase (non-mirrored) motion between left and right sagittal leg joints.

    For normal walking the left/right hip, knee and ankle deviate from their
    default positions in opposite directions (anti-phase).  Their sum of
    deviations should therefore be near zero.  Squaring the sum gives a
    differentiable penalty that grows whenever both legs tilt/flex the same
    way — the root cause of the shuffle-drag asymmetry.

    Expected joint order (preserve_order=True):
        left_hip_pitch, left_knee, left_ankle_pitch,
        right_hip_pitch, right_knee, right_ankle_pitch
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)

    q         = asset.data.joint_pos[:, joint_ids]
    q_default = asset.data.default_joint_pos[:, joint_ids]
    delta     = q - q_default  # (N, 6)

    left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle = delta.unbind(dim=-1)

    hip_penalty   = (left_hip   + right_hip  ) ** 2
    knee_penalty  = (left_knee  + right_knee ) ** 2
    ankle_penalty = (left_ankle + right_ankle) ** 2

    command = env.command_manager.get_command(command_name)
    moving  = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return (hip_penalty + knee_penalty + ankle_penalty) * moving.float()


def symmetric_arm_swing(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_joint_speed: float = 0.15,
    target_joint_speed: float = 1.0,
    anti_phase_std: float = 0.8,
):
    """Reward human-like shoulder-pitch arm swing: left and right arms move opposite each other."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    left_shoulder_vel, right_shoulder_vel = asset.data.joint_vel[:, joint_ids].unbind(dim=-1)

    arm_speed = 0.5 * (torch.abs(left_shoulder_vel) + torch.abs(right_shoulder_vel))
    arm_amp = torch.clamp((arm_speed - min_joint_speed) / (target_joint_speed - min_joint_speed), min=0.0, max=1.0)
    anti_phase = torch.exp(-((left_shoulder_vel + right_shoulder_vel) / anti_phase_std) ** 2)
    balance = torch.exp(-((torch.abs(left_shoulder_vel) - torch.abs(right_shoulder_vel)) / anti_phase_std) ** 2)

    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.norm(command[:, :2], dim=-1) > 0.1
    return arm_amp * anti_phase * balance * moving_command.float()
