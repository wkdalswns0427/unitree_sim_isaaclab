import torch

def base_lin_vel(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.root_lin_vel_b

def base_ang_vel(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.root_ang_vel_b

def projected_gravity(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.projected_gravity_b

def commanded_velocity(env, command_name: str):
    cmd = env.command_manager.get_command(command_name)
    return cmd

def joint_pos_rel(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.joint_pos - asset.data.default_joint_pos

def joint_vel_rel(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.joint_vel

def last_action(env):
    return env.action_manager.action