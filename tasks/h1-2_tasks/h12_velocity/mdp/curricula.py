def increase_command_ranges(env, command_name: str, lin_vel_x, lin_vel_y, ang_vel_z, num_steps: int):
    # (start_min, start_max, end_min, end_max)
    # Isaac Lab curricula often run per iteration, use env.common_step_counter
    t = min(env.common_step_counter / float(num_steps), 1.0)

    cmd_cfg = env.command_manager.get_term_cfg(command_name)
    r = cmd_cfg.ranges

    r.lin_vel_x = (lin_vel_x[0] * (1 - t) + lin_vel_x[2] * t, lin_vel_x[1] * (1 - t) + lin_vel_x[3] * t)
    r.lin_vel_y = (lin_vel_y[0] * (1 - t) + lin_vel_y[2] * t, lin_vel_y[1] * (1 - t) + lin_vel_y[3] * t)
    r.ang_vel_z = (ang_vel_z[0] * (1 - t) + ang_vel_z[2] * t, ang_vel_z[1] * (1 - t) + ang_vel_z[3] * t)