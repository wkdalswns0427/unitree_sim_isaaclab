# unitree_sim_isaaclab Local Run Guide (No Docker)

This guide is for running `unitree_sim_isaaclab` directly on your local Linux machine (no Docker), including:

- Running simulation tasks
- Sending keyboard movement commands
- Replaying existing episodes
- Generating `data.json` episodes for training
- (Optional) Converting episodes for `h1_mimic_tasks`

## 1. Prerequisites

- Ubuntu 22.04+ recommended
- NVIDIA driver + CUDA-capable GPU (CPU mode is also supported but slower)
- Isaac Sim + Isaac Lab already installed in your Python environment
- This repo cloned at:
  - `/home/{USER}/mj_ws/unitree_sim_isaaclab`

## 2. Environment Setup (Local)

From your host machine terminal:

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
pip install -r requirements.txt
```

Download assets once:

```bash
sudo apt update
sudo apt install -y git-lfs
. fetch_assets.sh
```

If you hit `ModuleNotFoundError: teleimager.image_server`, export:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
```

## 3. Run Simulation Locally

Example (GPU):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129
```

Notes:

- If `DISPLAY` is not set, Isaac Sim runs headless automatically.
- Headless `GLFW` warnings are common and not fatal by themselves.

## 4. Move Robot with Keyboard (G1 Wholebody Tasks)

[send_commands_keyboard.py](send_commands_keyboard.py) publishes high-level DDS run commands (`x/y/yaw/height`) and is intended for the G1 wholebody tasks (task IDs containing `Wholebody`). For H1-2 joint-level teleop, see [scene/keyboard_control.py](scene/keyboard_control.py) in §6 — that path drives joint targets directly and does not need a policy loaded.

Terminal A (run sim):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-G129-Dex1-Wholebody --enable_dex1_dds --robot_type g129
```

Terminal B (keyboard publisher):

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python send_commands_keyboard.py --backend stdin --channel 1
```

Default keys (from `send_commands_keyboard.py`):

- `W/S`: forward / backward
- `A/D`: left / right
- `Z/X`: rotate left / right
- `C`: crouch
- `SPACE`: reset commands to zero
- `Q`: quit publisher

Important:
- Commands are sent as a stream to the RL policy running inside `sim_main.py`. If the loaded policy was not trained for the robot/task pair, the robot may not move even though publishes succeed.
- `--backend pynput` is the alternative on desktop sessions; `stdin` is the right choice over SSH or in a headless terminal.

## 5. H1-2 Tasks (Training & Play)

Three RL tasks for the H1-2 robot are registered under [tasks/h1-2_tasks/](tasks/h1-2_tasks):

| Gym ID | Source file | Action subset | What the policy learns |
|---|---|---|---|
| `Isaac-H12-Velocity-Legonly-v0` | [h12_velocity/](tasks/h1-2_tasks/h12_velocity/) (flat_env_cfg + rough_env_cfg) | 13-DOF legs + torso | Tracks `base_velocity` (lin_vx, lin_vy, ang_wz) with anti-pendulum gait rewards (mirror symmetry, swing-knee flexion, event-based air-time). Arms held by actuator defaults. |
| `Isaac-H12-Velocity-Wholebody-v0` | [h12_velocity/wholebody_env_cfg.py](tasks/h1-2_tasks/h12_velocity/wholebody_env_cfg.py) | 21-DOF legs + torso + 8 arm joints | Same velocity command as legonly; arms are in the action space so the policy can swing them for momentum. Used for the 2026-05-08 / 2026-05-15 legacy runs. |
| `Isaac-H12-Squat-v0` | [h12_squat/squat_env_cfg.py](tasks/h1-2_tasks/h12_squat/squat_env_cfg.py) | 13-DOF legs + torso | Tracks a resampled pelvis-height target (alternating between `stand_high=1.00 m` and `squat_low=0.55 m`, 3-second holds). Has a `squat_depth` curriculum that widens the range over the first ~50k resets. |
| `Isaac-H12-Stand-v0` | [h12_stand/stand_env_cfg.py](tasks/h1-2_tasks/h12_stand/stand_env_cfg.py) | 13-DOF legs + torso | Holds the default pose at a fixed `STAND_PELVIS_HEIGHT=1.05 m` while resisting torso pushes, mass jitter, and external force/torque. Pure regulation task — no command. |

Each task also has a matching `*-Play-v0` ID that lowers `num_envs`, removes obs corruption, and uses a fixed/deterministic command — pass this to `play.py` for visual verification.

### Train

The main entry point is [scripts/reinforcement_learning/rsl_rl/train.py](scripts/reinforcement_learning/rsl_rl/train.py). It loads the env + RSL-RL configs from the gym registration, dumps `params/env.yaml` + `params/agent.yaml`, runs `OnPolicyRunner.learn(...)`, and auto-exports `policy.pt` (JIT) and `policy.onnx` to `<run>/exported/` when training finishes.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

# Velocity (legonly, 13-DOF) — recommended starting point
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-H12-Velocity-Legonly-v0 \
  --device cuda --headless \
  --num_envs 4096 --max_iterations 5000

# Velocity (wholebody, 21-DOF) — convenience wrapper pre-sets the task
python scripts/reinforcement_learning/rsl_rl/train_wholebody.py \
  --device cuda --headless \
  --num_envs 4096 --max_iterations 5000

# Squat (height-tracking, 13-DOF, has curriculum)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-H12-Squat-v0 \
  --device cuda --headless \
  --num_envs 4096 --max_iterations 5000

# Stand (regulation, 13-DOF)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-H12-Stand-v0 \
  --device cuda --headless \
  --num_envs 4096 --max_iterations 3000
```

Drop `--headless` to watch training in the GUI. Reduce `--num_envs` to 1024 if VRAM is limited (defaults to 64 when omitted, which is too few for stable PPO). Logs and checkpoints land under:

```
logs/rsl_rl/h12_velocity_legonly/<timestamp>/         ← Legonly
logs/rsl_rl/h12_velocity_wholebody/<timestamp>/       ← Wholebody
logs/rsl_rl/h12_squat/<timestamp>/                    ← Squat
logs/rsl_rl/h12_stand/<timestamp>/                    ← Stand
logs/rsl_rl/<exp>/<timestamp>/exported/policy.{pt,onnx}   ← auto-exported on completion
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/h12_velocity_legonly
```

### Play

[scripts/reinforcement_learning/rsl_rl/play.py](scripts/reinforcement_learning/rsl_rl/play.py) accepts an explicit `--task` and a `--checkpoint` path, then re-exports the loaded policy to `exported/` and rolls it out:

```bash
# Velocity (legonly)
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-H12-Velocity-Legonly-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/h12_velocity_legonly/2026-05-19_11-14-40/model_4999.pt

# Velocity (wholebody) — convenience wrapper pre-selects the wholebody task
python scripts/reinforcement_learning/rsl_rl/play_wholebody.py \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/h12_velocity_wholebody/2026-05-15_14-41-04/model_5000.pt

# Squat
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-H12-Squat-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/h12_squat/<timestamp>/model_5000.pt

# Stand
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-H12-Stand-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/h12_stand/<timestamp>/model_3000.pt
```

`--num_envs 1` is the cleanest for visual debugging. `--real-time` paces stepping to wall-clock. Add `--video` to record a single clip into `<run>/videos/play/`.

### Key config files per task

| Task | Env cfg | PPO cfg |
|---|---|---|
| Velocity Legonly | [h12_velocity/flat_env_cfg.py](tasks/h1-2_tasks/h12_velocity/flat_env_cfg.py) → [rough_env_cfg.py](tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py) | [h12_velocity/agents/rsl_rl_ppo_cfg.py](tasks/h1-2_tasks/h12_velocity/agents/rsl_rl_ppo_cfg.py) → `H12FlatPPORunnerCfg` |
| Velocity Wholebody | [h12_velocity/wholebody_env_cfg.py](tasks/h1-2_tasks/h12_velocity/wholebody_env_cfg.py) | `H12WholebodyFlatPPORunnerCfg` (same file) |
| Squat | [h12_squat/squat_env_cfg.py](tasks/h1-2_tasks/h12_squat/squat_env_cfg.py) | [h12_squat/agents/rsl_rl_ppo_cfg.py](tasks/h1-2_tasks/h12_squat/agents/rsl_rl_ppo_cfg.py) |
| Stand | [h12_stand/stand_env_cfg.py](tasks/h1-2_tasks/h12_stand/stand_env_cfg.py) | [h12_stand/agents/rsl_rl_ppo_cfg.py](tasks/h1-2_tasks/h12_stand/agents/rsl_rl_ppo_cfg.py) |

All four PPO runners share the same MLP and algorithm config — only `experiment_name` and `max_iterations` differ:

- Network: `[512, 256, 128]` ELU (actor + critic)
- `entropy_coef=0.01`, `learning_rate=1e-3` (adaptive), `desired_kl=0.01`, `num_learning_epochs=5`, `num_mini_batches=4`
- `num_steps_per_env=24`, `save_interval=50`
- `max_iterations`: 5000 (legonly / wholebody / squat) or 3000 (stand)
- Action scale: `0.5` (delta from default joint positions; play configs raise to `1.0`)

### Velocity reward composition (current)

Defined in `H12Rewards` in [rough_env_cfg.py](tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py), with overrides in `H12RoughEnvCfg.__post_init__` and [flat_env_cfg.py](tasks/h1-2_tasks/h12_velocity/flat_env_cfg.py):

| Term | Weight (flat) | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` (std=0.5) | +1.5 | Body-frame xy velocity tracking |
| `track_ang_vel_z_exp` (std=0.5) | +1.0 | Yaw-rate tracking |
| `feet_air_time` (threshold=0.3s) | +0.25 | Event-based — pays `last_air_time - thr` only on touchdown. Closes the "pendulum leg" loophole that `feet_air_time_positive_biped` allowed. |
| `no_single_foot_park` (max_stance=0.5s, F=20N) | -1.0 | Penalizes staying loaded on one foot too long. |
| `leg_mirror` | -2.0 | `(left_delta + right_delta)²` on hip-pitch / knee / ankle-pitch — forces anti-phase legs. |
| `flight_phase` (F=20N) | -1.0 | Penalizes frames where neither foot is loaded while commanded to move (anti-hop). |
| `swing_knee_flexion_{left,right}` (target=0.5 rad) | +0.3 each | Per-side knee flexion bonus during the swing phase. |
| `feet_slide` | -0.4 | Penalizes foot motion in contact. |
| `dof_pos_limits` (ankles only) | -1.0 | Ankle joint-limit barrier. |
| `joint_deviation_hip` | -0.2 | Hip yaw/roll near zero. |
| `joint_deviation_arms` | -0.4 | Arm anchor toward defaults (no active arm reward in legonly). |
| `joint_deviation_torso` | -0.1 | Torso joint near zero. |
| `flat_orientation_l2` | -2.5 | Keeps torso upright. |
| `ang_vel_xy_l2` | -0.05 | Penalizes roll/pitch rate. |
| `dof_acc_l2` (lower body) | -1.25e-7 | Joint smoothness. |
| `action_rate_l2` | -0.005 | Smooth action changes. |
| `dof_torques_l2` | 0.0 | Disabled (H1 baseline). |
| `lin_vel_z_l2` | None | Disabled — penalty would force a stiff drag gait. |
| `stand_still` | None | Disabled — competed with velocity tracking at low cmds. |
| `termination_penalty` | -200.0 | Discourages falls. |

`H12WholebodyFlatEnvCfg` inherits all of the above and only swaps the action / observation joint subset to `WHOLEBODY_JOINT_NAMES` (21 DOF).

### Squat reward composition

`H12SquatRewardsCfg` in [squat_env_cfg.py](tasks/h1-2_tasks/h12_squat/squat_env_cfg.py):

- `track_pelvis_height_exp` (+3.0, std=0.10) — Gaussian on `|z − target|`.
- `pelvis_height_l1` (-0.5) — close the last cm.
- `both_feet_grounded` (+0.5) + `feet_slide` (-0.4) — block the one-foot-balance + skating modes.
- `flat_orientation_l2` (-2.5), `lin_vel_xy_l2` (-1.0), `ang_vel_xy_l2` (-0.10), `ang_vel_z_l2` (-0.10) — stay in place and upright.
- `joint_deviation_arms` (-0.4), `joint_deviation_torso` (-2.0, heavy trunk anchor), `joint_deviation_hip` (-0.2).
- Curriculum `squat_depth` widens the height range from `(0.92, 1.00)` toward `(0.55, 1.00)` over `progression_resets=50000`.

### Stand reward composition

`H12StandRewardsCfg` in [stand_env_cfg.py](tasks/h1-2_tasks/h12_stand/stand_env_cfg.py):

- `alive` (+1.0) + `termination_penalty` (-200.0).
- `base_height_l2` (-10.0, target=1.05 m) — anchor at standing height.
- `flat_orientation_l2` (-5.0).
- Motion penalties: `lin_vel_xy_l2` (-2.0), `lin_vel_z_l2` (-2.0), `ang_vel_xy_l2` (-0.5), `ang_vel_z_l2` (-1.0).
- `both_feet_grounded` (+0.5), `feet_slide` (-0.4).
- Strong posture anchors: `joint_deviation_arms` (-1.0), `joint_deviation_torso` (-2.0), `joint_deviation_hip` (-0.5).
- Shorter episodes (`episode_length_s=10`), pushes every 4–8 s, sustained ±15 N / ±3 Nm torso force/torque, ±3 kg base mass + ±3 cm CoM jitter.

### Checkpoint compatibility

Older checkpoints trained with `log_std` (pre-rsl_rl API change) are auto-migrated on load — `train.py` also patches the `update_distribution` to clamp std into a numerically safe range, so a freshly initialized policy with bad std no longer crashes the first iteration.

### Common issues

- **Architecture mismatch on load**: PPO MLP shape must match the checkpoint. Old checkpoints used `[128, 128, 128]`; the current config is `[512, 256, 128]`. If you need to play an old checkpoint, edit `rsl_rl_ppo_cfg.py` to match.
- **Shuffling / stiff legs (velocity task)**: bump `feet_air_time.weight` and `feet_air_time.threshold` (currently 0.25 / 0.3 on flat); inspect `leg_mirror` weight; verify `action_scale=0.5`.
- **Falls backward**: ankle pitch default too positive; nudge `ankle_pitch` toward `-0.3` in [robots/unitree.py](robots/unitree.py).
- **Falls forward**: ankle pitch default too negative; try `-0.22`.
- **Squat: one-foot balance**: increase `both_feet_grounded` weight or tighten `feet_slide`.
- **Stand: drifts in place**: increase `lin_vel_xy_l2` / `flat_orientation_l2`; check that pushes aren't too strong for the policy stage.

---

## 6. Scene Folder

The [scene/](scene/) directory holds standalone Isaac Sim runners for the H1-2 robot — these are independent of the gym task pipeline above and load a USD scene directly. Useful for sandboxing PD gains, teleoperating joints, or running an exported policy outside of `play.py`.

### Contents

| File | Purpose |
|---|---|
| [scene/h1-2_cones.usd](scene/h1-2_cones.usd) | Default scene: H1-2 robot with FTP hands + a heavy-duty traffic cone. Used by `joint_motion.py` and `keyboard_control.py`. |
| [scene/configuration/](scene/configuration/) | Warehouse USD assets: `warehouse_h1_sc_base.usd` (~78 MB, main stage), `warehouse_h1_sc_physics.usd`, `warehouse_h1_sc_robot.usd`, `warehouse_h1_sc_sensor.usd`. Layered USDs for the larger warehouse demo scene. |
| [scene/open_scene.py](scene/open_scene.py) | Opens any USD via `AppLauncher`, pre-seeds PhysX drive attributes (target / stiffness / damping) on every joint **before** play, then teleports the robot into the validated standing stance (`H12_STANCE`) using `omni.isaac.dynamic_control`. Each frame re-sends position targets so the robot holds station against disturbances. |
| [scene/joint_motion.py](scene/joint_motion.py) | H1-2 stand-still + custom-motion sandbox. Authors SDK-matched PD gains (`_KP_SDK` / `_KD_SDK`) onto USD drive attributes before physics init, then runs in one of three modes: <br>• default — PD hold to `POLICY_JOINT_DEFAULTS` (no policy, no scripted motion);<br>• `--grab` — scripted arm reach + Surface Gripper sequence (mirrors `Krumi_python_sdk_unitree/example/h1_2/pick_cone_palms.py`);<br>• `--policy <path>` — loads a JIT-traced policy (default: `logs/rsl_rl/h12_velocity_legonly/2026-05-19_11-14-40/exported/policy.pt`) and drives the 21-DOF action subset using `--cmd_vx / --cmd_vy / --cmd_wz`. `--task {auto,legonly,squat}` selects the observation layout; `auto` probes the policy's input dim. |
| [scene/keyboard_control.py](scene/keyboard_control.py) | Keyboard joint teleop for the cones scene. Reuses the same SDK PD gains as `joint_motion.py`. Pelvis is pinned by default (`--no_pin_pelvis` to let legs balance; `--pin_feet` to also freeze the feet to world). Key bindings drive arms (`q/a w/s e/d r/f` and `u/j i/k o/l p/;`), torso (`t/g`), and coordinated leg poses (`5/6` stand/squat, `7/8` lean back/forward, `b/n` bend/un-bend); `0` snaps back to defaults, `-/=` scale the step rate, `ESC` quits. |
| [scene/run_bf_sim.txt](scene/run_bf_sim.txt) | Two-line ROS env helper — unsets CycloneDDS, sources `/opt/ros/humble/setup.bash`, and switches `RMW_IMPLEMENTATION` to FastRTPS. `source` it before running anything that talks DDS alongside ROS 2. |

### Common scene launches

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

# 1) Just open the cones scene and hold the validated standing stance.
python scene/open_scene.py

# 2) PD stand-still sandbox (no policy).
python scene/joint_motion.py

# 3) Cone-grab demo (scripted reach + Surface Gripper, no RL).
python scene/joint_motion.py --grab

# 4) Run an exported velocity policy in the cones scene at vx=0.4 m/s.
python scene/joint_motion.py \
  --policy logs/rsl_rl/h12_velocity_legonly/<timestamp>/exported/policy.pt \
  --cmd_vx 0.4

# 5) Keyboard joint teleop (pelvis pinned by default).
python scene/keyboard_control.py
python scene/keyboard_control.py --no_pin_pelvis      # let the legs balance
python scene/keyboard_control.py --pin_feet           # also freeze the feet
```

All scene runners accept `AppLauncher`'s standard flags (`--headless`, `--device cuda`, ...). `joint_motion.py` and `keyboard_control.py` both default `--scene` to `scene/h1-2_cones.usd`; pass `--scene <other.usd>` to load the warehouse stage or any custom USD.

---

## 7. Replay Existing Dataset

Use replay mode to load existing `data.json` episodes:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path /path/to/episode_root_or_data_json
```

Important:

- Use `--replay_data` (current code flag), not `--replay`.
- `--file_path` can be:
  - One `data.json`
  - A directory containing `episode_*/data.json`

## 8. Generate New `data.json` Episodes

In this repo, generation is wired through replay mode. Run replay + generation together:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path /path/to/source_dataset \
  --generate_data \
  --generate_data_dir ./data_gen
```

Optional flags:

- `--modify_light`
- `--modify_camera`
- `--rerun_log` (visualization only; not required for data generation)

Output layout:

```text
data_gen/
  episode_0000/
    data.json
    colors/
    depths/
    audios/
  episode_0001/
    ...
```

Quick check:

```bash
find ./data_gen -name data.json | sort
```

## 9. Convert Generated Episodes for `h1_mimic_tasks` (Optional)

If you want to feed these episodes into your mimic workflow:

```bash
cd /home/{USER}/mj_ws/IsaacLab_Humanoid/h1_mimic_tasks
conda activate {unitree_sim condaenv}
python scripts/mimic/import_unitree_reference.py \
  --input_path /home/{USER}/mj_ws/unitree_sim_isaaclab/data_gen \
  --output outputs/mimic/unitree_reference_raw.hdf5
```

## 10. H1-2 Wholebody -> Mimic Reference Pipeline

If your goal is to use H1-2 wholebody trajectories from this repo as Mimic reference data, use this exact flow.

Step 1: generate Unitree episodes (`data.json`) from replay.

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-H12-Velocity-Wholebody-v0 \
  --enable_inspire_dds \
  --robot_type h1_2 \
  --replay_data \
  --file_path /path/to/source_data_json_or_episode_dir \
  --generate_data \
  --generate_data_dir ./data_gen_h12
```

Step 2: verify generated episodes.

```bash
find ./data_gen_h12 -name data.json | sort
```

Step 3: convert to HDF5 reference in `h1_mimic_tasks`.

```bash
cd /home/{USER}/mj_ws/IsaacLab_Humanoid/h1_mimic_tasks
conda activate rical_unitree

python scripts/mimic/import_unitree_reference.py \
  --input_path /home/{USER}/mj_ws/unitree_sim_isaaclab/data_gen_h12 \
  --output outputs/mimic/unitree_reference_raw.hdf5 \
  --write_states
```

Step 4: run Mimic annotation and dataset generation.

```bash
export ISAACLAB_ROOT=/home/{USER}/RICAL_IsaacLab

python scripts/mimic/annotate_demos.py \
  --task H1-Pick-Block-Mimic-v0 \
  --input_file outputs/mimic/unitree_reference_raw.hdf5 \
  --output_file outputs/mimic/unitree_reference_annotated.hdf5 \
  --auto

python scripts/mimic/generate_dataset.py \
  --task H1-Pick-Block-Mimic-v0 \
  --input_file outputs/mimic/unitree_reference_annotated.hdf5 \
  --output_file outputs/mimic/unitree_reference_generated.hdf5
```

Important caveat:
- `import_unitree_reference.py` stores actions as `[left_arm, right_arm, left_ee, right_ee]`.
- `H1-Pick-Block-Mimic-v0` currently uses a 6D IK delta-pose action.
- So this is a reference-motion bridge, not guaranteed plug-and-play training data without action-space mapping.

## 11. Common Issues

- `No module named rerun.blueprint`:
  - `rerun` is optional unless you enable `--rerun_log`.
  - Data generation does not require rerun.
- `task_name ... is different from ...`:
  - Replay loader checks dataset task name. Match `--task` to source data.
- Headless warnings (`GLFW`, `MESA`, `left-click sim window`):
  - Usually expected on servers without display.
