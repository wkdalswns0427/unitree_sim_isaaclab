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

## 4. Move Robot with Keyboard (Wholebody Tasks Only)

Keyboard control publishes DDS run commands and is intended for tasks containing `Wholebody`.

Terminal A (run sim):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-G129-Dex1-Wholebody --enable_dex1_dds --robot_type g129

python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody --enable_inspire_dds --robot_type h1_2

```

Terminal B (keyboard publisher):

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python send_commands_keyboard.py --backend stdin --channel 1
```

Default keys:

- `W/S`: forward/backward
- `A/D`: left/right
- `Z/X`: rotate left/right
- `C`: crouch
- `Q`: quit keyboard publisher

Important:
- Keyboard control in `Wholebody` tasks sends high-level run commands (`x/y/yaw/height`) to the RL policy.
- If the loaded policy is not trained for your robot/task pair, the robot may not move even though commands are being published.

## 4.1 Train H1-2 Wholebody Policy (PPO)

This repo now includes a local RSL-RL training entrypoint for:
- `Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody`

Run training:

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody \
  --device cuda \
  --headless \
  --num_envs 64 \
  --max_iterations 3000
```

Outputs:
- checkpoints/logs: `logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/`
- exported policy: `logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/exported/policy.onnx`

Run sim with your trained ONNX:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody \
  --enable_inspire_dds \
  --robot_type h1_2 \
  --model_path logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/exported/policy.onnx
```

## 5. Replay Existing Dataset

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

## 6. Generate New `data.json` Episodes

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

## 7. Convert Generated Episodes for `h1_mimic_tasks` (Optional)

If you want to feed these episodes into your mimic workflow:

```bash
cd /home/{USER}/mj_ws/IsaacLab_Humanoid/h1_mimic_tasks
conda activate {unitree_sim condaenv}
python scripts/mimic/import_unitree_reference.py \
  --input_path /home/{USER}/mj_ws/unitree_sim_isaaclab/data_gen \
  --output outputs/mimic/unitree_reference_raw.hdf5
```

## 8. H1-2 Wholebody -> Mimic Reference Pipeline

If your goal is to use H1-2 wholebody trajectories from this repo as Mimic reference data, use this exact flow.

Step 1: generate Unitree episodes (`data.json`) from replay.

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Cylinder-H12-27dof-Inspire-Wholebody \
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

## 9. Common Issues

- `No module named rerun.blueprint`:
  - `rerun` is optional unless you enable `--rerun_log`.
  - Data generation does not require rerun.
- `task_name ... is different from ...`:
  - Replay loader checks dataset task name. Match `--task` to source data.
- Headless warnings (`GLFW`, `MESA`, `left-click sim window`):
  - Usually expected on servers without display.
