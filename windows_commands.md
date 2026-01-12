# Windows PowerShell Commands

This file contains Windows PowerShell equivalents for all commands in README.md.

## Installation

```powershell
git clone git@github.com:ggand0/pick-101.git
cd pick-101
uv sync
```

Assets (STL meshes) are stored via Git LFS and pulled automatically. Alternatively, copy from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100):
```powershell
Copy-Item -Recurse SO-ARM100/Simulation/SO101/assets models/so101/
```

## Quick Start

Run the top-down pick demo:

```powershell
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick.py --viewer
```

## Test Scripts

### Top-Down Pick (Headless)
```powershell
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick.py
```

### Top-Down Pick (With Viewer)
```powershell
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick.py --viewer
```

## State-Based RL (SAC)

Train an RL agent using low-dimensional state observations:

```powershell
$env:PYTHONPATH="."; uv run python train_lift.py --config configs/curriculum_stage3.yaml
```

### Progressive Curriculum (Petri Lid)

Each stage number matches curriculum_stage for clarity:

**Visualize curriculum stages (sanity check)**
```powershell
$env:PYTHONPATH="."; uv run python visualize_curriculum.py --config configs/grasp_stage1_close.yaml --episodes 3
```
This saves videos to `curriculum_videos/` showing the initial state and behavior for each stage.

**Stage 1: Hold grasped lid (150k steps) - Easiest**
```powershell
$env:PYTHONPATH="."; uv run python train_lift_petri_lid_free_dish.py --config configs/grasp_stage1_close.yaml
```
- curriculum_stage=1: Lid already in gripper, lifted
- Focus: Maintaining grasp at target height
- Success: Hold at 8cm

**Stage 2: Close and grasp lid (250k steps)**
```powershell
$env:PYTHONPATH="."; uv run python train_lift_petri_lid_free_dish.py `
  --config configs/grasp_stage2_close.yaml `
  --pretrained runs/grasp_stage1_close/<timestamp>/model_150000.zip
```
- curriculum_stage=2: Gripper at grasp height, open
- Focus: Close to grasp and hold
- Success: Grasp and hold at rest height

**Stage 3: Grasp and lift (400k steps)**
```powershell
$env:PYTHONPATH="."; uv run python train_lift_petri_lid_free_dish.py `
  --config configs/grasp_stage3_full_lift.yaml `
  --pretrained runs/grasp_stage2_close/<timestamp>/model_250000.zip
```
- curriculum_stage=3: Gripper near lid, open
- Focus: Grasping delicate lid + lifting
- Success: Grasp, lift to 8cm and hold

**Stage 4: Reach, grasp, and lift (600k steps) - Hardest**
```powershell
$env:PYTHONPATH="."; uv run python train_lift_petri_lid_free_dish.py `
  --config configs/grasp_stage4_full_task.yaml `
  --pretrained runs/grasp_stage3_full_lift/<timestamp>/model_400000.zip
```
- curriculum_stage=4: Gripper far from lid
- Focus: Reaching + descending + grasping + lifting
- Success: Full reach and lift

### Evaluation

Evaluate a trained model and generate videos:

```powershell
$env:PYTHONPATH="."; uv run python eval_cartesian.py `
  --run runs/lift_curriculum_s3/<timestamp> `
  --checkpoint 1000000
```

Evaluate a petri lid (free dish base) model and generate videos:

```powershell
$env:PYTHONPATH="."; uv run python eval_petri_lid_free_dish.py `
  --run runs/lift_petri_lid_free_dish_s3/<timestamp> `
  --checkpoint 100000
```

### Resume Training

To continue training from a checkpoint:

```powershell
$env:PYTHONPATH="."; uv run python train_lift.py `
  --config configs/curriculum_stage3.yaml `
  --resume runs/lift_curriculum_s3/<timestamp> `
  --timesteps 500000
```

## Image-Based RL (DrQ-v2)

Train an RL agent using wrist camera observations:

```powershell
$env:MUJOCO_GL="egl"; uv run python src/training/train_image_rl.py `
    --config configs/drqv2_lift_s3_v19.yaml
```

### Evaluate Checkpoint

```powershell
$env:MUJOCO_GL="egl"; uv run python src/training/eval_checkpoint.py `
    runs/image_rl/<timestamp>/snapshots/2000000_snapshot.pt `
    --num_episodes 10 `
    --reward_version v19 `
    --output_dir runs/image_rl/<timestamp>/eval
```

### X-Format Video (Side-by-Side Views)

```powershell
$env:MUJOCO_GL="egl"; uv run python src/training/eval_checkpoint.py `
    runs/image_rl/<timestamp>/snapshots/2000000_snapshot.pt `
    --num_episodes 5 `
    --reward_version v19 `
    --x-format `
    --output_dir runs/image_rl/<timestamp>/eval_x_post
```

## Visualization

### Simple Robot + Petri Dish Viewer

View just the robot arm and a petri dish (useful for lab automation visualization):

```powershell
uv run python visualize_arm_petri.py
```

### Camera Visualization

Multiple camera views of the arm:

```powershell
$env:PYTHONPATH="."; uv run python scripts/visualize_camera.py
```

## Notes

- PowerShell uses backticks (`` ` ``) for line continuation instead of backslashes (`\`)
- Environment variables are set with `$env:VARNAME="value";` instead of `VARNAME=value`
- The semicolon after environment variable assignment allows running the command on the same line
- `Copy-Item -Recurse` is the PowerShell equivalent of `cp -r`
