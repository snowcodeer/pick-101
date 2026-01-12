# pick-101

RL training for SO-101 to grasp and lift a petri dish lid, using state-based SAC and a curriculum.
Commands adapted for Windows.

## Forked From

Forked from the original pick-101 by Gota Gando: https://github.com/ggand0/pick-101

## Installation (Windows PowerShell)

```powershell
git clone https://github.com/snowcodeer/pick-101
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
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick_lid_free_dish.py --viewer
```

## Project Structure

```
models/so101/                  # Robot models and scenes
  so101_new_calib.xml          # Current robot with finger pads
  lift_petri_lid.xml           # Petri lid scene
  lift_petri_lid_free_dish.xml # Petri lid (free dish base)
  assets/                      # STL mesh files (Git LFS)

src/
├── controllers/
│   └── ik_controller.py       # Damped least-squares IK
└── envs/
    ├── lift_cube.py           # Gym environment with reward versions
    ├── lift_petri_lid.py      # Petri lid env
    └── lift_petri_lid_free_dish.py # Petri lid (free dish base) env   
  

configs/                       # Training configs
├── grasp_stage1_close.yaml     # Petri lid stage 1
├── grasp_stage2_close.yaml     # Petri lid stage 2
├── grasp_stage3_full_lift.yaml # Petri lid stage 3
├── grasp_stage4_full_task.yaml # Petri lid stage 4
├── lift_petri_lid_free_dish_s3.yaml # First petri lid curriculum s3
└── lift_petri_lid_free_dish_500k.yaml # First petri lid 500k run

tests/                         # Test scripts
scripts/                       # Utilities and plotting
examples/                      # Small demos and viewers
docs/                          # Additional documentation
```

## Test Scripts

| Script | Description |
|--------|-------------|
| `test_topdown_pick_lid.py` | Top-down pick on petri lid |
| `test_topdown_pick_lid_free_dish.py` | Top-down pick on lid with free dish base |
| `test_topdown_pick_video.py` | Video capture for top-down pick |
| `test_topdown_pick_lid_video.py` | Video capture for lid pick |
| `test_topdown_pick_lid_free_dish_video.py` | Video capture for lid + free dish base |

### Top-Down Pick (`test_topdown_pick.py`)

The current best pick-and-place implementation. Uses a 4-step sequence:

1. **Move above block** - Position fingertips 30mm above cube, gripper open
2. **Descend to block** - Lower to grasp height, gripper open
3. **Close gripper** - Gradual close with contact detection, then tighten
4. **Lift** - Raise cube to target height

Key features:
- Uses `gripperframe` site at fingertips for precise IK targeting
- Finger pad collision boxes for stable multi-contact grasping
- Elliptic cone friction model to prevent slip
- Contact detection to stop closing at optimal grip force
- Locked wrist joints (`wrist_flex=90 deg`, `wrist_roll=90 deg`) for top-down orientation

```powershell
# Headless
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick.py

# With viewer
$env:PYTHONPATH="."; uv run python tests/test_topdown_pick.py --viewer
```

## State-Based RL (SAC)

Training uses low-dimensional state observations for petri dish lid grasping and lifting. The main entrypoint is `train_lift_petri_lid_free_dish.py`, with curriculum configs in `configs/grasp_stage*.yaml` (see below).

Training outputs are saved to `runs/lift_petri_lid_free_dish_s*/<timestamp>/` and include checkpoints, normalization stats, and tensorboard logs.

## Petri Lid Curriculum (Free Dish Base)

Progressive stages for delicate lid grasping and lifting. Each stage aligns with `curriculum_stage` in the config.

**Visualize curriculum stages (sanity check)**
```powershell
$env:PYTHONPATH="."; uv run python visualize_curriculum.py --config configs/grasp_stage1_close.yaml --episodes 3
```
Saves videos to `curriculum_videos/`.

**Stage 1: Hold grasped lid (150k steps) - easiest**
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

**Stage 4: Reach, grasp, and lift (600k steps) - hardest**
```powershell
$env:PYTHONPATH="."; uv run python train_lift_petri_lid_free_dish.py `
  --config configs/grasp_stage4_full_task.yaml `
  --pretrained runs/grasp_stage3_full_lift/<timestamp>/model_400000.zip
```
- curriculum_stage=4: Gripper far from lid
- Focus: Reaching + descending + grasping + lifting
- Success: Full reach and lift

### Evaluation (Petri Lid)

```powershell
$env:PYTHONPATH="."; uv run python eval_petri_lid_free_dish.py `
  --run runs/lift_petri_lid_free_dish_s3/<timestamp> `
  --checkpoint 100000
```

## Petri Lid Scripts

- `train_lift_petri_lid_free_dish.py` - Curriculum training entrypoint
- `eval_petri_lid_free_dish.py` - Petri lid evaluation runner
- `visualize_arm_petri.py` - Simple arm + petri dish viewer
- `visualize_curriculum.py` - Curriculum stage visualization

## Robot Model Variants

| Model | Features | Use Case |
|-------|----------|----------|
| `so101_new_calib.xml` | Finger pads, fingertip sites | Current development |
| `so101_ik_grasp.xml` | Original model (no pads) | Legacy compatibility |
| `so101_horizontal_grasp.xml` | Horizontal approach config | Experimental |