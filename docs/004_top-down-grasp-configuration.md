# Top-Down Grasp Configuration for SO-101

## Problem

The IK controller only handles position control, not orientation. For top-down grasping where the gripper needs to approach vertically, manual joint configuration is required.

## Working Configuration

### Start Position
```python
start_config = [0.0, 0.7, -1.0, 0.0, 0.0]  # pan, shoulder, elbow, wrist_flex, wrist_roll
```

- Cube position: `(0.28, 0.0, 0.015)` - centered, close to robot base

### Two-Phase Lowering Approach

**Phase 1: Rotate Wrist (300 steps)**
- Rotate wrist_flex (joint 3) from 0.0 to 1.65
- Keep shoulder fixed at 0.7
- Keep elbow fixed at -1.0
- Keep gripper open

```python
for step in range(300):
    t = step / 300
    data.ctrl[3] = 0.0 + t * 1.65  # wrist rotation
    data.ctrl[2] = -1.0  # elbow fixed
    data.ctrl[1] = 0.7   # shoulder fixed
    data.ctrl[5] = 1.0   # gripper open
```

**Phase 2: Lower Elbow (200 steps)**
- Lower elbow (joint 2) from -1.0 to -0.3
- This brings fingers down to cube level
- Keep gripper open throughout

```python
for step in range(200):
    t = step / 200
    data.ctrl[2] = -1.0 + t * 0.7  # elbow from -1.0 to -0.3
    data.ctrl[5] = 1.0  # gripper still open
```

**Close Gripper (100 steps)**
```python
data.ctrl[5] = -1.0  # close gripper
```

**Lift (100 steps)**
- Move shoulder (joint 1) from 1.25 to 0.85

## Results

During Hold phase with closed gripper:
```
step 0: contacts=[0, 0, 28, 30], cube_z=0.017
step 10: contacts=[0, 0, 28, 30], cube_z=0.017
step 20: contacts=[0, 28, 30], cube_z=0.018
step 30: contacts=[0, 0, 28, 30], cube_z=0.018
step 40: contacts=[0, 0, 28, 30], cube_z=0.018
```

- Geoms 28 and 30 are both finger contact points
- Cube lifted from z=0.015 to z=0.018 (friction-based grasp)

## Key Learnings

1. **Don't move shoulder during descent** - Keep joint 1 fixed during wrist rotation and elbow lowering
2. **Two separate phases required** - First orient the gripper (wrist rotation), then lower (elbow movement)
3. **Close gripper only after lowering** - Fingers must be at cube level before closing
4. **Elbow controls descent** - Joint 2 going from -1.0 to -0.3 brings the gripper down
5. **Wrist flex value 1.65** - This angle points the gripper straight down for top-down approach

## Joint Reference

| Joint | Index | Name | Range |
|-------|-------|------|-------|
| 0 | shoulder_pan | pan | - |
| 1 | shoulder_lift | shoulder | - |
| 2 | elbow_flex | elbow | [-1.69, 1.69] |
| 3 | wrist_flex | wrist | [-1.66, 1.66] |
| 4 | wrist_roll | roll | - |
| 5 | gripper | gripper | - |
