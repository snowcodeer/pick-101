"""Test top-down grasp with cube moved closer to robot."""
import mujoco
import numpy as np
import imageio
from pathlib import Path

scene_path = Path("models/so101/lift_cube.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.32, 0.0, 0.05]
cam.distance = 0.45
cam.azimuth = 135
cam.elevation = -25

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

def check_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

def capture_frame():
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()

def get_finger_positions():
    return data.geom_xpos[27].copy(), data.geom_xpos[29].copy()

output_dir = Path("runs/lift_curriculum_s1/20251219_052903")
output_dir.mkdir(parents=True, exist_ok=True)

frames = []

# Move cube closer - within arm's top-down reach
cube_x, cube_y = 0.33, 0.0  # Closer to robot, centered

print("=== Top-Down Grasp Test (Cube at X=0.33) ===")

# Position gripper above cube, then rotate wrist down
cube_x, cube_y = 0.28, 0.0

# Start with wrist_flex at 0.0 (horizontal), elbow at -1.0, shoulder at 0.7
start_config = [0.0, 0.7, -1.0, 0.0, 0.0]

# Place cube first, then set arm pose
data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

# Set arm pose
data.qpos[0:5] = start_config
data.qpos[gripper_qpos_addr] = 1.0
data.ctrl[:5] = data.qpos[:5]
data.ctrl[5] = 1.0

mujoco.mj_forward(model, data)

static_f, moving_j = get_finger_positions()
print(f"Start: finger z=({static_f[2]:.3f}, {moving_j[2]:.3f})")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")

frames.append(capture_frame())

# Settle
for _ in range(50):
    mujoco.mj_step(model, data)
frames.append(capture_frame())

# Phase 1: Rotate wrist only, keep gripper open
print("\n--- Phase 1: Rotate wrist ---")
for step in range(300):
    t = step / 300
    # Rotate wrist to point down (joint 3) - from 0.0 to 1.65
    data.ctrl[3] = 0.0 + t * 1.65
    # Keep elbow fixed
    data.ctrl[2] = -1.0
    # Keep shoulder fixed
    data.ctrl[1] = 0.7
    # Keep gripper open
    data.ctrl[5] = 1.0

    mujoco.mj_step(model, data)

    static_f, moving_j = get_finger_positions()
    finger_z = static_f[2]

    if step % 50 == 0:
        contacts = check_contacts()
        print(f"  step {step}: finger z={finger_z:.3f}, contacts={contacts}")
        frames.append(capture_frame())


# Phase 2: Lower elbow to get fingers to cube level
print("\n--- Phase 2: Lower elbow ---")
for step in range(200):
    t = step / 200
    data.ctrl[2] = -1.0 + t * 0.7  # from -1.0 to -0.3
    data.ctrl[5] = 1.0  # Keep gripper open

    mujoco.mj_step(model, data)

    if step % 40 == 0:
        contacts = check_contacts()
        cube_z = data.qpos[cube_qpos_addr + 2]
        static_f, _ = get_finger_positions()
        print(f"step {step}: finger_z={static_f[2]:.3f}, contacts={contacts}, cube_z={cube_z:.3f}")
        frames.append(capture_frame())

# Close gripper
print("\n--- Closing ---")
for step in range(100):
    data.ctrl[5] = -1.0
    mujoco.mj_step(model, data)

    if step % 20 == 0:
        gripper = data.qpos[gripper_qpos_addr]
        contacts = check_contacts()
        cube_z = data.qpos[cube_qpos_addr + 2]
        print(f"step {step}: gripper={gripper:.3f}, contacts={contacts}, cube_z={cube_z:.3f}")
        frames.append(capture_frame())

# Lift
print("\n--- Lifting ---")
for step in range(100):
    t = step / 100
    data.ctrl[1] = 1.25 - t * 0.4  # from 1.25 to 0.85
    data.ctrl[5] = -1.0

    mujoco.mj_step(model, data)

    if step % 20 == 0:
        contacts = check_contacts()
        cube_z = data.qpos[cube_qpos_addr + 2]
        static_f, _ = get_finger_positions()
        print(f"step {step}: finger_z={static_f[2]:.3f}, contacts={contacts}, cube_z={cube_z:.3f}")
        frames.append(capture_frame())

# Hold
print("\n--- Hold ---")
for step in range(50):
    mujoco.mj_step(model, data)
    if step % 10 == 0:
        contacts = check_contacts()
        cube_z = data.qpos[cube_qpos_addr + 2]
        print(f"step {step}: contacts={contacts}, cube_z={cube_z:.3f}")
        frames.append(capture_frame())

cube_final = data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy()
print(f"\n=== Final ===")
print(f"Cube: {cube_final}")
print(f"Lifted: {cube_final[2] > 0.05}")

video_path = output_dir / "grasp_test.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"\nSaved {len(frames)} frames to {video_path}")

renderer.close()
