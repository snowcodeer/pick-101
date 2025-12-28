"""Test IK-based grasp - targeting finger position, not TCP."""
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from src.controllers.ik_controller import IKController

scene_path = Path("models/so101/lift_cube_ik_grasp.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

# Finger geom IDs for contact detection (old model without finger pads)
# 27, 28: static finger mesh geoms
# 29, 30: moving jaw mesh geoms

# Cube position
cube_x, cube_y = 0.32, 0.0

# Place cube
data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

# Pre-rotate wrist_roll by 90 degrees so fingers are horizontal
data.qpos[4] = np.pi / 2
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

ik = IKController(model, data)

def get_finger_mid():
    """Get midpoint between finger geoms."""
    f28 = data.geom_xpos[28]
    f30 = data.geom_xpos[30]
    return (f28 + f30) / 2

def get_tcp_to_finger_offset():
    """Get offset from TCP to finger midpoint in world coords."""
    tcp = ik.get_ee_position()
    finger_mid = get_finger_mid()
    return tcp - finger_mid

def get_contacts():
    """Get geom IDs contacting the cube."""
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

def is_grasping():
    """Check if both fingers contact cube."""
    contacts = get_contacts()
    has_static = 27 in contacts or 28 in contacts
    has_moving = 29 in contacts or 30 in contacts
    return has_static and has_moving

print("=== IK Grasp Test - Finger Position Targeting ===")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")
print(f"Static finger geoms: 27, 28 | Moving finger geoms: 29, 30")

with mujoco.viewer.launch_passive(model, data) as viewer:
    cube_z = 0.015

    # Step 1: Approach - target finger_mid slightly before cube
    print("\n--- Step 1: Approach ---")
    for step in range(400):
        offset = get_tcp_to_finger_offset()
        finger_target = np.array([cube_x - 0.03, cube_y, cube_z])
        tcp_target = finger_target + offset

        ctrl = ik.step_toward_target(tcp_target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65  # wrist_flex pointing down
        ctrl[4] = np.pi / 2  # wrist_roll horizontal
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

        if step % 100 == 0:
            finger_mid = get_finger_mid()
            print(f"  step {step}: finger_mid Z={finger_mid[2]:.4f}")

    # Step 2: Forward to cube
    print("\n--- Step 2: Forward ---")
    for step in range(400):
        offset = get_tcp_to_finger_offset()
        finger_target = np.array([cube_x, cube_y, cube_z])
        tcp_target = finger_target + offset

        ctrl = ik.step_toward_target(tcp_target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

    finger_mid = get_finger_mid()
    cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
    print(f"Finger mid: {finger_mid}")
    print(f"Cube: {cube_pos}")
    print(f"Finger mid Z vs Cube Z: {finger_mid[2]:.4f} vs {cube_pos[2]:.4f}")

    # Step 3: Close until grasp achieved
    print("\n--- Step 3: Close ---")
    grasp_gripper_action = 1.0
    for step in range(800):
        if is_grasping():
            print(f"  step {step}: GRASP ACHIEVED")
            break

        t = min(step / 600, 1.0)
        grasp_gripper_action = 1.0 - 2.0 * t

        offset = get_tcp_to_finger_offset()
        finger_target = np.array([cube_x, cube_y, cube_z])
        tcp_target = finger_target + offset

        ctrl = ik.step_toward_target(tcp_target, gripper_action=grasp_gripper_action, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

        if step % 100 == 0:
            gripper_pos = data.qpos[gripper_qpos_addr]
            print(f"  step {step}: gripper={gripper_pos:.3f}, contacts={get_contacts()}")

    # Step 4: Lift
    print("\n--- Step 4: Lift ---")
    lift_target_z = 0.10
    for step in range(300):
        t = min(step / 200, 1.0)
        current_z = cube_z + (lift_target_z - cube_z) * t

        offset = get_tcp_to_finger_offset()
        finger_target = np.array([cube_x, cube_y, current_z])
        tcp_target = finger_target + offset

        ctrl = ik.step_toward_target(tcp_target, gripper_action=grasp_gripper_action - 0.2, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

        if step % 50 == 0:
            cube_z_now = data.qpos[cube_qpos_addr + 2]
            print(f"  step {step}: target_z={current_z:.3f}, cube_z={cube_z_now:.3f}, grasping={is_grasping()}")

    cube_z_final = data.qpos[cube_qpos_addr + 2]
    print(f"\n=== Result ===")
    print(f"Cube Z: {cube_z_final:.4f}")
    print(f"Grasping: {is_grasping()}")
    print(f"Lifted: {cube_z_final > 0.05}")

    print("\nPress Ctrl+C to close...")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)
