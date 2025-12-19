"""Test IK-based grasp with wrist_roll pre-rotation."""
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from controllers.ik_controller import IKController

scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

cube_x, cube_y = 0.40, -0.10

# Place cube
data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

# Pre-rotate wrist_roll by 90 degrees so fingers are horizontal
data.qpos[4] = np.pi / 2  # wrist_roll
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

ik = IKController(model, data)

# Calculate finger midpoint offset from TCP
finger_27 = data.geom_xpos[27].copy()
finger_29 = data.geom_xpos[29].copy()
finger_mid_y = (finger_27[1] + finger_29[1]) / 2
tcp_y = ik.get_ee_position()[1]
y_offset = tcp_y - finger_mid_y  # How much to add to target Y to center fingers on cube

print("=== IK Grasp Test with wrist_roll=pi/2 ===")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")
print(f"Finger midpoint Y: {finger_mid_y:.4f}, TCP Y: {tcp_y:.4f}, offset: {y_offset:.4f}")

def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Step 1: Move to approach position (offset Y so fingers straddle cube)
    # Also recalculate Y offset dynamically as arm moves
    print("\n--- Step 1: Approach ---")
    for step in range(100):
        # Recalculate offset each step
        f27 = data.geom_xpos[27]
        f29 = data.geom_xpos[29]
        cur_finger_mid_y = (f27[1] + f29[1]) / 2
        cur_tcp_y = ik.get_ee_position()[1]
        cur_y_offset = cur_tcp_y - cur_finger_mid_y

        approach_pos = np.array([cube_x - 0.05, cube_y + cur_y_offset, 0.025])
        ctrl = ik.step_toward_target(approach_pos, gripper_action=1.0, gain=0.5)
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print(f"TCP: {ik.get_ee_position()}")
    print(f"Finger 27: {data.geom_xpos[27]}")
    print(f"Finger 29: {data.geom_xpos[29]}")
    print(f"Finger mid Y: {(data.geom_xpos[27][1] + data.geom_xpos[29][1])/2:.4f}, Cube Y: {cube_y}")

    # Step 2: Move forward to cube (dynamic Y offset)
    print("\n--- Step 2: Forward ---")
    for step in range(100):
        f27 = data.geom_xpos[27]
        f29 = data.geom_xpos[29]
        cur_finger_mid_y = (f27[1] + f29[1]) / 2
        cur_tcp_y = ik.get_ee_position()[1]
        cur_y_offset = cur_tcp_y - cur_finger_mid_y

        grasp_pos = np.array([cube_x, cube_y + cur_y_offset, 0.025])
        ctrl = ik.step_toward_target(grasp_pos, gripper_action=1.0, gain=0.5)
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print(f"TCP: {ik.get_ee_position()}")
    print(f"Cube: {data.qpos[cube_qpos_addr:cube_qpos_addr+3]}")

    # Step 3: Close gripper - wait until fully closed
    print("\n--- Step 3: Close ---")
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

    for step in range(300):  # More steps to fully close
        ctrl = ik.step_toward_target(grasp_pos, gripper_action=-1.0, gain=0.5)
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

        if step % 50 == 0:
            gripper_pos = data.qpos[gripper_qpos_addr]
            print(f"  step {step}: gripper={gripper_pos:.3f}, contacts={get_contacts()}")

    print(f"Contacts: {get_contacts()}")
    print(f"Cube: {data.qpos[cube_qpos_addr:cube_qpos_addr+3]}")
    print(f"Gripper: {data.qpos[gripper_qpos_addr]:.3f}")

    # Step 4: Lift
    print("\n--- Step 4: Lift ---")
    lift_pos = np.array([cube_x, cube_y + y_offset, 0.10])
    for step in range(100):
        ctrl = ik.step_toward_target(lift_pos, gripper_action=-1.0, gain=0.5)
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    contacts = get_contacts()
    print(f"Cube Z: {data.qpos[cube_qpos_addr+2]:.4f}")
    print(f"Contacts: {contacts}")

    has_27_28 = 27 in contacts or 28 in contacts
    has_29_30 = 29 in contacts or 30 in contacts
    print(f"Grasping (both fingers): {has_27_28 and has_29_30}")

    # Keep viewer open
    print("\nPress Ctrl+C to close...")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)
