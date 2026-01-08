"""Top-down pick adapted from ECE4560 SO-101 assignment.

Clean 4-step sequence:
1. Move above lid (30mm height), gripper open
2. Move down to lid, gripper open
3. Close gripper
4. Lift back up

Reference: https://maegantucker.com/ECE4560/assignment8-so101/
"""
import sys
from pathlib import Path

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import mujoco.viewer
import numpy as np
import time
from src.controllers.ik_controller import IKController

scene_path = Path("models/so101/lift_petri_lid.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

lid_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "petri_lid_joint")
lid_qpos_addr = model.jnt_qposadr[lid_joint_id]
lid_top_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_top")
lid_skirt_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_skirt")
graspframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")
gripperframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
static_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")
moving_fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "moving_fingertip")
static_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
moving_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")


def get_fingertip_midpoint():
    static_tip = data.site_xpos[static_fingertip_id].copy()
    moving_tip = data.site_xpos[moving_fingertip_id].copy()
    return (static_tip + moving_tip) / 2


def add_marker_sphere(scn, pos, rgba, size=0.008):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0, 0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    scn.ngeom += 1


def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == lid_top_geom_id or g1 == lid_skirt_geom_id:
            other = g2
            contacts.append(other)
        elif g2 == lid_top_geom_id or g2 == lid_skirt_geom_id:
            other = g1
            contacts.append(other)
    return contacts


def is_grasping():
    contacts = get_contacts()
    has_static = static_pad_geom_id in contacts
    has_moving = moving_pad_geom_id in contacts
    # Debug: print contact geom names
    if contacts:
        contact_names = [model.geom(g).name for g in contacts]
        print(f"   DEBUG contacts: {contact_names}, static={has_static}, moving={has_moving}")
    return has_static and has_moving


def move_to_position(ik, target_pos, gripper_action, steps, viewer=None, locked_joints=None):
    """Move graspframe directly to target position over N steps."""
    for step in range(steps):
        ctrl = ik.step_toward_target(
            target_pos,
            gripper_action=gripper_action,
            gain=0.5,
            locked_joints=locked_joints
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)  # green = gripperframe (fingertips)
            add_marker_sphere(viewer.user_scn, target_pos, [1, 1, 0, 1], size=0.008)  # yellow = target
            viewer.sync()

    # Return final error (graspframe to target)
    graspframe_pos = ik.get_ee_position()
    return np.linalg.norm(target_pos - graspframe_pos)


def pick_up_lid(lid_pos, viewer=None):
    """
    Pick up petri dish lid using clean top-down approach.

    Adapted from ECE4560 SO-101 assignment.
    """
    height_offset = 0.01  # 10mm above lid
    gripper_open = 0.3   # partially open (ECE4560 uses 50/100, not fully open)
    gripper_closed = -0.8  # tighter grip

    # Lid is thin, position fingertips slightly above lid surface to avoid pushing through base
    grasp_z_offset = 0.003  # 3mm above lid top to grasp rim properly

    # With wrist_roll=90°, fingers spread along Y axis
    # Offset to center grip on lid (static finger is offset from gripperframe)
    lid_radius = 0.0185  # 37mm diameter lid
    finger_width_offset = -0.015  # offset along Y to center grip

    # Lock wrist joints for top-down approach
    # wrist_flex (3) pointing down, wrist_roll (4) for finger orientation
    locked_joints = [3, 4]

    # Initialize arm in top-down configuration
    mujoco.mj_resetData(model, data)

    # Set lid position
    data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = lid_pos
    data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]  # Upside down

    # Top-down arm configuration
    data.qpos[3] = np.pi/2  # wrist_flex pointing down (90 degrees)
    data.qpos[4] = np.pi/2  # wrist_roll for horizontal fingers
    data.ctrl[3] = np.pi/2
    data.ctrl[4] = np.pi/2
    mujoco.mj_forward(model, data)

    # Use gripperframe (at fingertips) not graspframe (6cm behind)
    ik = IKController(model, data, end_effector_site="gripperframe")

    # Let lid settle
    for _ in range(50):
        mujoco.mj_step(model, data)

    actual_lid_pos = data.qpos[lid_qpos_addr:lid_qpos_addr+3].copy()
    print(f"\n=== Top-Down Pick ===")
    print(f"Lid position: {actual_lid_pos}")

    # Step 1: Move above lid with gripper open
    above_pos = actual_lid_pos.copy()
    above_pos[2] += grasp_z_offset + height_offset
    above_pos[1] += finger_width_offset  # Y offset to center grip
    print(f"\n1. Moving above lid...")
    error = move_to_position(ik, above_pos, gripper_open, steps=300, viewer=viewer, locked_joints=locked_joints)
    print(f"   Position error: {error:.4f}")

    # Step 2: Move down to lid with gripper open
    grasp_target = actual_lid_pos.copy()
    grasp_target[2] += grasp_z_offset
    grasp_target[1] += finger_width_offset  # Y offset to center grip
    print(f"\n2. Moving down to lid...")
    error = move_to_position(ik, grasp_target, gripper_open, steps=200, viewer=viewer, locked_joints=locked_joints)
    print(f"   Position error: {error:.4f}")

    # Step 3: Close gripper with contact detection
    print(f"\n3. Closing gripper...")
    contact_step = None
    contact_action = None
    tighten_amount = 0.4  # How much to tighten after contact
    grasp_action = gripper_closed

    for step in range(300):
        if contact_step is None:
            # Close gradually until contact
            t = min(step / 250, 1.0)
            gripper = gripper_open - 2.0 * t  # 1.0 to -1.0 over 250 steps
        else:
            # After contact, slowly tighten
            steps_since = step - contact_step
            t_slow = min(steps_since / 100, 1.0)
            target_action = max(contact_action - tighten_amount, -1.0)
            gripper = contact_action + (target_action - contact_action) * t_slow

        ctrl = ik.step_toward_target(grasp_target, gripper_action=gripper, gain=0.1, locked_joints=locked_joints)
        ctrl[3] = np.pi/2  # Force wrist_flex vertical
        ctrl[4] = np.pi/2  # Force wrist_roll
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)  # green = gripperframe (fingertips)
            viewer.sync()

        if step % 50 == 0:
            print(f"   step {step}: gripper={gripper:.2f}, grasping={is_grasping()}")

        # Detect contact
        if is_grasping() and contact_step is None:
            contact_step = step
            contact_action = gripper
            print(f"   Contact at step {step}, gripper={gripper:.2f}")

        # Check if done tightening
        if contact_step is not None:
            target_action = max(contact_action - tighten_amount, -1.0)
            if gripper <= target_action + 0.01:
                grasp_action = gripper
                print(f"   Tightened to {grasp_action:.2f}")
                break
    else:
        if contact_step is not None:
            grasp_action = gripper

    print(f"   Grasping: {is_grasping()}")

    # Step 4: Lift up
    print(f"\n4. Lifting...")
    lift_pos = actual_lid_pos.copy()
    lift_pos[2] += height_offset + 0.05  # lift higher

    for step in range(300):
        t = min(step / 200, 1.0)
        target = grasp_target + (lift_pos - grasp_target) * t

        ctrl = ik.step_toward_target(target, gripper_action=grasp_action, gain=0.3, locked_joints=locked_joints)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)  # green = gripperframe (fingertips)
            add_marker_sphere(viewer.user_scn, target, [1, 1, 0, 1], size=0.008)  # yellow = target
            viewer.sync()

        if step % 100 == 0:
            lid_z = data.qpos[lid_qpos_addr + 2]
            print(f"   step {step}: lid_z={lid_z:.3f}, grasping={is_grasping()}")

    # Hold at lift position
    print(f"\n5. Holding...")
    for step in range(200):
        ctrl = ik.step_toward_target(lift_pos, gripper_action=grasp_action, gain=0.3, locked_joints=locked_joints)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)  # green = gripperframe (fingertips)
            viewer.sync()

        if step % 100 == 0:
            lid_z = data.qpos[lid_qpos_addr + 2]
            print(f"   step {step}: lid_z={lid_z:.3f}, grasping={is_grasping()}")

    final_z = data.qpos[lid_qpos_addr + 2]
    success = final_z > actual_lid_pos[2] + 0.03
    print(f"\nFinal lid Z: {final_z:.4f}")
    print(f"Result: {'SUCCESS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    use_viewer = "--viewer" in sys.argv

    # Lid on dish
    lid_pos = np.array([0.25, 0.0, 0.0155])

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            pick_up_lid(lid_pos, viewer=viewer)
            print("\nViewer open. Ctrl+C to exit.")
            while viewer.is_running():
                time.sleep(0.1)
    else:
        print("Testing top-down pick:")
        pick_up_lid(lid_pos)
