"""Top-down lid pick where the dish base is also a freejoint."""
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

scene_path = Path("models/so101/lift_petri_lid_free_dish.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

lid_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "petri_lid_joint")
lid_qpos_addr = model.jnt_qposadr[lid_joint_id]
dish_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "petri_dish_joint")
dish_qpos_addr = model.jnt_qposadr[dish_joint_id]
lid_top_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_top")
lid_skirt_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_skirt")
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
        np.array(rgba, dtype=np.float32),
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
    return has_static and has_moving


def move_to_position(ik, target_pos, gripper_action, steps, viewer=None, locked_joints=None):
    """Move graspframe directly to target position over N steps."""
    for _ in range(steps):
        ctrl = ik.step_toward_target(
            target_pos,
            gripper_action=gripper_action,
            gain=0.5,
            locked_joints=locked_joints,
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)
            add_marker_sphere(viewer.user_scn, target_pos, [1, 1, 0, 1], size=0.008)
            viewer.sync()

    graspframe_pos = ik.get_ee_position()
    return np.linalg.norm(target_pos - graspframe_pos)


def pick_up_lid(lid_pos, viewer=None):
    """Pick up petri dish lid with a free-floating dish base."""
    height_offset = 0.03
    gripper_open = -0.2
    gripper_closed = -0.3

    grasp_z_offset = 0.002
    finger_width_offset = -0.015
    locked_joints = [3, 4]

    mujoco.mj_resetData(model, data)

    # Set dish position (freejoint) so it starts on the table
    data.qpos[dish_qpos_addr : dish_qpos_addr + 3] = [0.25, 0.0, 0.001]
    data.qpos[dish_qpos_addr + 3 : dish_qpos_addr + 7] = [1, 0, 0, 0]

    # Set lid position
    data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = lid_pos
    data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]

    data.qpos[3] = np.pi / 2
    data.qpos[4] = np.pi / 2
    data.ctrl[3] = np.pi / 2
    data.ctrl[4] = np.pi / 2
    mujoco.mj_forward(model, data)

    ik = IKController(model, data, end_effector_site="gripperframe")

    for _ in range(50):
        mujoco.mj_step(model, data)

    actual_lid_pos = data.qpos[lid_qpos_addr : lid_qpos_addr + 3].copy()
    print("\n=== Top-Down Pick (Free Dish) ===")
    print(f"Lid position: {actual_lid_pos}")

    above_pos = actual_lid_pos.copy()
    above_pos[2] += grasp_z_offset + height_offset
    above_pos[1] += finger_width_offset
    print("\n1. Moving above lid...")
    error = move_to_position(ik, above_pos, gripper_open, steps=300, viewer=viewer, locked_joints=locked_joints)
    print(f"   Position error: {error:.4f}")

    grasp_target = actual_lid_pos.copy()
    grasp_target[2] += grasp_z_offset
    grasp_target[1] += finger_width_offset
    print("\n2. Moving down to lid...")
    error = move_to_position(ik, grasp_target, gripper_open, steps=200, viewer=viewer, locked_joints=locked_joints)
    print(f"   Position error: {error:.4f}")

    print("\n3. Closing gripper...")
    contact_step = None
    contact_action = None
    tighten_amount = 0.05
    grasp_action = gripper_closed

    for step in range(300):
        if contact_step is None:
            t = min(step / 800, 1.0)
            gripper = gripper_open - 2.0 * t
        else:
            steps_since = step - contact_step
            t_slow = min(steps_since / 100, 1.0)
            target_action = max(contact_action - tighten_amount, -1.0)
            gripper = contact_action + (target_action - contact_action) * t_slow

        ctrl = ik.step_toward_target(
            grasp_target,
            gripper_action=gripper,
            gain=0.05,
            locked_joints=locked_joints,
        )
        ctrl[3] = np.pi / 2
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)
            viewer.sync()

        if step % 50 == 0:
            print(f"   step {step}: gripper={gripper:.2f}, grasping={is_grasping()}")

        if is_grasping() and contact_step is None:
            contact_step = step
            contact_action = gripper
            print(f"   Contact at step {step}, gripper={gripper:.2f}")

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

    print("\n4. Lifting...")
    lift_pos = actual_lid_pos.copy()
    lift_pos[2] += height_offset + 0.05

    for step in range(300):
        t = min(step / 200, 1.0)
        target = grasp_target + (lift_pos - grasp_target) * t

        ctrl = ik.step_toward_target(
            target,
            gripper_action=grasp_action,
            gain=0.3,
            locked_joints=locked_joints,
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)
            add_marker_sphere(viewer.user_scn, target, [1, 1, 0, 1], size=0.008)
            viewer.sync()

        if step % 100 == 0:
            lid_z = data.qpos[lid_qpos_addr + 2]
            print(f"   step {step}: lid_z={lid_z:.3f}, grasping={is_grasping()}")

    print("\n5. Holding...")
    for step in range(200):
        ctrl = ik.step_toward_target(
            lift_pos,
            gripper_action=grasp_action,
            gain=0.3,
            locked_joints=locked_joints,
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if viewer:
            viewer.user_scn.ngeom = 0
            gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
            add_marker_sphere(viewer.user_scn, gripperframe_pos, [0, 1, 0, 1], size=0.01)
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

    lid_pos = np.array([0.25, 0.0, 0.0155])

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            pick_up_lid(lid_pos, viewer=viewer)
            print("\nViewer open. Ctrl+C to exit.")
            while viewer.is_running():
                time.sleep(0.1)
    else:
        print("Testing top-down pick with free dish base:")
        pick_up_lid(lid_pos)
