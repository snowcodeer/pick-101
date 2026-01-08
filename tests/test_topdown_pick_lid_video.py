"""Top-down petri lid pick video capture with wide angle view.

Records the IK-based top-down pick motion for devlog.
Adapted from cube video script to the petri lid scene.
"""
import sys
from pathlib import Path

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import numpy as np
import imageio
from datetime import datetime
import argparse
from src.controllers.ik_controller import IKController

scene_path = Path("models/so101/lift_petri_lid.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

# Lid
lid_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "petri_lid_joint")
lid_qpos_addr = model.jnt_qposadr[lid_joint_id]
lid_top_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_top")
lid_skirt_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lid_skirt")

# Gripper
gripperframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
static_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
moving_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")


def setup_camera(angle="wide"):
    """Setup camera for different viewing angles."""
    cam = mujoco.MjvCamera()

    if angle == "wide":
        # Wide angle - shows entire robot and workspace
        cam.lookat[:] = [0.20, 0.0, 0.08]
        cam.distance = 0.7
        cam.azimuth = 145
        cam.elevation = -25
    elif angle == "close":
        # Close-up on gripper and lid
        cam.lookat[:] = [0.25, 0.0, 0.04]
        cam.distance = 0.35
        cam.azimuth = 135
        cam.elevation = -20
    elif angle == "top":
        # Top-down view
        cam.lookat[:] = [0.25, 0.0, 0.04]
        cam.distance = 0.5
        cam.azimuth = 180
        cam.elevation = -60
    else:
        raise ValueError(f"Unknown angle: {angle}")

    return cam


def get_contacts_with_lid():
    """Return list of 'other' geoms in contacts involving lid_top or lid_skirt."""
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == lid_top_geom_id or g1 == lid_skirt_geom_id:
            contacts.append(g2)
        elif g2 == lid_top_geom_id or g2 == lid_skirt_geom_id:
            contacts.append(g1)
    return contacts


def is_grasping():
    """Strict pinch definition: both pads contacting lid (may be transient for thin objects)."""
    contacts = get_contacts_with_lid()
    has_static = static_pad_geom_id in contacts
    has_moving = moving_pad_geom_id in contacts
    return has_static and has_moving


def record_topdown_pick_lid(lid_pos, renderer, cam, frame_interval=5, scene_option=None):
    """Record top-down lid pick sequence, returns frames + success."""
    frames = []

    # --- Tuned params from your working script ---
    height_offset = 0.03
    gripper_open = -0.2
    gripper_closed = -0.3

    grasp_z_offset = 0.002
    finger_width_offset = -0.015
    locked_joints = [3, 4]

    # Initialize
    mujoco.mj_resetData(model, data)

    # Place lid and flip upside down (as in your script)
    data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = lid_pos
    data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]

    # Top-down wrist configuration
    data.qpos[3] = np.pi / 2
    data.qpos[4] = np.pi / 2
    data.ctrl[3] = np.pi / 2
    data.ctrl[4] = np.pi / 2
    mujoco.mj_forward(model, data)

    ik = IKController(model, data, end_effector_site="gripperframe")

    # Settle
    for _ in range(50):
        mujoco.mj_step(model, data)

    actual_lid_pos = data.qpos[lid_qpos_addr : lid_qpos_addr + 3].copy()

    def capture():
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        return renderer.render().copy()

    def run_steps(target_pos, gripper_action, steps, gain=0.5):
        for step in range(steps):
            ctrl = ik.step_toward_target(
                target_pos,
                gripper_action=gripper_action,
                gain=gain,
                locked_joints=locked_joints,
            )
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if step % frame_interval == 0:
                frames.append(capture())

    # Initial frame
    frames.append(capture())
    print(f"Lid at: {actual_lid_pos}")

    # Step 1: Move above lid
    print("1. Moving above...")
    above_pos = actual_lid_pos.copy()
    above_pos[2] += grasp_z_offset + height_offset
    above_pos[1] += finger_width_offset
    run_steps(above_pos, gripper_open, steps=300, gain=0.5)

    # Step 2: Move down
    print("2. Moving down...")
    grasp_target = actual_lid_pos.copy()
    grasp_target[2] += grasp_z_offset
    grasp_target[1] += finger_width_offset
    run_steps(grasp_target, gripper_open, steps=200, gain=0.5)

    # Step 3: Close gripper (your slow close + gentle gain)
    print("3. Closing gripper...")
    contact_step = None
    contact_action = None
    tighten_amount = 0.05
    grasp_action = gripper_closed

    for step in range(300):
        if contact_step is None:
            # Slow close (matches your script)
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
        # Keep wrist locked
        ctrl[3] = np.pi / 2
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            frames.append(capture())

        if is_grasping() and contact_step is None:
            contact_step = step
            contact_action = gripper
            print(f"   Pinch-contact (both pads) at step {step}")

        if contact_step is not None:
            target_action = max(contact_action - tighten_amount, -1.0)
            if gripper <= target_action + 0.01:
                grasp_action = gripper
                break
    else:
        if contact_step is not None:
            grasp_action = gripper

    # Step 4: Lift
    print("4. Lifting...")
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

        if step % frame_interval == 0:
            frames.append(capture())

    # Step 5: Hold
    print("5. Holding...")
    for step in range(150):
        ctrl = ik.step_toward_target(
            lift_pos,
            gripper_action=grasp_action,
            gain=0.3,
            locked_joints=locked_joints,
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            frames.append(capture())

    final_z = data.qpos[lid_qpos_addr + 2]
    success = final_z > actual_lid_pos[2] + 0.03
    print(f"\nFinal lid Z: {final_z:.4f}")
    print(f"Result: {'SUCCESS' if success else 'FAIL'}")

    return frames, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", choices=["wide", "close", "top"], default="wide")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hide-sites", action="store_true", help="Hide site markers")
    args = parser.parse_args()

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = setup_camera(args.angle)

    scene_option = None
    if args.hide_sites:
        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[:] = 0  # hide all sites

    lid_pos = np.array([0.25, 0.0, 0.0155])

    print(f"=== Top-Down Lid Pick Video ({args.angle} angle) ===")
    frames, success = record_topdown_pick_lid(
        lid_pos,
        renderer,
        cam,
        frame_interval=5,
        scene_option=scene_option,
    )

    # Output path
    if args.output:
        video_path = Path(args.output)
    else:
        output_dir = Path("devlogs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"topdown_pick_lid_{args.angle}_{timestamp}.mp4"

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(video_path), frames, fps=args.fps)
    print(f"\nSaved {len(frames)} frames to {video_path}")

    renderer.close()


if __name__ == "__main__":
    main()