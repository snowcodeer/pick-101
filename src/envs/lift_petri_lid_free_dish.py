"""Lift petri dish lid environment with Cartesian (end-effector) action space and free dish base.

Adapted from lift_cube.py for petri dish lid manipulation.
Key differences:
- Lid is much thinner (1.25mm vs 30mm cube)
- Lid is lighter (5g vs 30g)
- Lid diameter: 37mm (radius 18.5mm)
- Two collision geoms: top disk (0.25mm) + skirt (1mm)
- Tighter gripper threshold (0.20 vs 0.25) for delicate manipulation
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.controllers.ik_controller import IKController


class LiftPetriLidFreeDishCartesianEnv(gym.Env):
    """Lift petri dish lid with Cartesian action space.

    Action space (4 dims):
        - Delta X, Y, Z for end-effector position
        - Gripper open/close (-1 to 1)

    Observation space (21 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Gripper orientation (3) - euler angles
        - Lid position (3)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.02,  # 2cm per step max
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "dense",
        reward_version: str = "v19",
        curriculum_stage: int = 0,  # Initial condition: 0=full task, 1=lid held lifted, 2=lid held on table, 3=gripper near lid, 4=gripper far
        lock_wrist: bool = False,  # Lock wrist joints for stable grasping
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.reward_type = reward_type
        self.reward_version = reward_version
        self.curriculum_stage = curriculum_stage
        self.lock_wrist = lock_wrist
        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._grasp_streak = 0
        self._reset_gripper_action = None
        self._prev_action = np.zeros(4)
        self._open_gripper_count = 0
        self._dish_start_pos = None
        self._top_contact_count = 0  # Track consecutive steps with top contact but no grasp
        self._has_rewarded_skirt_contact = False  # One-time skirt contact bonus
        self._has_rewarded_closing = False  # One-time closing after contact bonus

        # Load model
        scene_path = Path(__file__).parent.parent.parent / "models/so101/lift_petri_lid_free_dish.xml"
        print(f"[env] Scene: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Get finger pad geom IDs for contact detection
        self._static_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self._moving_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")

        # Get lid geom IDs (two geoms: top disk + skirt)
        self._lid_top_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lid_top")
        self._lid_skirt_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lid_skirt")

        # Dish joint (freejoint)
        self._dish_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "petri_dish_joint")
        self._dish_qpos_addr = self.model.jnt_qposadr[self._dish_joint_id]
        # Initialize IK controller
        self.ik = IKController(self.model, self.data, end_effector_site="gripperframe")

        # Joint info
        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space
        obs_dim = 6 + 6 + 3 + 3 + 3  # joints pos/vel + gripper pos + gripper euler + lid pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Track target EE position
        self._target_ee_pos = None

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()
        gripper_pos = self.ik.get_ee_position()

        # Get gripper orientation as euler angles
        gripper_mat = self.ik.get_ee_orientation()
        gripper_euler = self._rotation_matrix_to_euler(gripper_mat)

        lid_pos = self.data.sensor("lid_pos").data.copy()

        return np.concatenate([joint_pos, joint_vel, gripper_pos, gripper_euler, lid_pos]).astype(np.float32)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _set_dish_pose(self, pos: np.ndarray | None = None):
        if pos is None:
            pos = np.array([0.25, 0.0, 0.001], dtype=np.float64)
        self.data.qpos[self._dish_qpos_addr : self._dish_qpos_addr + 3] = pos
        self.data.qpos[self._dish_qpos_addr + 3 : self._dish_qpos_addr + 7] = [1, 0, 0, 0]

    def _check_lid_contacts(self) -> tuple[bool, bool]:
        """Check if lid contacts static finger pad and moving finger pad.

        Checks both lid_top and lid_skirt geoms for contact.
        """
        has_static_contact = False
        has_moving_contact = False

        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2

            other_geom = None
            if geom1 == self._lid_top_geom_id or geom1 == self._lid_skirt_geom_id:
                other_geom = geom2
            elif geom2 == self._lid_top_geom_id or geom2 == self._lid_skirt_geom_id:
                other_geom = geom1

            if other_geom is not None:
                if other_geom == self._static_pad_geom_id:
                    has_static_contact = True
                if other_geom == self._moving_pad_geom_id:
                    has_moving_contact = True

        return has_static_contact, has_moving_contact

    def _get_static_finger_pos(self) -> np.ndarray:
        """Get position of static finger pad centroid."""
        return self.data.geom_xpos[self._static_pad_geom_id].copy()

    def _get_moving_finger_pos(self) -> np.ndarray:
        """Get position of moving finger pad centroid."""
        return self.data.geom_xpos[self._moving_pad_geom_id].copy()

    def _is_grasping(self) -> bool:
        """Check if lid is properly grasped (pinched between gripper and jaw).

        Uses tighter threshold (0.20 vs 0.25) for thinner lid.
        """
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.20  # Tighter for thin lid

        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        is_pinched = has_gripper_contact and has_jaw_contact

        return is_closed and is_pinched

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        lid_pos = self.data.sensor("lid_pos").data.copy()
        dish_pos = self.data.qpos[self._dish_qpos_addr : self._dish_qpos_addr + 3].copy()

        gripper_to_lid = np.linalg.norm(gripper_pos - lid_pos)
        lid_z = lid_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and lid_z >= self.lift_height

        # Check dish hasn't moved significantly (threshold: 1cm displacement)
        dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos) if self._dish_start_pos is not None else 0.0
        dish_stable = dish_disp < 0.01  # Dish should stay within 1cm of start position

        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        top_contact = False
        skirt_contact = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 == self._lid_top_geom_id or g2 == self._lid_top_geom_id:
                top_contact = True
            if g1 == self._lid_skirt_geom_id or g2 == self._lid_skirt_geom_id:
                skirt_contact = True

        return {
            "gripper_to_lid": gripper_to_lid,
            "lid_pos": lid_pos.copy(),
            "lid_z": lid_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "dish_pos": dish_pos.copy(),
            "dish_disp": dish_disp,
            "dish_stable": dish_stable,
            "has_gripper_contact": has_gripper_contact,
            "has_jaw_contact": has_jaw_contact,
            "has_top_contact": top_contact,
            "has_skirt_contact": skirt_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "hold_count": self._hold_count,
            "grasp_streak": self._grasp_streak,
            "is_success": self._hold_count >= self.hold_steps and dish_stable,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Get lid joint info
        lid_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "petri_lid_joint"
        )
        lid_qpos_addr = self.model.jnt_qposadr[lid_joint_id]

        if self.curriculum_stage == 0:
            # Initial condition 0: Full task - lid on dish, arm at home position
            if self.np_random is not None:
                # Randomize dish+lid together
                dish_x = 0.30 + self.np_random.uniform(-0.03, 0.03)
                dish_y = 0.0 + self.np_random.uniform(-0.03, 0.03)
                self._set_dish_pose(pos=np.array([dish_x, dish_y, 0.001]))

                # Place lid on top of dish
                lid_x, lid_y = dish_x, dish_y
                self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = [lid_x, lid_y, 0.0155]
                self.data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]  # Upside down (fits on dish)

        elif self.curriculum_stage == 1:
            # Initial condition 1: Lid in closed gripper at lift height (easiest - just hold)
            self._reset_with_lid_in_gripper(lid_qpos_addr, lift_height=self.lift_height)

        elif self.curriculum_stage == 2:
            # Initial condition 2: Lid in closed gripper at table level (need to lift)
            self._reset_with_lid_in_gripper(lid_qpos_addr, lift_height=0.03)

        elif self.curriculum_stage == 3:
            # Initial condition 3: Gripper near lid, open (need to close and lift)
            # Used by grasp_stage1, grasp_stage2, grasp_stage3 training configs
            self._reset_gripper_near_lid(lid_qpos_addr)

        elif self.curriculum_stage == 4:
            # Initial condition 4: Gripper far from lid (need to reach, descend, grasp, lift)
            self._reset_gripper_far_from_lid(lid_qpos_addr)

        mujoco.mj_forward(self.model, self.data)

        # Initialize target EE position to current position
        self._target_ee_pos = self.ik.get_ee_position().copy()

        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._grasp_streak = 0
        self._prev_action = np.zeros(4)
        self._open_gripper_count = 0
        self._top_contact_count = 0
        self._has_rewarded_skirt_contact = False
        self._has_rewarded_closing = False
        self._dish_start_pos = self.data.qpos[self._dish_qpos_addr : self._dish_qpos_addr + 3].copy()

        return self._get_obs(), self._get_info()

    def _reset_with_lid_in_gripper(self, lid_qpos_addr: int, lift_height: float):
        """Reset with lid grasped in gripper at specified height."""
        # Constants from test_topdown_pick_lid_free_dish.py (proven to work for 1.25mm lid)
        height_offset = 0.03
        gripper_open = -0.2  # Already somewhat closed (not wide open)
        gripper_closed = -0.3  # Gentle close for thin lid
        grasp_z_offset = 0.002  # 2mm above lid center (not 5mm)
        finger_width_offset = -0.015
        locked_joints = [3, 4]

        # Randomize dish+lid position together
        if self.np_random is not None:
            dish_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            dish_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            dish_x, dish_y = 0.25, 0.0

        # Place dish at randomized position
        self._set_dish_pose(pos=np.array([dish_x, dish_y, 0.001]))

        # Place lid on top of dish (same XY)
        lid_x, lid_y = dish_x, dish_y
        lid_z = 0.0155  # Resting on dish

        # Place lid on dish
        self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = [lid_x, lid_y, lid_z]
        self.data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]  # Upside down

        # Top-down arm configuration
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let lid settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual lid position after settle
        actual_lid_pos = self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3].copy()

        # Step 1: Move above lid with gripper open
        above_pos = actual_lid_pos.copy()
        above_pos[2] += grasp_z_offset + height_offset
        above_pos[1] += finger_width_offset
        for _ in range(300):
            ctrl = self.ik.step_toward_target(above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        # Step 2: Move down to lid with gripper open
        grasp_target = actual_lid_pos.copy()
        grasp_target[2] += grasp_z_offset
        grasp_target[1] += finger_width_offset
        for _ in range(200):
            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        # Step 3: Close gripper with contact detection
        contact_step = None
        contact_action = None
        tighten_amount = 0.02  # Very gentle tightening - less than test script to avoid gripping base
        grasp_action = gripper_closed

        for step in range(300):
            if contact_step is None:
                t = min(step / 800, 1.0)  # Slow closing - matches test script
                gripper = gripper_open - 2.0 * t
            else:
                steps_since = step - contact_step
                t_slow = min(steps_since / 100, 1.0)
                target_action = max(contact_action - tighten_amount, -1.0)
                gripper = contact_action + (target_action - contact_action) * t_slow

            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper, gain=0.05, locked_joints=locked_joints)
            ctrl[3] = np.pi / 2
            ctrl[4] = np.pi / 2
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

            # Detect contact using is_grasping logic
            if self._is_grasping() and contact_step is None:
                contact_step = step
                contact_action = gripper

            # Check if done tightening
            if contact_step is not None:
                target_action = max(contact_action - tighten_amount, -1.0)
                if gripper <= target_action + 0.01:
                    grasp_action = gripper
                    break

        # Handle case where loop finished without breaking
        if contact_step is not None and gripper > max(contact_action - tighten_amount, -1.0) + 0.01:
            grasp_action = gripper

        # Step 4: Lift to target height (if specified)
        if lift_height > 0.03:  # Only lift if target is above table level
            lift_target = grasp_target.copy()
            lift_target[2] = lift_height + 0.003  # Lift 3mm above target to ensure threshold is met
            for _ in range(400):  # Slow, steady lift
                ctrl = self.ik.step_toward_target(lift_target, gripper_action=grasp_action, gain=0.3, locked_joints=locked_joints)
                ctrl[3] = np.pi / 2
                ctrl[4] = np.pi / 2
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)

        # Store the gripper action for curriculum learning
        self._reset_gripper_action = grasp_action

    def _reset_gripper_near_lid(self, lid_qpos_addr: int):
        """Reset with gripper positioned above lid, open, ready to grasp."""
        # Constants from test_topdown_pick_lid_free_dish.py (proven to work for 1.25mm lid)
        height_offset = 0.03
        gripper_open = -0.2  # Already somewhat closed (not wide open)
        grasp_z_offset = 0.002  # 2mm above lid center (not 5mm)
        finger_width_offset = -0.015

        # Randomize dish+lid position together (they should be aligned)
        if self.np_random is not None:
            dish_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            dish_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            dish_x, dish_y = 0.25, 0.0

        # Place dish at randomized position
        self._set_dish_pose(pos=np.array([dish_x, dish_y, 0.001]))

        # Place lid on top of dish (same XY position)
        lid_x, lid_y = dish_x, dish_y
        lid_z = 0.0155

        # Place lid on dish
        self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = [lid_x, lid_y, lid_z]
        self.data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]

        # Top-down arm configuration
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let lid settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual lid position after settle
        actual_lid_pos = self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3].copy()

        # Position gripper above lid
        above_pos = actual_lid_pos.copy()
        above_pos[1] += finger_width_offset
        above_pos[2] = actual_lid_pos[2] + grasp_z_offset + height_offset

        # Move to above position with gripper open
        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(100):
            ctrl = self.ik.step_toward_target(
                above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def _reset_gripper_far_from_lid(self, lid_qpos_addr: int):
        """Reset with gripper positioned further from lid - must reach first."""
        # From test_topdown_pick_lid_free_dish.py (proven to work for 1.25mm lid)
        gripper_open = -0.2  # Already somewhat closed (not wide open)

        # Randomize dish+lid position together
        if self.np_random is not None:
            dish_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            dish_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            dish_x, dish_y = 0.25, 0.0

        # Place dish at randomized position
        self._set_dish_pose(pos=np.array([dish_x, dish_y, 0.001]))

        # Place lid on top of dish (same XY)
        lid_x, lid_y = dish_x, dish_y
        lid_z = 0.0155

        # Place lid on dish
        self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3] = [lid_x, lid_y, lid_z]
        self.data.qpos[lid_qpos_addr + 3 : lid_qpos_addr + 7] = [0, 1, 0, 0]

        # Top-down arm configuration
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let lid settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Read actual lid position after settle
        actual_lid_pos = self.data.qpos[lid_qpos_addr : lid_qpos_addr + 3].copy()

        # Position gripper at a distance from lid
        if self.np_random is not None:
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(0.08, 0.12)
            offset_x = distance * np.cos(angle)
            offset_y = distance * np.sin(angle)
        else:
            offset_x, offset_y = 0.10, 0.0

        start_pos = np.array([
            actual_lid_pos[0] + offset_x,
            actual_lid_pos[1] + offset_y,
            self.lift_height + 0.02
        ])

        # Clamp to workspace bounds
        start_pos[0] = np.clip(start_pos[0], 0.15, 0.45)
        start_pos[1] = np.clip(start_pos[1], -0.25, 0.25)

        # Move to start position with gripper open
        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(150):
            ctrl = self.ik.step_toward_target(
                start_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Parse action
        action = np.clip(action, -1.0, 1.0)
        delta_xyz = action[:3] * self.action_scale
        gripper_action = action[3]

        # Update target end-effector position
        self._target_ee_pos += delta_xyz

        # Clamp to workspace bounds
        self._target_ee_pos[0] = np.clip(self._target_ee_pos[0], 0.1, 0.5)
        self._target_ee_pos[1] = np.clip(self._target_ee_pos[1], -0.3, 0.3)
        self._target_ee_pos[2] = np.clip(self._target_ee_pos[2], 0.01, 0.4)

        gripper_target = gripper_action
        if self.lock_wrist and self._reset_gripper_action is not None:
            gripper_target = self._reset_gripper_action

        # Use IK to compute joint controls
        if self.lock_wrist:
            ctrl = self.ik.step_toward_target(
                self._target_ee_pos,
                gripper_action=gripper_target,
                gain=0.5,
                locked_joints=[4],
            )
            ctrl[4] = np.pi / 2  # wrist_roll horizontal
        else:
            ctrl = self.ik.step_toward_target(
                self._target_ee_pos,
                gripper_action=gripper_target,
                gain=0.5,
            )

        # Apply control and step simulation
        self.data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get state
        obs = self._get_obs()
        info = self._get_info()

        # Update hold counter
        if info["is_lifted"]:
            self._hold_count += 1
        else:
            self._hold_count = 0
        info["hold_count"] = self._hold_count

        # Update grasp streak (consecutive steps grasping)
        if info["is_grasping"]:
            self._grasp_streak += 1
        else:
            self._grasp_streak = 0
        info["grasp_streak"] = self._grasp_streak

        # Update top contact count (consecutive steps with top contact but no grasp)
        if info["has_top_contact"] and not info["is_grasping"]:
            self._top_contact_count += 1
        else:
            self._top_contact_count = 0
        info["top_contact_count"] = self._top_contact_count

        # Check success
        is_success = self._hold_count >= self.hold_steps
        info["is_success"] = is_success

        # Compute reward
        reward = self._compute_reward(info, was_grasping=self._was_grasping, action=action)

        # Update previous grasp state and action
        self._was_grasping = info["is_grasping"]
        self._prev_action = action.copy()

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        if self.reward_type == "sparse":
            return 0.0 if info["is_success"] else -1.0

        # Dispatch to version-specific reward function
        reward_fn = getattr(self, f"_reward_{self.reward_version}", None)
        if reward_fn is None:
            raise ValueError(f"Unknown reward version: {self.reward_version}")
        return reward_fn(info, was_grasping=was_grasping, action=action)

    def _reward_v19(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V19: Dual reach + strong grasp + lift gradient + hold bonus.

        Best performing reward for cube lifting - adapted for petri lid.
        """
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        is_closed = gripper_state < 0.20  # Tighter threshold for thin lid
        dish_pos = info["dish_pos"]

        # Standard gripper reach
        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)

        # Moving finger reach - only applies when gripper is close to lid
        reach_threshold = 0.7
        if gripper_reach < reach_threshold:
            reach_reward = gripper_reach
        else:
            # Always check actual moving finger distance (no shortcut for closing)
            moving_finger_pos = self._get_moving_finger_pos()
            moving_to_lid = np.linalg.norm(moving_finger_pos - lid_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_lid)

            reach_reward = (gripper_reach + moving_reach) * 0.5

        reward += reach_reward

        # Height alignment reward - encourage descending to correct grasp height
        grasp_z_offset = 0.005  # Target: 5mm above lid center
        target_grasp_z = lid_z + grasp_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)

        # Height alignment reward - always reward being at correct height
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            reward += height_alignment * 0.5  # Always reward correct height (stacks with everything)

        # Contact rewards - ALWAYS reward contacts (especially during closing!)
        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        if gripper_reach > 0.7 and z_error < 0.01:  # At correct height
            if has_gripper_contact or has_jaw_contact:
                reward += 1.0  # Strong reward for any contact
            if has_gripper_contact and has_jaw_contact:
                reward += 1.5  # Very strong reward for bilateral contact

        # Closing incentive - reward closing action when at correct height
        if gripper_reach > 0.7 and z_error < 0.01 and action is not None:
            # Negative gripper action = closing (contacts will form during closing)
            if action[3] < -0.3:  # Actively closing
                reward += 1.0  # Strong incentive to close at correct height

        # Push-down penalty
        if lid_z < 0.01:
            push_penalty = (0.01 - lid_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus - stacks with contact rewards (contacts persist during grasp)
        if is_grasping:
            reward += 5.0  # Strong bonus on top of contact rewards (total ~8.7)

            # Gentle grasp bonus (target ~ -0.30 for thin lid)
            gentle_target = -0.30
            gentle_band = 0.10  # reward within [-0.40, -0.20]
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.6
            else:
                reward -= (gentle_delta - gentle_band) * 1.0

            # Continuous lift reward
            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)  # 0.0155 = lid resting height
            reward += lift_progress * 4.0

            # Binary lift bonus at 0.02m
            if lid_z > 0.02:
                reward += 1.0

            # Linear threshold ramp from 0.04m to lift_height
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 2.0

        # Target height bonus
        if lid_z > self.lift_height:
            reward += 1.0

            # Hold count bonus - escalating reward for sustained height
            reward += 0.5 * hold_count

        # Penalize pushing the dish base (freejoint drift)
        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            reward -= dish_disp * 3.0  # Reduced from 10.0 - was too harsh, prevented learning

        # Gripper jitter penalty - penalize rapid gripper changes when near grasp
        if action is not None and self._prev_action is not None and gripper_reach > 0.7:
            gripper_delta = abs(action[3] - self._prev_action[3])
            reward -= gripper_delta * 1.0  # Penalize rapid gripper oscillations

        # Action rate penalty during hold phase
        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            action_penalty = 0.02 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v20(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V20: First proper reward designed for delicate petri lid grasping.

        Key features for thin (1.25mm) lid manipulation:
        - Dual reach (gripper + moving finger) for precise positioning
        - Height alignment reward to guide descent to correct grasp height
        - Contact detection rewards (bilateral grasping)
        - Closing incentive when positioned correctly
        - Stacking rewards (no cliffs) for smooth learning
        - Gripper jitter penalty for stable control
        - Direct gripper control (no rate limiting)
        - Gentle grasp pressure targeting
        """
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        is_closed = gripper_state < 0.20  # Tighter threshold for thin lid
        dish_pos = info["dish_pos"]

        # Standard gripper reach
        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)

        # Moving finger reach - only applies when gripper is close to lid
        reach_threshold = 0.7
        if gripper_reach < reach_threshold:
            reach_reward = gripper_reach
        else:
            # Always check actual moving finger distance (no shortcut for closing)
            moving_finger_pos = self._get_moving_finger_pos()
            moving_to_lid = np.linalg.norm(moving_finger_pos - lid_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_lid)

            reach_reward = (gripper_reach + moving_reach) * 0.5

        reward += reach_reward

        # Height alignment reward - always reward being at correct height
        grasp_z_offset = 0.005  # Target: 5mm above lid center
        target_grasp_z = lid_z + grasp_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)

        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            reward += height_alignment * 0.5  # Always reward correct height (stacks with everything)

        # Contact rewards - ALWAYS reward contacts (especially during closing!)
        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        if gripper_reach > 0.7 and z_error < 0.01:  # At correct height
            if has_gripper_contact or has_jaw_contact:
                reward += 1.0  # Strong reward for any contact
            if has_gripper_contact and has_jaw_contact:
                reward += 1.5  # Very strong reward for bilateral contact

        # Closing incentive - reward closing action when at correct height
        if gripper_reach > 0.7 and z_error < 0.01 and action is not None:
            # Negative gripper action = closing (contacts will form during closing)
            if action[3] < -0.3:  # Actively closing
                reward += 1.0  # Strong incentive to close at correct height

        # Push-down penalty
        if lid_z < 0.01:
            push_penalty = (0.01 - lid_z) * 50.0
            reward -= push_penalty

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus - stacks with contact rewards (contacts persist during grasp)
        if is_grasping:
            reward += 5.0  # Strong bonus on top of contact rewards (total ~9.0)

            # Gentle grasp bonus (target ~ -0.30 for thin lid)
            gentle_target = -0.30
            gentle_band = 0.10  # reward within [-0.40, -0.20]
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.6
            else:
                reward -= (gentle_delta - gentle_band) * 1.0

            # Continuous lift reward
            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)  # 0.0155 = lid resting height
            reward += lift_progress * 4.0

            # Binary lift bonus at 0.02m
            if lid_z > 0.02:
                reward += 1.0

            # Linear threshold ramp from 0.04m to lift_height
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 2.0

        # Target height bonus
        if lid_z > self.lift_height:
            reward += 1.0

            # Hold count bonus - escalating reward for sustained height
            reward += 0.5 * hold_count

        # Penalize pushing the dish base (freejoint drift)
        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            reward -= dish_disp * 3.0  # Reduced from 10.0 - was too harsh, prevented learning

        # Gripper jitter penalty - penalize rapid gripper changes when near grasp
        if action is not None and self._prev_action is not None and gripper_reach > 0.7:
            gripper_delta = abs(action[3] - self._prev_action[3])
            reward -= gripper_delta * 1.0  # Penalize rapid gripper oscillations

        # Action rate penalty during hold phase
        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            action_penalty = 0.02 * np.sum(action_delta**2)
            reward -= action_penalty

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v21(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V21: Simplified reward inspired by cube lifting.

        Minimal complexity - let the agent figure out the details.
        Based on cube reward structure which works well.
        """
        reward = 0.0
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        is_grasping = info["is_grasping"]
        dish_pos = info["dish_pos"]

        # Reach: encourage gripper to approach lid
        reach_reward = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reward += reach_reward

        # Contact bonus: bilateral contact = good positioning
        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        if has_gripper_contact and has_jaw_contact:
            reward += 0.5

        # Grasp bonus: lid held between fingers
        if is_grasping:
            reward += 1.0

        # Lift bonus: lid raised above dish
        if lid_z > 0.02:
            reward += 2.0

        # Target height bonus
        if lid_z > self.lift_height:
            reward += 1.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Penalize pushing the dish base
        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            reward -= dish_disp * 3.0  # Reduced from 10.0 - was too harsh, prevented learning

        # Success bonus
        if info["is_success"]:
            reward += 10.0

        return reward

    def _reward_v22(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V22: Anti-hacking reward with gated reach/contact and stronger lift/hold."""
        reward = 0.0
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        grasp_streak = info.get("grasp_streak", 0)
        dish_pos = info["dish_pos"]

        # Reach reward (gated if no progress after grasp attempt)
        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reward += gripper_reach * 0.5

        # Height alignment near grasp height
        grasp_z_offset = 0.005
        target_grasp_z = lid_z + grasp_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            reward += height_alignment * 0.5

        # Contact rewards only when close and aligned
        has_gripper_contact, has_jaw_contact = self._check_lid_contacts()
        if gripper_reach > 0.7 and z_error < 0.01:
            if has_gripper_contact or has_jaw_contact:
                reward += 0.5
            if has_gripper_contact and has_jaw_contact:
                reward += 1.0

        # Penalize lingering near lid without grasping after early steps
        if self._step_count > 50 and not is_grasping and gripper_reach > 0.7:
            reward -= 0.5

        # Push-down penalty
        if lid_z < 0.01:
            reward -= (0.01 - lid_z) * 50.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp + gentle grasp target
        if is_grasping:
            reward += 2.0
            gentle_target = -0.30
            gentle_band = 0.10
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.5
            else:
                reward -= (gentle_delta - gentle_band) * 0.5

            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)
            reward += lift_progress * 6.0
            if lid_z > 0.02:
                reward += 1.0
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 3.0

        # Target height + hold bonus
        if lid_z > self.lift_height:
            reward += 2.0
            reward += 0.75 * hold_count

        # Dish drift penalty (weaker to avoid dominating)
        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            dish_penalty = min(dish_disp * 2.0, 1.0)  # Cap at -1.0 per step
            reward -= dish_penalty

        # Action rate penalty during hold
        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            reward -= 0.02 * np.sum(action_delta**2)

        if info["is_success"]:
            reward += 15.0

        return reward

    def _reward_v23(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V23: Favor rim (skirt) grasps and penalize top-only contact."""
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        grasp_streak = info.get("grasp_streak", 0)
        dish_pos = info["dish_pos"]

        # Reach reward
        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reward += gripper_reach * 0.5

        # Prefer alignment with the rim (skirt center ~1.75mm above lid origin)
        rim_z_offset = 0.00175
        target_grasp_z = lid_z + rim_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            reward += height_alignment * 0.6

        # Contact type detection (top vs skirt)
        top_contact = False
        skirt_contact = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 == self._lid_top_geom_id or g2 == self._lid_top_geom_id:
                top_contact = True
            if g1 == self._lid_skirt_geom_id or g2 == self._lid_skirt_geom_id:
                skirt_contact = True

        # Contact shaping: reward skirt, penalize top-only
        if gripper_reach > 0.7 and z_error < 0.01:
            if skirt_contact:
                reward += 2.0
            if top_contact and not skirt_contact:
                reward -= 2.0

        # Penalize hovering close without grasping after initial steps
        if self._step_count > 50 and not is_grasping and gripper_reach > 0.7:
            reward -= 0.5
        if self._step_count > 50 and not is_grasping:
            reward -= 0.1

        # Penalize closing without contact (prevents "close and wait" hacks)
        if action is not None and action[3] < -0.2 and not skirt_contact and not top_contact and gripper_reach > 0.7:
            reward -= 1.0

        # Penalize staying high without grasping
        if not is_grasping and gripper_z > lid_z + 0.05:
            reward -= 0.5

        # Penalize repeated light contact without grasping
        if (top_contact or skirt_contact) and not is_grasping:
            reward -= 0.5

        # Push-down penalty
        if lid_z < 0.01:
            reward -= (0.01 - lid_z) * 50.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus + gentle grasp target
        if is_grasping:
            reward += 4.0
            reward += 0.1 * min(grasp_streak, 20)
            gentle_target = -0.30
            gentle_band = 0.10
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.8
            else:
                reward -= (gentle_delta - gentle_band) * 0.6

            # Lift shaping
            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)
            reward += lift_progress * 6.0
            if lid_z > 0.02:
                reward += 1.0
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 3.0

        # Target height + hold bonus
        if lid_z > self.lift_height:
            reward += 2.0
            reward += 0.75 * hold_count

        # Penalize pushing the dish base
        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            dish_penalty = min(dish_disp * 2.0, 1.0)  # Cap at -1.0 per step
            reward -= dish_penalty

        # Action rate penalty during hold phase
        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            reward -= 0.02 * np.sum(action_delta**2)

        # Success bonus
        if info["is_success"]:
            reward += 30.0

        return reward

    def _reward_v24(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V24: Penalize one-sided contact without closing or grasping."""
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        dish_pos = info["dish_pos"]
        grasp_streak = info.get("grasp_streak", 0)
        top_contact_streak = self._top_contact_streak
        top_contact_streak = self._top_contact_streak
        top_contact_streak = self._top_contact_streak

        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reward += gripper_reach * 0.5

        rim_z_offset = 0.00175
        target_grasp_z = lid_z + rim_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            reward += height_alignment * 0.6

        top_contact = False
        skirt_contact = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 == self._lid_top_geom_id or g2 == self._lid_top_geom_id:
                top_contact = True
            if g1 == self._lid_skirt_geom_id or g2 == self._lid_skirt_geom_id:
                skirt_contact = True

        if gripper_reach > 0.7 and z_error < 0.01:
            if skirt_contact:
                reward += 2.0
            if top_contact and not skirt_contact:
                reward -= 2.0

        # Penalize lingering single-sided contact without grasping
        if (top_contact ^ skirt_contact) and not is_grasping:
            reward -= 1.5

        # Penalize closing without contact (prevents "close and wait" hacks)
        if action is not None and action[3] < -0.2 and not skirt_contact and not top_contact and gripper_reach > 0.7:
            reward -= 1.0

        # Penalize staying high without grasping
        if not is_grasping and gripper_z > lid_z + 0.05:
            reward -= 0.5

        # Penalize repeated light contact without grasping
        if (top_contact or skirt_contact) and not is_grasping:
            reward -= 0.5

        # Push-down penalty
        if lid_z < 0.01:
            reward -= (0.01 - lid_z) * 50.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        # Grasp bonus + gentle grasp target
        if is_grasping:
            reward += 6.0
            reward += 0.1 * min(grasp_streak, 20)
            gentle_target = -0.30
            gentle_band = 0.10
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.8
            else:
                reward -= (gentle_delta - gentle_band) * 0.6

            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)
            reward += lift_progress * 6.0
            if lid_z > 0.02:
                reward += 1.0
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 3.0

        if lid_z > self.lift_height:
            reward += 2.0
            reward += 0.75 * hold_count

        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            reward -= dish_disp * 2.0

        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            reward -= 0.02 * np.sum(action_delta**2)

        if info["is_success"]:
            reward += 30.0

        return reward

    def _reward_v25(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V25: Warm-up one-sided contact, decay reach/align, penalize prolonged top contact."""
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        dish_pos = info["dish_pos"]
        grasp_streak = info.get("grasp_streak", 0)
        top_contact_count = info.get("top_contact_count", 0)

        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reach_scale = 0.5 if self._step_count < 100 else 0.2
        reward += gripper_reach * reach_scale

        rim_z_offset = 0.00175
        target_grasp_z = lid_z + rim_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            align_scale = 0.6 if self._step_count < 100 else 0.3
            reward += height_alignment * align_scale

        top_contact = False
        skirt_contact = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 == self._lid_top_geom_id or g2 == self._lid_top_geom_id:
                top_contact = True
            if g1 == self._lid_skirt_geom_id or g2 == self._lid_skirt_geom_id:
                skirt_contact = True

        # Option A: Simplified contact penalties

        # Reward skirt contact when positioned correctly (good grasp location)
        if gripper_reach > 0.7 and z_error < 0.01 and skirt_contact:
            reward += 2.0

        # Immediate penalty for top-only contact when positioned (wrong grasp location)
        if gripper_reach > 0.7 and z_error < 0.01 and top_contact and not skirt_contact:
            reward -= 2.0

        # Escalating penalty for prolonged top contact (arm gets stuck)
        if top_contact_count > 10:
            # After 10 steps of top contact without grasping, escalate penalty
            stuck_penalty = min((top_contact_count - 10) * 0.1, 3.0)  # Cap at -3.0
            reward -= stuck_penalty

        # Penalize closing without contact (prevents "close and wait" hacks)
        if action is not None and action[3] < -0.2 and not skirt_contact and not top_contact and gripper_reach > 0.7:
            reward -= 1.0

        # Penalize pinch-lift without grasping lid
        if not is_grasping and gripper_z > lid_z + 0.05 and not top_contact and not skirt_contact:
            reward -= 1.0

        # Push-down penalty
        if lid_z < 0.01:
            reward -= (0.01 - lid_z) * 50.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        if is_grasping:
            reward += 6.0
            reward += 0.1 * min(grasp_streak, 20)
            gentle_target = -0.30
            gentle_band = 0.10
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.8
            else:
                reward -= (gentle_delta - gentle_band) * 0.6

            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)
            reward += lift_progress * 6.0
            if lid_z > 0.02:
                reward += 1.0
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 3.0

        if lid_z > self.lift_height:
            reward += 2.0
            reward += 0.75 * hold_count

        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            if dish_disp > 0.01:  # 1cm leeway before penalty starts
                dish_penalty = min((dish_disp - 0.01) * 2.0, 1.0)  # Cap at -1.0 per step
                reward -= dish_penalty

        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            reward -= 0.02 * np.sum(action_delta**2)

        if info["is_success"]:
            reward += 30.0

        return reward

    def _reward_v26(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        """V26: One-time bonuses for contact/closing, increased grasp reward."""
        reward = 0.0
        lid_pos = info["lid_pos"]
        lid_z = info["lid_z"]
        gripper_to_lid = info["gripper_to_lid"]
        gripper_state = info["gripper_state"]
        is_grasping = info["is_grasping"]
        hold_count = info["hold_count"]
        dish_pos = info["dish_pos"]
        grasp_streak = info.get("grasp_streak", 0)
        top_contact_count = info.get("top_contact_count", 0)

        gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_lid)
        reach_scale = 0.5 if self._step_count < 100 else 0.2
        reward += gripper_reach * reach_scale

        rim_z_offset = 0.00175
        target_grasp_z = lid_z + rim_z_offset
        gripper_z = info["gripper_pos"][2]
        z_error = abs(gripper_z - target_grasp_z)
        if gripper_reach > 0.7:
            height_alignment = 1.0 - np.tanh(50.0 * z_error)
            align_scale = 0.6 if self._step_count < 100 else 0.3
            reward += height_alignment * align_scale

        top_contact = False
        skirt_contact = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 == self._lid_top_geom_id or g2 == self._lid_top_geom_id:
                top_contact = True
            if g1 == self._lid_skirt_geom_id or g2 == self._lid_skirt_geom_id:
                skirt_contact = True

        # One-time bonus for first skirt contact when positioned correctly
        if gripper_reach > 0.7 and z_error < 0.01 and skirt_contact and not self._has_rewarded_skirt_contact:
            reward += 5.0  # One-time bonus (was continuous 2.0/step)
            self._has_rewarded_skirt_contact = True

        # One-time bonus for closing after making skirt contact
        if skirt_contact and action is not None and not self._has_rewarded_closing:
            closing_action = -action[3]  # Negative gripper action = closing
            if closing_action > 0.5:  # Closing significantly (threshold to avoid tiny actions)
                reward += 8.0  # One-time bonus for decisive closing
                self._has_rewarded_closing = True

        # Immediate penalty for top-only contact when positioned
        if gripper_reach > 0.7 and z_error < 0.01 and top_contact and not skirt_contact:
            reward -= 2.0

        # Escalating penalty for prolonged top contact (arm gets stuck)
        if top_contact_count > 10:
            stuck_penalty = min((top_contact_count - 10) * 0.1, 3.0)
            reward -= stuck_penalty

        # Penalize closing without contact (prevents "close and wait" hacks)
        if action is not None and action[3] < -0.2 and not skirt_contact and not top_contact and gripper_reach > 0.7:
            reward -= 1.0

        # Penalize pinch-lift without grasping lid
        if not is_grasping and gripper_z > lid_z + 0.05 and not top_contact and not skirt_contact:
            reward -= 1.0

        # Push-down penalty
        if lid_z < 0.01:
            reward -= (0.01 - lid_z) * 50.0

        # Drop penalty
        if was_grasping and not is_grasping:
            reward -= 2.0

        if is_grasping:
            reward += 15.0  # Increased from 6.0 - make grasping clearly better than contact farming
            reward += 0.1 * min(grasp_streak, 20)
            gentle_target = -0.30
            gentle_band = 0.10
            gentle_delta = abs(gripper_state - gentle_target)
            if gentle_delta <= gentle_band:
                reward += (1.0 - gentle_delta / gentle_band) * 0.8
            else:
                reward -= (gentle_delta - gentle_band) * 0.6

            lift_progress = max(0, lid_z - 0.0155) / (self.lift_height - 0.0155)
            reward += lift_progress * 6.0
            if lid_z > 0.02:
                reward += 1.0
            if lid_z > 0.04:
                threshold_progress = min(1.0, (lid_z - 0.04) / (self.lift_height - 0.04))
                reward += threshold_progress * 3.0

        if lid_z > self.lift_height:
            reward += 2.0
            reward += 0.75 * hold_count

        if self._dish_start_pos is not None:
            dish_disp = np.linalg.norm(dish_pos - self._dish_start_pos)
            if dish_disp > 0.01:  # 1cm leeway
                dish_penalty = min((dish_disp - 0.01) * 2.0, 1.0)
                reward -= dish_penalty

        if action is not None and lid_z > self.lift_height and hold_count > 0:
            action_delta = action - self._prev_action
            reward -= 0.02 * np.sum(action_delta**2)

        if info["is_success"]:
            reward += 30.0

        return reward

    def render(self, camera: str = "closeup") -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            cam = mujoco.MjvCamera()
            if camera == "closeup":
                # Side view close to lid
                cam.lookat[:] = [0.30, 0.0, 0.03]
                cam.distance = 0.35
                cam.azimuth = 90
                cam.elevation = -15
            elif camera == "wide":
                # Diagonal view of arm and lid
                cam.lookat[:] = [0.25, 0.0, 0.05]
                cam.distance = 0.8
                cam.azimuth = 135
                cam.elevation = -25
            else:  # "wide2"
                # Diagonal view from other side
                cam.lookat[:] = [0.25, 0.0, 0.05]
                cam.distance = 0.8
                cam.azimuth = 45
                cam.elevation = -25
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
