"""Simple visualization of SO-101 arm with a petri dish.

Shows the SO-101 robot with wide gripper and a scaled-down petri dish (35mm base, 37mm lid).
The dish base is fixed, and the lid is movable with visible collision geometry for debugging.

Usage:
    python visualize_arm_petri.py
"""
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


def create_petri_dish_scene():
    """Create XML with robot arm and petri dish using existing STL files."""
    xml = """
    <mujoco model="so101_with_petri">
        <include file="so101_new_calib.xml"/>

        <option timestep="0.002" gravity="0 0 -9.81" noslip_iterations="3" impratio="10" cone="elliptic"/>

        <default>
            <geom solref="0.001 1" solimp="0.99 0.99 0.001"/>
        </default>

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="90" elevation="-20" offwidth="640" offheight="480"/>
            <map znear="0.005"/>
        </visual>

        <asset>
            <!-- Skybox background -->
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>

            <!-- Ground plane texture -->
            <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                     rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
                     width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true"
                      texrepeat="5 5" reflectance="0.2"/>

            <!-- Petri dish meshes (scaled down to 35mm/37mm from 87mm/91mm) -->
            <mesh name="petri_dish" file="petri_dish.STL" scale="0.000402 0.000402 0.000402"/>
            <mesh name="petri_dish_lid" file="petri_dish_lid.STL" scale="0.000407 0.000407 0.000407"/>

            <!-- Petri dish materials -->
            <material name="dish_plastic" rgba="0.85 0.85 0.9 0.5" shininess="0.9" specular="0.8"/>
            <material name="lid_plastic" rgba="0.8 0.8 0.85 0.3" shininess="0.9" specular="0.8"/>
        </asset>

        <worldbody>
            <!-- Lighting -->
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>

            <!-- Ground plane -->
            <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>

            <!-- Petri Dish Base: FIXED (no freejoint) - OD=35mm, H=4mm total -->
            <body name="petri_dish" pos="0.25 0 0.001">
                <!-- Visual mesh (no collision) -->
                <geom type="mesh" mesh="petri_dish" material="dish_plastic" contype="0" conaffinity="0" group="0"/>
                <!-- Collision: bottom + outer wall (VISIBLE FOR DEBUG) -->
                <!-- Bottom: 0.5mm full height (half-height=0.00025) -->
                <geom name="dish_bottom" type="cylinder" size="0.0175 0.00025" pos="0 0 0"
                      rgba="0 1 0 0.3" mass="0.005" friction="0.5 0.05 0.001" contype="1" conaffinity="1"/>
                <!-- Wall: 3.5mm full height (half-height=0.00175) -->
                <geom name="dish_wall" type="cylinder" size="0.0175 0.00175" pos="0 0 0.002"
                      rgba="0 1 0 0.3" mass="0.004" friction="0.5 0.05 0.001" contype="1" conaffinity="1"/>
            </body>

            <!-- Petri Dish Lid: MOVABLE (with freejoint) - OD=37mm, H=2.5mm total -->
            <body name="petri_lid" pos="0.25 0 0.0155" quat="0 1 0 0">
                <freejoint name="petri_lid_joint"/>
                <!-- Visual mesh (no collision, no mass) -->
                <geom type="mesh" mesh="petri_dish_lid" material="lid_plastic" contype="0" conaffinity="0" group="0" mass="0"/>
                <!-- Collision: flat top + graspable rim (VISIBLE FOR DEBUG) -->
                <!-- Flat top surface: thin disk covering the dish, 0.5mm thick -->
                <geom name="lid_top" type="cylinder" size="0.0185 0.00025" pos="0 0 0.0005"
                      rgba="1 0 0 0.5" mass="0.001" friction="0.5 0.05 0.001" contype="1" conaffinity="1"/>
                <!-- Graspable rim: outer edge only, 2mm thick -->
                <geom name="lid_skirt" type="cylinder" size="0.0185 0.001" pos="0 0 0.00175"
                      rgba="1 0.5 0 0.7" mass="0.001" friction="0.5 0.05 0.001" contype="1" conaffinity="1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Write to temporary file in models/so101 directory
    temp_path = Path("models/so101/temp_petri_viz.xml")
    temp_path.write_text(xml)
    return temp_path


def main():
    print("SO-101 Robot Arm + Petri Dish Visualization")
    print("=" * 50)
    print("Controls:")
    print("  - Drag with mouse to rotate view")
    print("  - Right-click drag to pan")
    print("  - Scroll to zoom")
    print("  - Double-click to select/deselect objects")
    print("  - Ctrl + click to apply forces")
    print("  - Press ESC or close window to exit")
    print("=" * 50)

    xml_path = create_petri_dish_scene()

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)

        mujoco.mj_resetData(model, data)

        joint_names = [model.joint(i).name for i in range(model.njnt)]

        if 'shoulder_pan' in joint_names:
            idx = joint_names.index('shoulder_pan')
            data.qpos[idx] = 0.0 
            data.ctrl[idx] = 0.0

        if 'shoulder_lift' in joint_names:
            idx = joint_names.index('shoulder_lift')
            data.qpos[idx] = -0.5 
            data.ctrl[idx] = -0.5

        if 'elbow_flex' in joint_names:
            idx = joint_names.index('elbow_flex')
            data.qpos[idx] = 1.0
            data.ctrl[idx] = 1.0

        if 'wrist_flex' in joint_names:
            idx = joint_names.index('wrist_flex')
            data.qpos[idx] = 1.0  
            data.ctrl[idx] = 1.0

        if 'wrist_roll' in joint_names:
            idx = joint_names.index('wrist_roll')
            data.qpos[idx] = 1.5
            data.ctrl[idx] = 1.5

        mujoco.mj_forward(model, data)

        # Separate robot joints from object joints
        robot_joints = []
        object_joints = []
        for i in range(model.njnt):
            joint_name = model.joint(i).name
            if 'lid' in joint_name:
                object_joints.append((i, joint_name))
            else:
                robot_joints.append((i, joint_name))

        print(f"\nRobot arm has {len(robot_joints)} joints:")
        for i, name in robot_joints:
            print(f"  {i}: {name}")

        if object_joints:
            print(f"\nPetri dish lid (movable) with freejoint:")
            for i, name in object_joints:
                print(f"  {i}: {name}")

        print(f"\nPetri dish base: FIXED (no freejoint)")

        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.25, 0.0, 0.15]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20

            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

    finally:
        # Cleanup
        if xml_path.exists():
            xml_path.unlink()

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
