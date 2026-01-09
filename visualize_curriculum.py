"""Visualize curriculum stages and save videos for sanity checking.

Usage:
    python visualize_curriculum.py --config configs/grasp_stage1_close.yaml --episodes 3
"""
import argparse
from pathlib import Path

import imageio
import numpy as np
import yaml

from src.envs.lift_petri_lid_free_dish import LiftPetriLidFreeDishCartesianEnv


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def visualize_curriculum_stage(config_path: str, num_episodes: int = 3, output_dir: str = "curriculum_videos"):
    """Run episodes with random actions to visualize initial states and behavior."""
    config = load_config(config_path)
    env_cfg = config["env"]
    exp_name = config["experiment"]["name"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment with rendering
    env = LiftPetriLidFreeDishCartesianEnv(
        render_mode="rgb_array",
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v20"),
        curriculum_stage=env_cfg.get("curriculum_stage", 0),
        lock_wrist=env_cfg.get("lock_wrist", False),
    )

    curriculum_stage = env_cfg.get("curriculum_stage", 0)
    print(f"Visualizing {exp_name}")
    print(f"Curriculum stage: {curriculum_stage}")
    print(f"Lock wrist: {env_cfg.get('lock_wrist', False)}")
    print(f"Action scale: {env_cfg.get('action_scale', 0.02)}")
    print(f"Max episode steps: {env_cfg.get('max_episode_steps', 200)}")

    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        frames = []
        obs, info = env.reset()

        # Debug initial state after reset
        print(f"  Initial state:")
        print(f"    - lid_z: {info.get('lid_z', 0.0):.4f}m, target: {env_cfg.get('lift_height', 0.08):.4f}m")
        print(f"    - is_grasping: {info.get('is_grasping', False)}")
        print(f"    - is_lifted: {info.get('is_lifted', False)}")

        # Capture initial frame (most important for sanity check)
        frame = env.render()
        frames.append(frame)

        done = False
        step = 0
        total_reward = 0

        while not done and step < env_cfg.get("max_episode_steps", 200):
            # Use zero action (no movement) to see what happens with current state
            # This is good for Stage 1 - should just hold position
            if curriculum_stage == 1:
                # Try to hold still
                action = np.zeros(4)
            else:
                # Small random actions for other stages
                action = env.action_space.sample() * 0.3

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Capture frame every 5 steps to reduce file size
            if step % 5 == 0:
                frame = env.render()
                frames.append(frame)

            step += 1

        # Capture final frame
        frame = env.render()
        frames.append(frame)

        # Save video
        video_path = output_path / f"{exp_name}_ep{ep + 1}.mp4"
        imageio.mimsave(video_path, frames, fps=20)

        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Success: {info.get('is_success', False)}")
        print(f"  Lid lifted: {info.get('is_lifted', False)}")
        print(f"    - lid_z: {info.get('lid_z', 0.0):.4f}m, target: {env_cfg.get('lift_height', 0.08):.4f}m")
        print(f"    - is_grasping: {info.get('is_grasping', False)}")
        print(f"    - hold_count: {info.get('hold_count', 0)}/{env_cfg.get('hold_steps', 10)}")
        print(f"  Dish displacement: {info.get('dish_disp', 0.0)*1000:.1f}mm (stable: {info.get('dish_stable', False)})")
        print(f"  Saved to: {video_path}")

    env.close()
    print(f"\nAll videos saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize curriculum stages")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--output", type=str, default="curriculum_videos", help="Output directory")
    args = parser.parse_args()

    visualize_curriculum_stage(args.config, args.episodes, args.output)


if __name__ == "__main__":
    main()
