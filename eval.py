"""Evaluate and visualize a trained policy."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.pick_cube import PickCubeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory (e.g., runs/pick_cube/20241213_215500)")
    parser.add_argument("--model", type=str, default=None, help="Model to load: 'best', 'final', or checkpoint name (default: best)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (default: from config)")
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer")
    parser.add_argument("--record", action="store_true", help="Record videos to exp_dir/videos/")
    parser.add_argument("--width", type=int, default=None, help="Video width (default: from config)")
    parser.add_argument("--height", type=int, default=None, help="Video height (default: from config)")
    args = parser.parse_args()

    # Load experiment directory
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    # Load config from experiment
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        eval_cfg = config.get("eval", {})
    else:
        eval_cfg = {}

    # Resolve parameters (CLI overrides config)
    episodes = args.episodes or eval_cfg.get("episodes", 5)
    video_width = args.width or eval_cfg.get("video_width", 640)
    video_height = args.height or eval_cfg.get("video_height", 480)

    # Find model to load
    model_choice = args.model or "best"
    if model_choice == "best":
        model_path = exp_dir / "best_model" / "best_model.zip"
    elif model_choice == "final":
        model_path = exp_dir / "final_model.zip"
    else:
        # Assume it's a checkpoint name
        model_path = exp_dir / "checkpoints" / model_choice
        if not model_path.suffix:
            model_path = model_path.with_suffix(".zip")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Available models:")
        if (exp_dir / "best_model" / "best_model.zip").exists():
            print("  --model best")
        if (exp_dir / "final_model.zip").exists():
            print("  --model final")
        checkpoints = list((exp_dir / "checkpoints").glob("*.zip")) if (exp_dir / "checkpoints").exists() else []
        for ckpt in checkpoints[:5]:
            print(f"  --model {ckpt.name}")
        return

    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create environment
    env = PickCubeEnv(render_mode=None)

    # Load normalization stats if available
    vec_normalize_path = exp_dir / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        # Wrap the same env instance so viewer stays in sync
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec_env = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    else:
        use_vec_env = False
        print("No normalization stats found, using raw observations")

    # Setup video recording if requested
    renderer = None
    record_dir = None
    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = exp_dir / "videos" / timestamp
        record_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, width=video_width, height=video_height)
        print(f"Recording videos to {record_dir}")

    # Run episodes
    successes = 0
    total_rewards = []

    for ep in range(episodes):
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        done = False
        episode_reward = 0
        step = 0
        frames = []

        # Optional: launch viewer
        viewer = None
        if args.render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        while not done:
            # Record frame if recording
            if renderer is not None:
                renderer.update_scene(env.data)
                frames.append(renderer.render())

            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            if use_vec_env:
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                info = infos[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            step += 1

            # Update viewer with real-time pacing
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)  # ~50fps for visible motion

        if viewer is not None:
            viewer.close()

        # Save video
        if frames:
            video_path = record_dir / f"episode_{ep + 1}.mp4"
            imageio.mimsave(video_path, frames, fps=50)
            print(f"Saved video to {video_path}")

        total_rewards.append(episode_reward)
        if info.get("is_success", False):
            successes += 1

        print(f"Episode {ep + 1}: reward={episode_reward:.2f}, steps={step}, success={info.get('is_success', False)}")

    # Summary
    print(f"\n{'='*40}")
    print(f"Results over {episodes} episodes:")
    print(f"  Success rate: {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"  Mean reward: {sum(total_rewards)/len(total_rewards):.2f}")

    if renderer is not None:
        renderer.close()
    env.close()
    if use_vec_env:
        vec_env.close()


if __name__ == "__main__":
    main()
