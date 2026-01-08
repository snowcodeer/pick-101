"""Train a lift policy for the petri lid (free dish base) using SAC."""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.envs.lift_petri_lid_free_dish import LiftPetriLidFreeDishCartesianEnv


class VecNormalizeSaveCallback(BaseCallback):
    """Periodically save VecNormalize stats during training."""

    def __init__(self, save_freq: int, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.training_env
            if hasattr(vec_env, "save"):
                vec_env.save(self.save_path)
                if self.verbose:
                    print(f"Saved vec_normalize to {self.save_path}")
        return True


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    env = LiftPetriLidFreeDishCartesianEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v19"),
        curriculum_stage=env_cfg.get("curriculum_stage", 0),
        lock_wrist=env_cfg.get("lock_wrist", False),
    )
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lift_petri_lid_free_dish_500k.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model .zip file (transfer learning)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps from config (useful for resuming)")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Environment: LiftPetriLidFreeDishCartesianEnv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / f"{timestamp}_resumed"
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.yaml")

    if args.resume:
        with open(output_dir / "RESUME_INFO.txt", "w") as f:
            f.write(f"Resumed from: {resume_dir}\n")
            f.write(f"Timestamp: {timestamp}\n")

    pretrained = args.pretrained or train_cfg.get("pretrained")
    vec_normalize_path = None
    if args.resume:
        vec_normalize_path = resume_dir / "vec_normalize.pkl"

    env = DummyVecEnv([lambda: make_env(env_cfg)])

    if vec_normalize_path and vec_normalize_path.exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    elif vec_normalize_path:
        raise ValueError(f"vec_normalize.pkl not found: {vec_normalize_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=env_cfg["normalize_reward"],
        )
        if pretrained:
            print("Using fresh VecNormalize for transfer learning")

    eval_env = DummyVecEnv([lambda: make_env(env_cfg)])
    if vec_normalize_path and vec_normalize_path.exists():
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=False,
            training=False,
        )

    resume_step = 0
    if args.resume:
        checkpoints = list((resume_dir / "checkpoints").glob("*.zip"))
        def get_step_number(path):
            name = path.stem
            parts = name.split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_step = get_step_number(latest_checkpoint)
            model = SAC.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint} (step {resume_step})")
            with open(output_dir / "RESUME_INFO.txt", "a") as f:
                f.write(f"Checkpoint: {latest_checkpoint}\n")
                f.write(f"Resume step: {resume_step}\n")
        else:
            raise ValueError(f"No checkpoints found in {resume_dir / 'checkpoints'}")
    elif pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise ValueError(f"Pretrained model not found: {pretrained_path}")
        model = SAC.load(pretrained_path, env=env, device=device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.num_timesteps = 0
        model._episode_num = 0
        model.replay_buffer.reset()
        print(f"Loaded pretrained weights from {pretrained_path}")
        with open(output_dir / "PRETRAINED_INFO.txt", "w") as f:
            f.write(f"Pretrained from: {pretrained_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            learning_starts=sac_cfg["learning_starts"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_lift",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )

    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
        resume_step=resume_step,
    )
    vecnorm_callback = VecNormalizeSaveCallback(
        save_freq=train_cfg["save_freq"],
        save_path=output_dir / "vec_normalize.pkl",
        verbose=1,
    )

    timesteps = args.timesteps if args.timesteps is not None else train_cfg["timesteps"]

    if args.resume:
        print(f"Loaded model num_timesteps: {model.num_timesteps}")
        reset_num_timesteps = False
        learn_timesteps = timesteps
        target_total = model.num_timesteps + timesteps
        print(f"\nResuming Lift (Free Dish) training from step {model.num_timesteps}...")
        print(f"Training for {timesteps} additional timesteps (target: {target_total} total)")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        target_total = timesteps
        print(f"\nStarting Lift (Free Dish) training for {timesteps} timesteps...")

    print("Action space: delta XYZ + gripper (4 dims)")
    print(f"Output directory: {output_dir}")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=[checkpoint_callback, eval_callback, plot_callback, vecnorm_callback],
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
