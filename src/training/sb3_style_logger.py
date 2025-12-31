"""SB3-style logger for RoboBase.

Provides familiar Stable-Baselines3 table-style logging format while using
RoboBase's DrQ-v2 training with data augmentation.

Usage:
    # MUST be called before importing robobase.workspace
    from src.training.sb3_style_logger import patch_robobase_logger
    patch_robobase_logger()

    # Then import and use normally
    from robobase.workspace import Workspace
"""

import datetime
from collections import defaultdict

from termcolor import colored


# Mapping from RoboBase metric names to SB3-style names
METRIC_NAME_MAP = {
    # Eval metrics
    "episode_reward": "ep_rew_mean",
    "episode_length": "ep_len_mean",
    "success_rate": "success_rate",
    # Train metrics
    "env_steps": "total_timesteps",
    "env_episodes": "episodes",
    "env_steps_per_second": "fps",
    "agent_batched_updates_per_second": "update_fps",
    "buffer_size": "buffer_size",
    "iteration": "iterations",
    "total_time": "time_elapsed",
    # Agent metrics (passed through)
    "actor_loss": "actor_loss",
    "critic_loss": "critic_loss",
    "alpha_loss": "alpha_loss",
    "alpha": "alpha",
}

# Categories for grouping metrics in SB3 style
ROLLOUT_METRICS = {"ep_rew_mean", "ep_len_mean", "success_rate"}
TIME_METRICS = {"fps", "update_fps", "iterations", "total_timesteps", "time_elapsed", "episodes"}
TRAIN_METRICS = {"actor_loss", "critic_loss", "alpha_loss", "alpha", "buffer_size"}


def _format_value(value, ty):
    """Format a value for display."""
    if ty == "int":
        return str(int(value))
    elif ty == "float":
        if abs(value) < 0.01 and value != 0:
            return f"{value:.2e}"
        elif abs(value) >= 1000:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"
    elif ty == "time":
        return str(datetime.timedelta(seconds=int(value)))
    else:
        return str(value)


def _get_metric_type(key):
    """Infer metric type from key name."""
    if "loss" in key or "alpha" in key:
        return "float"
    elif key == "time_elapsed":
        return "time"
    elif any(x in key for x in ["steps", "episodes", "iterations", "size", "len"]):
        return "int"
    elif "reward" in key or "fps" in key or "rate" in key:
        return "float"
    return "float"


def _categorize_metric(key):
    """Determine which category a metric belongs to."""
    if key in ROLLOUT_METRICS:
        return "rollout"
    elif key in TIME_METRICS:
        return "time"
    elif key in TRAIN_METRICS:
        return "train"
    else:
        return "other"


# Will hold reference to the original MetersGroup before patching
_OriginalMetersGroup = None


class SB3StyleMetersGroup:
    """MetersGroup that outputs SB3-style table format."""

    def __init__(self, csv_file_name, formating, save_csv: bool):
        global _OriginalMetersGroup
        # Use composition with ORIGINAL class to keep CSV functionality
        self._inner = _OriginalMetersGroup(csv_file_name, formating, save_csv)
        self._meters = self._inner._meters
        self._formating = formating
        self._save_csv = save_csv

    def log(self, key, value, n=1):
        self._inner.log(key, value, n)

    def _prime_meters(self):
        return self._inner._prime_meters()

    def _dump_to_csv(self, data):
        self._inner._dump_to_csv(data)

    def _dump_to_console(self, data, prefix):
        """Output in SB3-style table format."""
        if not data:
            return

        # Convert keys to SB3-style names
        sb3_data = {}
        for key, value in data.items():
            sb3_key = METRIC_NAME_MAP.get(key, key)
            sb3_data[sb3_key] = value

        # Group by category
        categories = defaultdict(dict)
        for key, value in sb3_data.items():
            cat = _categorize_metric(key)
            categories[cat][key] = value

        # Add prefix-specific metrics
        if prefix == "eval":
            categories["rollout"] = {
                k: v for k, v in sb3_data.items() if k in ROLLOUT_METRICS
            }

        # Calculate column widths
        key_width = 22
        val_width = 12

        # Print table
        border = "-" * (key_width + val_width + 5)

        if prefix == "train":
            color = "yellow"
        elif prefix == "pretrain":
            color = "red"
        else:
            color = "green"

        print(colored(border, color))

        # Print each category
        category_order = ["rollout", "time", "train", "other"]
        for cat in category_order:
            if cat not in categories or not categories[cat]:
                continue

            # Category header
            header = f"| {cat}/"
            print(colored(f"{header:<{key_width}}|{' ' * val_width}|", color))

            # Metrics in this category
            for key, value in sorted(categories[cat].items()):
                ty = _get_metric_type(key)
                formatted = _format_value(value, ty)
                line = f"|    {key:<{key_width - 5}}| {formatted:<{val_width - 1}}|"
                print(colored(line, color))

        print(colored(border, color))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        if self._save_csv:
            self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


def patch_robobase_logger():
    """Monkey-patch RoboBase's logger to use SB3-style output.

    Must be called BEFORE importing robobase.workspace.
    """
    global _OriginalMetersGroup

    import robobase.logger as rb_logger

    # Store original BEFORE patching (critical to avoid recursion)
    _OriginalMetersGroup = rb_logger.MetersGroup

    # Replace with our SB3-style version
    rb_logger.MetersGroup = SB3StyleMetersGroup

    print("Patched RoboBase logger for SB3-style output")
