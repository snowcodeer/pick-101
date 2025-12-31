"""Train image-based RL agent for SO-101 lift task using RoboBase."""

# Use spawn for multiprocessing (required for EGL/GPU rendering on AMD)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from robobase.workspace import Workspace

from src.training.so101_factory import SO101Factory


# Find RoboBase config path
import robobase
ROBOBASE_CFG_PATH = str(Path(robobase.__file__).parent / "cfgs")


@hydra.main(
    config_path="cfgs",
    config_name="so101_lift",
    version_base=None,
)
def main(cfg: DictConfig):
    """Train image-based RL agent."""
    # Create workspace with SO-101 factory
    workspace = Workspace(cfg, env_factory=SO101Factory())
    workspace.train()


if __name__ == "__main__":
    # Register RoboBase config path with hydra
    from hydra import initialize_config_dir
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    class RoboBaseSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path):
            search_path.prepend(
                provider="robobase",
                path=f"file://{ROBOBASE_CFG_PATH}",
            )

    Plugins.instance().register(RoboBaseSearchPathPlugin)
    main()
