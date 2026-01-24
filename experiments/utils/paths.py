"""Path configuration for SatCLIP experiments.

This module provides centralized path management using environment variables.
All paths are configurable via environment variables, with sensible defaults.

Environment Variables:
    SATCLIP_ROOT: Project root directory
    SATCLIP_DATA_DIR: Base data directory
    SATCLIP_DATASET_PATH: Preprocessed HF dataset location
    SATCLIP_LOGS_DIR: Training logs directory
    SATCLIP_HF_CACHE: HuggingFace cache directory

Usage:
    from experiments.utils.paths import Paths

    # Get paths (auto-detects environment)
    paths = Paths()
    print(paths.dataset)  # /path/to/dataset
    print(paths.logs)     # /path/to/logs

    # Or use class methods directly
    dataset_path = Paths.get_dataset_path()
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _detect_environment() -> str:
    """Detect if running on HPC or locally."""
    if os.path.exists("/engrfs"):
        return "hpc"
    elif os.path.exists("/Users"):
        return "local"
    return "unknown"


def _get_default_root() -> Path:
    """Get default project root based on environment."""
    env = _detect_environment()
    if env == "hpc":
        return Path("/engrfs/project/jacobsn/hiqbal/src/satclip")
    elif env == "local":
        return Path("/Users/hamzaiqbal/grad/learned_activation/satclip")
    # Fallback: use this file's location
    return Path(__file__).parent.parent.parent


def _get_default_data_dir() -> Path:
    """Get default data directory based on environment."""
    env = _detect_environment()
    if env == "hpc":
        return Path("/engrfs/tmp/jacobsn/hiqbal_satclip")
    else:
        return _get_default_root() / "data"


@dataclass
class Paths:
    """Centralized path configuration.

    All paths can be overridden via environment variables or constructor args.
    """

    root: Path
    data_dir: Path
    dataset: Path
    logs: Path
    hf_cache: Path
    configs: Path
    scripts: Path

    def __init__(
        self,
        root: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        dataset: Optional[Path] = None,
        logs: Optional[Path] = None,
        hf_cache: Optional[Path] = None,
    ):
        """Initialize paths from environment variables or arguments.

        Args:
            root: Project root (env: SATCLIP_ROOT)
            data_dir: Data directory (env: SATCLIP_DATA_DIR)
            dataset: Dataset path (env: SATCLIP_DATASET_PATH)
            logs: Logs directory (env: SATCLIP_LOGS_DIR)
            hf_cache: HF cache directory (env: SATCLIP_HF_CACHE)
        """
        # Root path
        self.root = Path(
            root or os.environ.get("SATCLIP_ROOT") or _get_default_root()
        )

        # Data directory
        self.data_dir = Path(
            data_dir or os.environ.get("SATCLIP_DATA_DIR") or _get_default_data_dir()
        )

        # Dataset path
        self.dataset = Path(
            dataset
            or os.environ.get("SATCLIP_DATASET_PATH")
            or (self.data_dir / "satclip_hf_preprocessed")
        )

        # Logs directory
        self.logs = Path(
            logs or os.environ.get("SATCLIP_LOGS_DIR") or (self.data_dir / "logs")
        )

        # HuggingFace cache
        self.hf_cache = Path(
            hf_cache
            or os.environ.get("SATCLIP_HF_CACHE")
            or (self.data_dir / "hf_cache")
        )

        # Derived paths (always relative to root)
        self.configs = self.root / "experiments" / "configs"
        self.scripts = self.root / "experiments" / "scripts"

    @classmethod
    def get_dataset_path(cls) -> str:
        """Get dataset path as string (for config files)."""
        return str(cls().dataset)

    @classmethod
    def get_logs_dir(cls) -> str:
        """Get logs directory as string."""
        return str(cls().logs)

    @classmethod
    def get_hf_cache(cls) -> str:
        """Get HF cache directory as string."""
        return str(cls().hf_cache)

    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.logs.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        return (
            f"Paths(\n"
            f"  root={self.root}\n"
            f"  data_dir={self.data_dir}\n"
            f"  dataset={self.dataset}\n"
            f"  logs={self.logs}\n"
            f"  hf_cache={self.hf_cache}\n"
            f")"
        )


# Convenience: default paths instance
_default_paths: Optional[Paths] = None


def get_paths() -> Paths:
    """Get default paths instance (cached)."""
    global _default_paths
    if _default_paths is None:
        _default_paths = Paths()
    return _default_paths


# Environment detection helper
def get_environment() -> str:
    """Get current environment name ('hpc', 'local', or 'unknown')."""
    return _detect_environment()


def is_hpc() -> bool:
    """Check if running on HPC."""
    return _detect_environment() == "hpc"


def is_local() -> bool:
    """Check if running locally."""
    return _detect_environment() == "local"
