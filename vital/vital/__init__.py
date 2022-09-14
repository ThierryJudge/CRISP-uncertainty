import os
from pathlib import Path

ENV_VITAL_HOME = "VITAL_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def get_vital_home() -> Path:
    """Resolves the home directory for the `vital` library, used to save/cache data reusable across scripts/runs.

    Returns:
        Path to the home directory for the `vital` library.
    """
    return Path(os.getenv(ENV_VITAL_HOME, os.getenv("XDG_CACHE_HOME", DEFAULT_CACHE_DIR))).expanduser() / "vital"
