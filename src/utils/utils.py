"""
Miscellaneous useful functions for I/O handling operations. 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Loads the YAML config file into a plain dict.

    Args:
        config_path: Path to config/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger instance.

    Args:
        name: Logger name.
        level: Logging level, defaults to INFO.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

