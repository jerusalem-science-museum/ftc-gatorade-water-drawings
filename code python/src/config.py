"""
Configuration I/O module for Water Drawing App.

Pure functions for loading and saving configuration.
No dependencies on other application modules.
"""

import os

import yaml


def get_config_path() -> str:
    """Get the path to config.yaml relative to this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        path: Path to config.yaml file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
        print(f"Configuration loaded from {path}")
        return config


def save_config(path: str, config: dict) -> bool:
    """
    Save configuration to YAML file.

    Args:
        path: Path to config.yaml file.
        config: Configuration dictionary to save.

    Returns:
        True if save was successful, False otherwise.
    """
    try:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Configuration saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False
