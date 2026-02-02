"""
Configuration I/O module for Water Drawing App.

Pure functions for loading and saving configuration.
No dependencies on other application modules.
"""

import json
import os


def get_config_path() -> str:
    """Get the path to config.json relative to this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to config.json file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
        print(f"Configuration loaded from {path}")
        return config


def save_config(path: str, config: dict) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        path: Path to config.json file.
        config: Configuration dictionary to save.
        
    Returns:
        True if save was successful, False otherwise.
    """
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False
