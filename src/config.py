"""Configuration persistence for Heretic Converter."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "heretic_config.json"

PRESETS = {
    "Safe & Surgical": {
        "kl_ceiling": 0.8,
        "refusal_target": 8,
        "target_output_layers_only": True,
    },
    "Balanced": {
        "kl_ceiling": 1.2,
        "refusal_target": 5,
        "target_output_layers_only": False,
    },
    "Aggressive": {
        "kl_ceiling": 2.0,
        "refusal_target": 2,
        "target_output_layers_only": False,
    },
    "Custom": {},
}

DEFAULTS = {
    "preset": "Balanced",
    "kl_ceiling": 1.0,
    "n_trials": 450,
    "refusal_target": 5,
    "target_output_layers_only": True,
    "layer_range_min": 0,
    "layer_range_max": 0,
    "quantization": "None",
    "batch_size": 0,
    "max_response_length": 100,
    "mode": "Decensor",
}


def load_config() -> dict:
    """Load config from disk, merging with defaults for any missing keys."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return config


def save_config(config: dict) -> None:
    """Persist config to disk."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except OSError:
        pass
