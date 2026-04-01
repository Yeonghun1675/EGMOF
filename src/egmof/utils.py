import numpy as np
import pandas as pd
from pathlib import Path
from .desc2mof import Scaler


def create_scaler(
    mean_path: str | Path,
    std_path: str | Path,
    feature_names: list[str],
    target_mean: float = 0,
    target_std: float = 1,
) -> Scaler:
    """Create Scaler from mean/std CSV files.

    Args:
        mean_path: Path to mean CSV
        std_path: Path to std CSV
        feature_names: List of feature column names
        target_mean: Target normalization mean
        target_std: Target normalization std

    Returns:
        Scaler instance
    """
    mean = pd.read_csv(mean_path)[feature_names]
    std = pd.read_csv(std_path)[feature_names]
    return Scaler(
        np.array(mean).squeeze(),
        np.array(std).squeeze(),
        target_mean,
        target_std,
    )


def load_feature_names(feature_name_path: str) -> list[str]:
    """Load feature names from text file.

    Args:
        feature_name_path: Path to feature name text file

    Returns:
        List of feature names
    """
    with open(feature_name_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config

    Returns:
        Config dict
    """
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
