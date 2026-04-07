import hashlib
import os
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import yaml
from pathlib import Path
from tqdm.auto import tqdm

from . import __root_dir__
from .desc2mof import __desc2mof_dir__, Scaler

ZENODO_RECORD = "https://zenodo.org/records/19362907"
ZENODO_BASE = f"{ZENODO_RECORD}/files"
ZENODO_FILES = {
    "desc2mof_best.ckpt": {
        "url": f"{ZENODO_BASE}/desc2mof_best.ckpt",
        "md5": "afcf6e3610e50ca48f18dd11a5146bef",
    },
    "mof2desc_best.ckpt": {
        "url": f"{ZENODO_BASE}/mof2desc_best.ckpt",
        "md5": "a711a747a12cd8e3c1d3bc8100fa3cba",
    },
    "prop2desc_ckpt.zip": {
        "url": f"{ZENODO_BASE}/prop2desc_ckpt.zip",
        "md5": "42eee2ea274fe86a2c476bae61b98fe2",
    },
    "rf_ckpt.zip": {
        "url": f"{ZENODO_BASE}/rf_ckpt.zip",
        "md5": "d058aa691e640af4f1e193f379bdfac3",
    },
}


def _download_with_progress(url: str, dest: Path, md5: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    if md5:
        actual_md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual_md5 != md5:
            dest.unlink()
            raise ValueError(f"MD5 mismatch: expected {md5}, got {actual_md5}")


DEFAULT_DESC2MOF_FEATURE_NAME = os.path.join(
    __desc2mof_dir__, "data", "feature_name.txt"
)
=======
import yaml
from .desc2mof import Scaler
>>>>>>> origin/main


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


def _load_sk_scaler(config_path: str | Path) -> tuple[Scaler, list[float] | None]:
    """Load scaler and feature_importances from config (YAML/JSON)."""
    path = str(config_path)
    if path.endswith(".json"):
        import json

        with open(path, "r") as f:
            yaml_data = json.load(f)
    elif path.endswith(".yaml"):
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}")

    feature_importances = yaml_data.get("feature_importances", None)
    scaler_dict = yaml_data.get("scaler_value", yaml_data)

    scaler = Scaler(
        scaler_dict["mean"],
        scaler_dict["std"],
        scaler_dict["target_mean"],
        scaler_dict["target_std"],
    )

    return scaler, feature_importances


DEFAULT_DESC2MOF_CKPT = os.path.join(
    __root_dir__, "checkpoints", "desc2mof", "desc2mof_best.ckpt"
)
DEFAULT_MOF2DESC_CKPT = os.path.join(
    __root_dir__, "checkpoints", "mof2desc", "mof2desc_best.ckpt"
)


def _require_ckpt(filename: str, default_dir: Path) -> Path:
    default_path = default_dir / filename
    if default_path.exists():
        return default_path

    func_name = filename.rsplit(".", 1)[0]
    raise FileNotFoundError(
        f"{filename} not found at {default_path}\n\n"
        f"To download:\n"
        f"  from egmof import download_{func_name}\n"
        f"  download_{func_name}()\n\n"
        f"Or download all:\n"
        f"  from egmof import download_all\n"
        f"  download_all()\n\n"
        f"See: {ZENODO_RECORD}"
    )


def download_desc2mof(dest_dir: str | Path | None = None) -> Path:
    dest_dir = Path(dest_dir or os.path.dirname(DEFAULT_DESC2MOF_CKPT))
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = "desc2mof_best.ckpt"
    info = ZENODO_FILES[filename]
    print(f"[INFO] Downloading {filename} from Zenodo...")
    _download_with_progress(info["url"], dest_dir / filename, info["md5"])
    return dest_dir / filename


def download_mof2desc(dest_dir: str | Path | None = None) -> Path:
    dest_dir = Path(dest_dir or os.path.dirname(DEFAULT_MOF2DESC_CKPT))
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = "mof2desc_best.ckpt"
    info = ZENODO_FILES[filename]
    print(f"[INFO] Downloading {filename} from Zenodo...")
    _download_with_progress(info["url"], dest_dir / filename, info["md5"])
    return dest_dir / filename


def download_prop2desc(dest_dir: str | Path | None = None) -> Path:
    dest_dir = Path(dest_dir or os.path.join(__root_dir__, "checkpoints"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = "prop2desc_ckpt.zip"
    info = ZENODO_FILES[filename]
    zip_path = dest_dir / filename
    print(f"[INFO] Downloading {filename} from Zenodo...")
    _download_with_progress(info["url"], zip_path, info["md5"])
    print(f"[INFO] Extracting {filename}...")
    with ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    zip_path.unlink()
    extracted = dest_dir / "prop2desc_ckpt"
    subdir = extracted if extracted.is_dir() else dest_dir / "prop2desc"
    ckpt_files = list(subdir.glob("*.ckpt")) + list(subdir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {subdir}")
    target = dest_dir / "prop2desc_best.ckpt"
    ckpt_files[0].rename(target)
    if subdir.is_dir():
        for f in subdir.glob("*"):
            f.unlink()
        subdir.rmdir()
    print(f"[OK] Saved to {target}")
    return target


def download_rf(dest_dir: str | Path | None = None) -> Path:
    dest_dir = Path(dest_dir or os.path.join(__root_dir__, "checkpoints"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = "rf_ckpt.zip"
    info = ZENODO_FILES[filename]
    zip_path = dest_dir / filename
    print(f"[INFO] Downloading {filename} from Zenodo...")
    _download_with_progress(info["url"], zip_path, info["md5"])
    print(f"[INFO] Extracting {filename}...")
    with ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    zip_path.unlink()
    extracted = dest_dir / "rf_ckpt"
    subdir = extracted if extracted.is_dir() else dest_dir / "rf"
    ckpt_files = list(subdir.glob("*.pkl")) + list(subdir.glob("*.joblib"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model found in {subdir}")
    target = dest_dir / "rf_ckpt.pkl"
    ckpt_files[0].rename(target)
    if subdir.is_dir():
        for f in subdir.glob("*"):
            f.unlink()
        subdir.rmdir()
    print(f"[OK] Saved to {target}")
    return target


def download_all(dest_dir: str | Path | None = None) -> dict[str, Path]:
    dest_dir = Path(dest_dir or __root_dir__)
    return {
        "desc2mof": download_desc2mof(dest_dir / "checkpoints" / "desc2mof"),
        "mof2desc": download_mof2desc(dest_dir / "checkpoints" / "mof2desc"),
        "prop2desc": download_prop2desc(dest_dir / "checkpoints"),
        "rf": download_rf(dest_dir / "checkpoints"),
    }
=======
>>>>>>> origin/main
