from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

# import joblib
import numpy as np
import pandas as pd
import yaml
import lightning as pl
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from egmof import mof2desc

from .utils import (
    _require_ckpt,
    download_desc2mof,
    download_mof2desc,
    download_prop2desc,
    download_rf,
    download_all,
)

from .prop2desc import (
    Prop2Desc,
    run_train_prop2desc,
)

from .desc2mof import (
    Desc2MOF as Desc2MOFModel,
    Scaler,
)

from .constants import (
    DEFAULT_DESC2MOF_CKPT,
    DEFAULT_DESC2MOF_CONFIG,
    DEFAULT_DESC2MOF_FEATURE_NAME,
    DEFAULT_DESC2MOF_MEAN,
    DEFAULT_DESC2MOF_STD,
    DEFAULT_MOF2DESC_CKPT,
    DEFAULT_MOF2DESC_CONFIG,
)

from .data import Datamodule
from .data.dataset import CSVDataset, TextSplitDataset, JsonSplitDataset
from .train import train_desc2mof, train_mof2desc
from .utils import create_scaler, load_feature_names, load_config, _load_sk_scaler
from .generate import run_desc2mof, run_mof2desc_and_select


DEFAULT_DESC2MOF_CKPT = os.path.join(
    __root_dir__, "checkpoints", "desc2mof", "desc2mof_best.ckpt"
)
DEFAULT_MOF2DESC_CKPT = os.path.join(
    __root_dir__, "checkpoints", "mof2desc", "mof2desc_best.ckpt"
)
DEFAULT_DESC2MOF_CONFIG = os.path.join(
    __root_dir__, "config", "desc2mof_training_config.yaml"
)
DEFAULT_MOF2DESC_CONFIG = os.path.join(
    __root_dir__, "config", "mof2desc_training_config.yaml"
)
DEFAULT_DESC2MOF_MEAN = os.path.join(__desc2mof_dir__, "data", "mean_all.csv")
DEFAULT_DESC2MOF_STD = os.path.join(__desc2mof_dir__, "data", "std_all.csv")
DEFAULT_DESC2MOF_FEATURE_NAME = os.path.join(
    __desc2mof_dir__, "data", "feature_name.txt"
)


class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""

    def __init__(
        self,
        prop2desc_ckpt_path: Optional[str | Path] = None,
        prop2desc_config_path: Optional[str | Path] = None,
        skmodel_ckpt_dir: Optional[str | Path] = None,
        load_pretrained_modules: bool = True,
        accelerator: Literal["cpu", "cuda"] = "cpu",
        devices: int | List[int] = 1,
    ) -> None:
        self.prop2desc_ckpt_path: Optional[str | Path] = prop2desc_ckpt_path
        self.prop2desc_config_path: Optional[str | Path] = prop2desc_config_path
        self.skmodel_ckpt_dir: Optional[str | Path] = skmodel_ckpt_dir
        self.load_pretrained_modules: bool = load_pretrained_modules
        self.accelerator: Literal["cpu", "cuda"] = accelerator
        self.devices: int | List[int] = devices

        self.prop2desc: Optional[Prop2Desc] = None
        self.desc2mof: Optional[Desc2MOFModel] = None
        self.mof2desc: Optional[object] = None
        self.sk_model: Optional[object] = None
        self._sk_feature_importances: Optional[list[float]] = None
        self._sk_scaler: Optional[object] = None
        self.setup()

    def setup(self):
        if self.load_pretrained_modules:
            self._load_pretrained_desc2mof()
            self._load_pretrained_mof2desc()

        if self.prop2desc_ckpt_path and self.prop2desc_config_path:
            self.load_prop2desc(
                ckpt_path=self.prop2desc_ckpt_path,
                config_path=self.prop2desc_config_path,
            )

        self._load_sk_model()

    def _load_sk_model(self):
        """Load sklearn model and/or scaler with feature importances."""
        if self.skmodel_ckpt_dir:
            skmodel_path = _require_ckpt(
                Path(self.skmodel_ckpt_dir).name,
                Path(self.skmodel_ckpt_dir).parent,
            )
            import joblib

            self.sk_model = joblib.load(skmodel_path)
            self._sk_feature_importances = self.sk_model.feature_importances_.tolist()
        else:
            print(
                "WARNING: skmodel not provided. Guided Decoding (WMSE calculation) will be skipped.\n\n"
                "To download sklearn model:\n"
                "  from egmof import download_rf\n"
                "  download_rf()\n"
                f"See: {ZENODO_RECORD}\n\n"
                "Or use prop2desc_config_path containing 'feature_importances' for WMSE.\n"
            )

        if self.prop2desc_config_path:
            scaler, fi_from_yaml = _load_sk_scaler(self.prop2desc_config_path)
            self._sk_scaler = scaler
            if self._sk_feature_importances is None:
                self._sk_feature_importances = fi_from_yaml

    def _load_pretrained_desc2mof(self):
        self.load_desc2mof(
            ckpt_path=DEFAULT_DESC2MOF_CKPT,
            config_path=DEFAULT_DESC2MOF_CONFIG,
            mean_path=DEFAULT_DESC2MOF_MEAN,
            std_path=DEFAULT_DESC2MOF_STD,
            feature_name_path=DEFAULT_DESC2MOF_FEATURE_NAME,
        )

    def _load_pretrained_mof2desc(self):
        if not hasattr(self, "_desc2mof_scaler") or self._desc2mof_scaler is None:
            self._load_pretrained_desc2mof()
        self.load_mof2desc(
            ckpt_path=DEFAULT_MOF2DESC_CKPT,
            config_path=DEFAULT_MOF2DESC_CONFIG,
            scaler=self._desc2mof_scaler,
        )

    def load_prop2desc(self, ckpt_path: str | Path, config_path: str | Path) -> None:
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"prop2desc config not found: {config_path}\n"
                f"Expected: {config_path}\n\n"
                f"To download:\n"
                f"  from egmof import download_prop2desc\n"
                f"  download_prop2desc()\n\n"
                f"See: {ZENODO_RECORD}"
            )

        if self.prop2desc is not None:
            return
        self.prop2desc = Prop2Desc.load(
            ckpt_path=ckpt_path, config_path=config_path
        ).to(device=self.accelerator)

    def load_desc2mof(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        mean_path: str | Path,
        std_path: str | Path,
        feature_name_path: str | Path,
    ) -> None:
        """Load desc2mof model from checkpoint.

        Args:
            ckpt_path: Path to desc2mof checkpoint (.ckpt)
            config_path: Path to desc2mof config YAML
            mean_path: Path to mean CSV for scaler
            std_path: Path to std CSV for scaler
            feature_name_path: Path to feature name text file
        """
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        mean_path = Path(mean_path)
        std_path = Path(std_path)
        feature_name_path = Path(feature_name_path)

        for path, name in [
            (config_path, "desc2mof config"),
            (mean_path, "desc2mof mean"),
            (std_path, "desc2mof std"),
            (feature_name_path, "feature name"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} not found: {path}\n"
                    f"Expected: {path}\n\n"
                    f"To download:\n"
                    f"  from egmof import download_desc2mof\n"
                    f"  download_desc2mof()\n\n"
                    f"See: {ZENODO_RECORD}"
                )

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load scaler
        with open(feature_name_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]

        mean = pd.read_csv(mean_path)[feature_names]
        std = pd.read_csv(std_path)[feature_names]
        scaler = Scaler(
            np.array(mean).squeeze(),
            np.array(std).squeeze(),
            0,  # target_mean
            1,  # target_std
        )

        # Load model
        self.desc2mof = Desc2MOFModel.load_from_checkpoint(
            ckpt_path,
            config=config,
            strict=False,
            weights_only=False,
        )

        # Store for later use
        self._desc2mof_scaler = scaler
        self._desc2mof_feature_names = feature_names
        self._desc2mof_config = config

    def load_mof2desc(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        scaler: Scaler,
    ) -> None:
        """Load mof2desc model from checkpoint.

        Args:
            ckpt_path: Path to mof2desc checkpoint (.ckpt)
            config_path: Path to mof2desc config YAML
            scaler: Scaler for descriptor normalization (can use desc2mof's scaler)
        """
        ckpt_path = _require_ckpt(
            Path(ckpt_path).name,
            Path(ckpt_path).parent,
        )
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"mof2desc config not found: {config_path}\n"
                f"Expected: {config_path}\n\n"
                f"To download:\n"
                f"  from egmof import download_mof2desc\n"
                f"  download_mof2desc()\n\n"
                f"See: {ZENODO_RECORD}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        from .mof2desc import MOF2Desc as MOF2DescModel

        self.mof2desc = MOF2DescModel.load_from_checkpoint(
            ckpt_path,
            config=config,
            scaler=scaler,
            strict=False,
            weights_only=False,
        )

    def train(
        self,
        data_path: Optional[str | Path] = None,
        task: Optional[str] = None,
        prop2desc_config_path: Optional[str | Path] = None,
        desc2mof_config_path: Optional[str | Path] = None,
        mof2desc_config_path: Optional[str | Path] = None,
    ):
        if data_path and self.prop2desc is None:
            self.train_prop2desc(data_path, task, prop2desc_config_path)
        if self.desc2mof is None:
            self.train_desc2mof(desc2mof_config_path)
        if self.mof2desc is None:
            self.train_mof2desc(mof2desc_config_path)

    def train_prop2desc(
        self,
        data_path: str | Path = None,
        task: Optional[str] = None,
        prop2desc_config_path: Optional[str | Path] = None,
    ) -> Prop2Desc:
        """train prop2desc model"""
        self.prop2desc = run_train_prop2desc(
            config_path=prop2desc_config_path,
            data_path=data_path,
            task=task,
            accelerator=self.accelerator,
            devices=self.devices,
        )
        return self.prop2desc

    def train_desc2mof(
        self,
        config_path: Optional[str | Path] = None,
        train_data_dir: Optional[str | Path] = None,
        val_data_dir: Optional[str | Path] = None,
        test_data_dir: Optional[str | Path] = None,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 200,
        log_dir: str = "./logs_desc2mof",
        ckpt_dir: str = "./ckpt_desc2mof",
    ) -> Desc2MOFModel:
        if config_path is None:
            config_path = DEFAULT_DESC2MOF_CONFIG

        config = load_config(config_path)

        feature_names = load_feature_names(
            config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)
        )

        if hasattr(self, "_desc2mof_scaler") and self._desc2mof_scaler is not None:
            scaler = self._desc2mof_scaler
        else:
            scaler = create_scaler(
                config["mean_dir"],
                config["std_dir"],
                feature_names,
            )

        train_dir = train_data_dir or config.get("train_data_dir")
        val_dir = val_data_dir or config.get("val_data_dir")
        test_dir = test_data_dir or config.get("test_data_dir")

        self.desc2mof = train_desc2mof(
            config=config,
            scaler=scaler,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
        )

        return self.desc2mof

    def train_mof2desc(
        self,
        config_path: Optional[str | Path] = None,
        train_data_dir: Optional[str | Path] = None,
        valid_data_dir: Optional[str | Path] = None,
        test_data_dir: Optional[str | Path] = None,
        accelerator: str = "gpu",
        devices: int = 1,
        max_epochs: int = 500,
        log_dir: str = "./logs_mof2desc",
        ckpt_dir: str = "./ckpt_mof2desc",
    ):
        if config_path is None:
            config_path = DEFAULT_MOF2DESC_CONFIG

        config = load_config(config_path)

        feature_names = load_feature_names(
            config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)
        )

        if hasattr(self, "_desc2mof_scaler") and self._desc2mof_scaler is not None:
            scaler = self._desc2mof_scaler
        else:
            scaler = create_scaler(
                config["mean_dir"],
                config["std_dir"],
                feature_names,
            )

        train_dir = train_data_dir or config.get("train_data_dir")
        valid_dir = valid_data_dir or config.get("valid_data_dir")
        test_dir = test_data_dir or config.get("test_data_dir")

        self.mof2desc = train_mof2desc(
            config=config,
            scaler=scaler,
            train_dir=train_dir,
            valid_dir=valid_dir,
            test_dir=test_dir,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
        )

        return self.mof2desc

    def train_sk_model(self, sk_model_config_path: str | Path):
        # TODO: Train sklearn model. (make train_sklearn.py and call it here.)
        raise NotImplementedError("Training sklearn model is not implemented yet.")

    def generate(
        self,
        num_samples: int = 100,
        target_value: Optional[float | int] = None,
        output_type: Literal["df", "token"] = "df",
        topk: int = 5,
        temperature: float = 1.0,
        wmse_target: float = 0.5,
        batch_size: int = 256,
        num_workers: int = 0,
        save_descriptor_path: Optional[str] = None,
    ) -> Union[pd.DataFrame, List[str]]:
        """Generate MOF structures from target property.

        Pipeline:
        1. prop2desc.sample() → descriptor tensor
        2. desc2mof inference → MOF tokens (beam search)
        3. mof2desc + wmse → select best MOFs

        Args:
            num_samples: Number of MOFs to generate per target
            target_value: Target property value for conditional generation
            output_type: 'df' returns DataFrame, 'token' returns MOF name list
            topk: Beam width for desc2mof beam search
            temperature: Sampling temperature for desc2mof
            wmse_target: WMSE threshold for filtering generated MOFs
            batch_size: Batch size for inference
            num_workers: Num workers for DataLoader
            save_descriptor_path: If provided, save generated descriptors to CSV

        Returns:
            DataFrame with generated MOFs (if output_type='df') or list of MOF names
        """
        if self.prop2desc is None:
            raise ValueError("prop2desc not loaded. Please load or train first.")

        desc_tensor = self.prop2desc.sample(
            num_samples=num_samples,
            target=target_value,
        )
        if len(desc_tensor.shape) == 3:
            desc_tensor = desc_tensor.squeeze(1)

        if save_descriptor_path:
            desc_df = pd.DataFrame(desc_tensor.detach().cpu().numpy())
            desc_df.to_csv(save_descriptor_path, index=False)
            print(f"Descriptors saved to {save_descriptor_path}")

        if self.desc2mof is None or self.mof2desc is None:
            raise ValueError(
                "desc2mof/mof2desc not loaded. Please load or train first."
            )

        device = self.accelerator if self.accelerator in ["cuda", "cpu"] else "cuda"

        all_output, mof_names, target_data = run_desc2mof(
            model=self.desc2mof.model,
            target_desc=desc_tensor,
            scaler=self._desc2mof_scaler,
            feature_names=self._desc2mof_feature_names,
            feature_name_dir=DEFAULT_DESC2MOF_FEATURE_NAME,
            topk=topk,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        weights = getattr(self, "_sk_feature_importances", None)

        if weights is None:
            print(
                "Warning: _sk_feature_importances not found. "
                "Guided Decoding (WMSE calculation) will be skipped."
            )

        desc2mof_scaler = getattr(self, "_desc2mof_scaler", None)

        valid_df, _, log_list = run_mof2desc_and_select(
            model=self.mof2desc,
            all_output=all_output,
            target_data=target_data,
            feature_size=self._desc2mof_config.get("feature_size", 183),
            weights=weights,
            sk_model=self.sk_model,
            sk_scaler=getattr(self, "_sk_scaler", None),
            desc2mof_scaler=desc2mof_scaler,
            topk=topk,
            wmse_target=wmse_target,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        print(f"Generated {len(valid_df)} valid MOFs out of {num_samples}")

        if output_type == "df":
            return valid_df
        elif output_type == "token":
            return valid_df["filename"].tolist()
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
