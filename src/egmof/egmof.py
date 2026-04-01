from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import selfies
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from egmof import mof2desc

from .prop2desc import (
    Prop2Desc,
    train_prop2desc,
)

from .desc2mof import (
    Desc2MOF as Desc2MOFModel,
    MOFGenDataset,
    Scaler,
    MOF_ENCODE_DICT,
    MOF_DECODE_DICT,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    CN_IDS,
    decode_token2mof,
    __desc2mof_dir__,
)
from . import __root_dir__
from .data import Datamodule
from .data.dataset import CSVDataset, TextSplitDataset, JsonSplitDataset



class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""
    def __init__(
        self,
        data_path: str | Path,
        target_property: Optional[str] = None,
        load_pretrained_modules: bool = True,
        accelerator: Literal["cpu", "cuda"] = "cpu",
        devices: int | List[int] = 1,
        prop2desc_ckpt_path: Optional[str | Path] = None,
        prop2desc_config_path: Optional[str | Path] = None,
    ) -> None:
        self.data_path: Path = Path(data_path) if data_path else None
        self.target_property: Optional[str] = target_property
        self.load_pretrained_modules: bool = load_pretrained_modules
        self.prop2desc: Optional[Prop2Desc] = None
        self.desc2mof: Optional[Desc2MOFModel] = None
        self.mof2desc: Optional[object] =  None   # TODO: Implement Type
        self.sk_model: Optional[object] = None    # TODO: Implement Type
        self.setup()


    def setup(self):
        if self.load_pretrained_modules:
            # TODO: Load desc2mof and mof2desc model from checkpoint if available.
            raise NotImplementedError("Loading pretrained modules is not implemented yet.")

        # Dataset
        raise NotImplementedError("Data module is not implemented yet.")

    
    def train(
        self,
        prop2desc_config_path: Optional[str | Path] = None,
        desc2mof_config_path: Optional[str | Path] = None,
        mof2desc_config_path: Optional[str | Path] = None,
    ):
        # TODO: Train prop2desc model.
        if self.prop2desc is None:
            self.train_prop2desc(prop2desc_config_path)
        if self.desc2mof is None:
            self.train_desc2mof(desc2mof_config_path)
        if self.mof2desc is None:
            self.train_mof2desc(mof2desc_config_path)

    def train_prop2desc(self, prop2desc_config_path: str | Path) -> None:
        if self.data_path is None:
            raise ValueError("data_path must be provided to train prop2desc.")

        if prop2desc_config_path is None:
            prop2desc_config = OmegaConf.merge(
                OmegaConf.load(Path(__root_dir__) / "config" / "prop2desc_model_config.yaml"),
                OmegaConf.load(Path(__root_dir__) / "config" / "prop2desc_training_config.yaml"),
            )
        else:
            prop2desc_config = Path(prop2desc_config_path)

        self.prop2desc = train_prop2desc(
            config_path=prop2desc_config,
            data_path=self.data_path,
            target=self.target_property,
        )

    def train_desc2mof(self, desc2mof_config_path: str | Path):
        # TODO: Train desc2mof model. (make desc2mof/train.py and call it here.)
        raise NotImplementedError("Training desc2mof is not implemented yet.")

    def train_mof2desc(self, mof2desc_config_path: str | Path):
        # TODO: Train mof2desc model. (make mof2desc/train.py and call it here.)
        raise NotImplementedError("Training mof2desc is not implemented yet.")

    def train_sk_model(self, sk_model_config_path: str | Path):
        # TODO: Train sklearn model. (make train_sklearn.py and call it here.)
        raise NotImplementedError("Training sklearn model is not implemented yet.")

    def generate(self):
        if not self.prop2desc:
            raise ValueError("prop2desc model is not trained yet.")
        if not self.desc2mof:
            raise ValueError("desc2mof model is not trained yet.")
        if not self.mof2desc:
            raise ValueError("mof2desc model is not trained yet.")
        if not self.sk_model:
            raise ValueError("sklearn model is not trained yet.")

        # TODO: Generate descriptors using desc2mof model. (make desc2mof/generate.py and call it here.)
        pass

        # TODO: Generate descriptors using mof2desc model. (make mof2desc/generate.py and call it here.)
        pass

        # TODO: Generate descriptors using sklearn model. (make train_sklearn.py and call it here.)
        pass

        # TODO: Generate descriptors using prop2desc model. (make prop2desc/generate.py and call it here.)
        pass

        raise NotImplementedError("Generating descriptors is not implemented yet.")
