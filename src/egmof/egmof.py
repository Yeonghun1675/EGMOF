from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path

from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything

from .prop2desc import Prop2Desc
from .data import Datamodule
from .data.dataset import CSVDataset, TextSplitDataset, JsonSplitDataset


class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""

    def __init__(
        self,
        target: str,
        data_path: str | Path,
        prop2desc_model_config_path: Optional[str | Path] = None,
        prop2desc_training_config_path: Optional[str | Path] = None,
        # ....
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.target = target
        self.data_path = Path(data_path)
        self.prop2desc_model_config_path = prop2desc_model_config_path
        self.prop2desc_training_config_path = prop2desc_training_config_path
        self.overrides = overrides or {}

        self.cfg = self._load_config()
        self.prop2desc: Optional[Prop2Desc] = None
        self.datamodule: Optional[Datamodule] = None
        self.trainer: Optional[Trainer] = None

    def _load_config(self):
        cfgs = []
        if self.prop2desc_model_config_path:
            cfgs.append(OmegaConf.load(self.prop2desc_model_config_path))
        if self.prop2desc_training_config_path:
            cfgs.append(OmegaConf.load(self.prop2desc_training_config_path))
        if cfgs:
            cfg = OmegaConf.merge(*cfgs)
        else:
            cfg = OmegaConf.create({})

        if self.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(self.overrides))
        return cfg

    def _dataset_cls_from_name(self, name: str):
        name = (name or "csv").lower()
        if name == "csv":
            return CSVDataset
        if name in ("text", "txt"):
            return TextSplitDataset
        if name == "json":
            return JsonSplitDataset
        raise ValueError(f"Unknown dataset_cls: {name}. Use one of [csv, text, json].")

    def build_datamodule(self) -> Datamodule:
        dm_cfg = self.cfg.get("datamodule", {})
        dataset_cls = self._dataset_cls_from_name(dm_cfg.get("dataset_cls", "csv"))
        self.datamodule = Datamodule(
            path=self.data_path,
            batch_size=int(dm_cfg.get("batch_size", 64)),
            num_workers=int(dm_cfg.get("num_workers", 4)),
            dataset_cls=dataset_cls,
            task=dm_cfg.get("task", None),
            target=self.target,
        )
        return self.datamodule

    def build_model(self, scaler_value: Optional[Dict[str, Any]] = None) -> Prop2Desc:
        m_cfg = self.cfg.get("model", {})
        if scaler_value is None:
            scaler_value = m_cfg.get("scaler_value", None)
        self.prop2desc = Prop2Desc(
            in_channels=int(m_cfg["in_channels"]),
            timestep=int(m_cfg.get("timestep", 1000)),
            lr=float(m_cfg.get("lr", 1e-4)),
            dim=int(m_cfg.get("dim", m_cfg["in_channels"])),
            dim_mults=list(m_cfg.get("dim_mults", [1, 2])),
            condition=m_cfg.get("condition", None),
            out_channels=m_cfg.get("out_channels", None),
            num_classes=int(m_cfg.get("num_classes", 0)),
            cond_dim=int(m_cfg.get("cond_dim", 0)),
            scaler_mode=m_cfg.get("scaler_mode", "standard"),
            scaler_value=scaler_value,
        )
        return self.prop2desc

    def build_trainer(self) -> Trainer:
        t_cfg = self.cfg.get("trainer", {})
        self.trainer = Trainer(**OmegaConf.to_container(t_cfg, resolve=True))
        return self.trainer

    def train(self):
        """Train the EGMOF model"""
        seed = int(self.cfg.get("seed", 42))
        seed_everything(seed, workers=True)

        dm = self.build_datamodule()
        dm.setup("fit")

        model_cfg = self.cfg.get("model", {})
        scaler_value = model_cfg.get("scaler_value", None)
        if scaler_value is None:
            scaler_mode = model_cfg.get("scaler_mode", "standard")
            if scaler_mode == "standard":
                scaler_value = dm.get_mean_and_std()
            elif scaler_mode == "minmax":
                scaler_value = dm.get_min_and_max()
            else:
                raise ValueError(f"Unknown scaler_mode: {scaler_mode}")

        model = self.build_model(scaler_value=scaler_value)
        trainer = self.build_trainer()
        trainer.fit(model, datamodule=dm)

    def generate(
        self, 
        num_samples: int = 100,
        target: Optional[str] = None,
        output_type: Literal['cif', 'token', "Atoms"] = 'Atoms',
    ) -> Union[str, List[str]]:
        """Generate a new MOF structure"""
        raise NotImplementedError("EGMOF is not implemented yet")

    def load(self):
        """Load the EGMOF model"""
        raise NotImplementedError("EGMOF is not implemented yet")

    def save(self):
        """Save the EGMOF model"""
        raise NotImplementedError("EGMOF is not implemented yet")
