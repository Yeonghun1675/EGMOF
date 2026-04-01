from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Type, Optional

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from egmof.data import Datamodule
from egmof.data.dataset import CSVDataset, JsonSplitDataset, TextSplitDataset

from .model import Prop2Desc
from ..constants import DEFAULT_PROP2DESC_CONFIG


DatasetType = Type[CSVDataset | TextSplitDataset | JsonSplitDataset]


def _dataset_cls_from_name(name: str) -> DatasetType:
    name = (name or "json").lower()
    if name == "csv":
        return CSVDataset
    if name in {"text", "txt"}:
        return TextSplitDataset
    if name == "json":
        return JsonSplitDataset
    raise ValueError(f"Unknown dataset_cls: {name}. Use one of [csv, text, json].")


def _build_datamodule(cfg: Any, data_path: Optional[str | Path], task: Optional[str | None]) -> Datamodule:
    dm_cfg = cfg.datamodule
    dataset_cls = _dataset_cls_from_name(dm_cfg.dataset_cls)
    
    if data_path:
        dm_cfg.data_path = data_path
    if task:
        dm_cfg.task = task

    return Datamodule(
        path=dm_cfg.data_path,
        batch_size=int(dm_cfg.batch_size),
        num_workers=int(dm_cfg.num_workers),
        dataset_cls=dataset_cls,
        task=dm_cfg.task,
        target=task,
    )

# TODO: check the name that if it is not in DEFULAT_PROP2DESC_TRAINING_CONFIG.yaml
def _build_model(cfg: Any, scaler_value: dict[str, Any] | None) -> Prop2Desc:
    model_cfg = cfg.model
    if scaler_value:
        model_cfg.scaler_value = scaler_value

    return Prop2Desc(
        in_channels=int(model_cfg.in_channels),
        timestep=int(model_cfg.timestep),
        lr=float(model_cfg.lr),
        dim=int(model_cfg.dim),
        dim_mults=list(model_cfg.dim_mults),
        condition=model_cfg.condition,
        out_channels=model_cfg.out_channels,
        num_classes=model_cfg.num_classes,
        cond_dim=model_cfg.cond_dim,
        scaler_mode=model_cfg.scaler_mode,
        scaler_value=model_cfg.scaler_value,
    )


def _resolve_scaler_value(cfg: Any, datamodule: Datamodule) -> dict[str, Any] | None:
    model_cfg = cfg.get("model", {})
    scaler_value = model_cfg.get("scaler_value", None)
    if scaler_value is not None:
        return OmegaConf.to_container(scaler_value, resolve=True)

    scaler_mode = model_cfg.get("scaler_mode", "standard")
    if scaler_mode == "standard":
        return datamodule.get_mean_and_std()
    if scaler_mode == "minmax":
        return datamodule.get_min_and_max()
    raise ValueError(f"Unknown scaler_mode: {scaler_mode}")


def _build_loggers(cfg: Any, log_dir: str | Path | None = None) -> list[Any]:
    logger_cfg = cfg.logger
    if log_dir:
        logger_cfg.log_dir = log_dir

    if not logger_cfg.exp_name:
        logger_cfg.exp_name = cfg.datamodule.task

    logger_tb = TensorBoardLogger(
        save_dir = logger_cfg.log_dir,
        name = logger_cfg.exp_name
    )

    logger_wandb = WandbLogger(
        project = logger_cfg.project,
        name = f"{logger_cfg.exp_name}_version_{logger_tb.version}",
        save_dir = logger_cfg.log_dir,
    )

    return [logger_tb, logger_wandb]


def _build_trainer(cfg: Any, log_dir: str | Path | None, ckpt_dir: str | Path | None) -> Trainer:
    trainer_cfg = cfg.trainer

    callbacks = []
    if ckpt_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )
    if log_dir is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    existing_callbacks = trainer_cfg.pop("callbacks", None)
    if existing_callbacks:
        callbacks.extend(existing_callbacks)

    loggers = _build_loggers(cfg, log_dir)

    return Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        num_nodes=trainer_cfg.num_nodes,
        precision=trainer_cfg.precision,
        benchmark=True,
        max_epochs=trainer_cfg.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        val_check_interval=trainer_cfg.val_check_interval,
    )


def run_train_prop2desc(
    config_path: str | Path | Any,
    data_path: str | Path,
    task: str | None,
    log_dir: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
    test_only: bool = False,
    accelerator: str | None = None,
    devices: str | None = None,
) -> Prop2Desc:

    config_path = config_path if isinstance(config_path, str | Path) else DEFAULT_PROP2DESC_CONFIG

    cfg = OmegaConf.load(config_path)
    seed_everything(int(cfg.seed), workers=True)

    if accelerator:
        cfg.trainer.accelerator = accelerator
    if devices:
        cfg.trainer.devices = devices    

    datamodule = _build_datamodule(cfg, data_path=data_path, task=task)
    datamodule.setup("test" if test_only else "fit")

    if not test_only:
        datamodule.setup("fit")
    scaler_value = _resolve_scaler_value(cfg, datamodule)
    model = _build_model(cfg, scaler_value=scaler_value)

    trainer = _build_trainer(cfg, log_dir=log_dir, ckpt_dir=ckpt_dir)
    ckpt_path = cfg.get("ckpt_path", None)

    if test_only:
        datamodule.setup("test")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        return model

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the prop2desc diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing train/val/test splits.")
    parser.add_argument("--target", type=str, default=None, help="Target property column name.")
    parser.add_argument("--log_dir", type=str, default=None, help="Optional Lightning log directory.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Optional checkpoint directory.")
    parser.add_argument("--test_only", action="store_true", help="Run test instead of training.")
    args = parser.parse_args()

    run_train_prop2desc(
        config_path=Path(args.config),
        data_path=Path(args.data_path),
        target=args.target,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
