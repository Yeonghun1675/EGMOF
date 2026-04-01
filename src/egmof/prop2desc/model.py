from __future__ import annotations

from pickle import UnpicklingError
from typing import Any, Dict, List, Literal, Optional

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from lightning.pytorch import LightningModule

from .scheduler import BetaScheduler
from .unet import Unet1D
from .utils import Scaler


class Prop2Desc(LightningModule):
    """Diffusion model for 1D descriptors conditioned on target properties."""

    def __init__(
        self,
        in_channels: int,
        timestep: int,
        lr: float,
        dim: int,
        dim_mults: List[int],
        condition: Literal["numeric", "binary", "class", None] = "numeric",
        out_channels: Optional[int] = None,
        num_classes: int = 0,
        cond_dim: int = 0,
        scaler_mode: Literal["minmax", "standard"] = "standard",
        scaler_value: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.timestep = timestep
        self.lr = lr
        self.dim = dim
        self.dim_mults = dim_mults
        self.condition = condition
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        if self.condition == "class":
            if self.num_classes <= 0 or self.cond_dim <= 0:
                raise ValueError("For condition='class', `num_classes` and `cond_dim` must be > 0.")
            self.cond_embedding: Optional[nn.Embedding] = nn.Embedding(self.num_classes, self.cond_dim)
        else:
            self.cond_embedding = None

        self.model = Unet1D(
            channels=1,
            dim=self.in_channels,
            dim_mults=self.dim_mults,
            condition=self.condition,
        )
        self.scheduler = BetaScheduler(timestep=self.timestep)

        self.scaler: Optional[Scaler]
        if scaler_value is None:
            self.scaler = None
        else:
            self.scaler = Scaler(mode=scaler_mode, **scaler_value)

    def encode(self, desc: torch.Tensor, target: Optional[torch.Tensor] = None):
        """Scale descriptor (and optional target) if a scaler is configured."""
        if self.scaler is None:
            return (desc, target) if target is not None else desc
        return self.scaler.encode(desc, target) if target is not None else self.scaler.encode(batch=desc)

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Scale target if a scaler is configured."""
        if self.scaler is None:
            return target
        return self.scaler.encode(target=target)

    def decode(self, desc: torch.Tensor) -> torch.Tensor:
        """Inverse-scale descriptor if a scaler is configured."""
        if self.scaler is None:
            return desc
        return self.scaler.decode(batch=desc)

    def q_sample(self, x_0: torch.Tensor, batched_t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample `x_t` from `q(x_t | x_0)` by adding Gaussian noise."""
        return self.scheduler.q_sample(x_0, batched_t, noise)

    def parse_descriptor(self, desc: torch.Tensor) -> torch.Tensor:
        """Ensure descriptor tensor has shape [B, 1, padded_D]."""
        if len(desc.shape) == 2:
            desc = desc[:, None, :]

        B, _, dim = desc.shape
        mults = self.dim_mults[-1]
        if n_pad := dim % mults:
            padding = torch.zeros([B, 1, mults - n_pad], device=desc.device, dtype=desc.dtype)
            desc = torch.cat([desc, padding], dim=-1)

        return desc

    def get_cond_batch(self, x_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Build the conditioned UNet input by concatenating/embedding `cond`."""
        B, _, dim = x_t.shape

        if not isinstance(cond, torch.Tensor) or len(cond.shape) == 1:
            if isinstance(cond, torch.Tensor):
                cond = (
                    cond.to(device=x_t.device, dtype=x_t.dtype)
                    .view(B, 1, 1)
                    .expand(-1, 1, dim)
                )
            else:
                cond = torch.full(
                    (B, 1, dim),
                    float(cond),
                    dtype=x_t.dtype,
                    device=x_t.device,
                )
        elif len(cond.shape) == 2 and self.condition in ["numeric", "binary"]:
            cond = repeat(cond, "b c -> b c d", d=dim)
        elif len(cond.shape) == 2 and self.condition in ["class"]:
            if self.cond_embedding is None:
                raise RuntimeError("cond_embedding is not initialized for condition='class'.")
            cond = self.cond_embedding(cond)
        elif len(cond.shape) == 3:
            pass
        else:
            raise ValueError(f"cond shape must be 1, 2, or 3 dims, but got {cond.shape}")

        return torch.concat([x_t, cond], dim=1)

    def forward(self, x_t: torch.Tensor, batched_t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict noise given noised descriptors and timesteps."""
        if self.condition:
            x_t = self.get_cond_batch(x_t, cond)
        return self.model(x_t, batched_t)

    def diffusion_loss(self, desc: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MSE loss between predicted noise and injected noise."""
        desc = self.parse_descriptor(desc)
        B, *_ = desc.shape

        batched_t = self.scheduler.uniform_sample_t(B, self.device)
        noise = torch.randn_like(desc)
        x_t = self.q_sample(desc, batched_t, noise)
        pred_noise = self.forward(x_t, batched_t, cond=cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 8,
        target: Any = None,
        return_trajectory: bool = False,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate descriptors by running reverse diffusion."""
        decode = self.decode

        if self.condition is None:
            s_target = None
        else:
            if isinstance(target, (float, int)):
                target = torch.ones([num_samples, 1], dtype=torch.float, device=self.device) * target
            s_target = self.encode_target(target)

        mults = self.dim_mults[-1]
        if n_pad := self.dim % mults:
            dim = self.dim + (mults - n_pad)
        else:
            dim = self.dim

        if latent is None:
            x_t = torch.randn(num_samples, 1, dim, device=self.device)
            rand_imgs = x_t
        else:
            if latent.shape != (num_samples, 1, dim):
                raise ValueError(f"latent shape must be ({num_samples}, 1, {dim}), but got {latent.shape}")
            x_t = latent
            rand_imgs = latent

        if return_trajectory:
            trajectory = [decode(x_t[:, :, : self.dim])]

        for t in tqdm(range(self.timestep, 0, -1), desc="Sampling"):
            batched_t = torch.full((num_samples,), t, dtype=torch.long, device=self.device)

            alpha = self.scheduler.alphas[batched_t][:, None, None]
            alpha_cumprod = self.scheduler.alphas_cumprod[batched_t][:, None, None]
            sigma = self.scheduler.sigmas[batched_t][:, None, None]

            x_t_uncond = x_t
            if self.condition:
                x_t = self.get_cond_batch(x_t, s_target)

            pred_noise = self.model(x_t, batched_t)
            model_mean = (1 / torch.sqrt(alpha)) * (
                x_t_uncond - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * pred_noise
            )
            x_t = model_mean + sigma * torch.randn_like(rand_imgs)

            if return_trajectory:
                trajectory.append(decode(x_t[:, :, : self.dim]))

        if return_trajectory:
            return trajectory
        return decode(x_t[:, :, : self.dim])

    def training_step(self, batch):
        desc, target = batch
        if self.scaler is None:
            s_desc, s_target = desc, target
        else:
            s_desc, s_target = self.scaler.encode(desc, target)
        loss = self.diffusion_loss(s_desc, s_target)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        desc, target = batch
        if self.scaler is None:
            s_desc, s_target = desc, target
        else:
            s_desc, s_target = self.scaler.encode(desc, target)
        loss = self.diffusion_loss(s_desc, s_target)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        desc, target = batch
        if self.scaler is None:
            s_desc, s_target = desc, target
        else:
            s_desc, s_target = self.scaler.encode(desc, target)
        loss = self.diffusion_loss(s_desc, s_target)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path,
        config_path: str | Path,
    ) -> Prop2Desc:
        try:
            model = cls.load_from_checkpoint(ckpt_path, map_location="cpu", hparams_file=config_path)
            
        except UnpicklingError:   # Load old checkpoint format
            config = OmegaConf.load(config_path)
            model = cls(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                dim=config.dim,
                dim_mults=config.dim_mults,
                condition=config.condition,
                num_classes=config.num_classes,
                cond_dim=config.cond_dim,
                timestep=config.timestep,
                lr=config.lr,
                scaler_mode=config.scaler_mode,
                scaler_value=config.scaler_value,
            )

            state_dict = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False
            )
            model.load_state_dict(state_dict["state_dict"])

        return model