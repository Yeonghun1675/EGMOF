# Revised from code:
# 1. https://github.com/hspark1212/diffusion_world/
# 2. https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from typing import Optional

import torch
import torch.nn as nn


def linear_beta_schedule(timesteps, beta_start, beta_end) -> torch.Tensor:
    """Create a linear beta schedule for diffusion."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008) -> torch.Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class BetaScheduler(nn.Module):
    """Precompute diffusion schedule terms for training and sampling."""
    def __init__(
        self,
        timestep: int,
        scheduler_mode: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02, 

    ) -> None:
        super().__init__()
        self.timestep = timestep
        
        betas = self._get_betas(scheduler_mode, timestep, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)
        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )
        sigmas = torch.sqrt(sigmas)

        def _register_buffer(name: str, val: torch.Tensor) -> None:
            self.register_buffer(name, val.to(torch.float32))

        _register_buffer('alphas', alphas)
        _register_buffer('betas', betas)
        _register_buffer('alphas_cumprod', alphas_cumprod)
        _register_buffer('sigmas', sigmas)
                        

    def _get_betas(
        self,
        beta_schedule: str,
        timestep: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Return betas for the requested schedule."""
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timestep, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timestep)
        else:
            raise ValueError(f'unknown beta schedule: {beta_schedule}')
        return torch.cat([torch.zeros([1]), betas], dim=0)
        
    def q_sample(
        self, 
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) by adding noise."""
        alpha_cumprod = self.alphas_cumprod[t][:, None, None]

        return (
            torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt((1-alpha_cumprod)) * noise
        )
 
    def uniform_sample_t(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Uniformly sample timesteps t in the range [1, timestep] (inclusive)."""
        return torch.randint(
            low=1,
            high=self.timestep + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )