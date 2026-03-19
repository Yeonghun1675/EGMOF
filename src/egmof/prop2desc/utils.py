# Revised from code:
# 1. https://github.com/hspark1212/diffusion_world/
# 2. https://github.com/lucidrains/denoising-diffusion-pytorch

from typing import Optional, Any, Dict
import torch
import torch.nn as nn


def exists(x):
    """Return True if `x` is not None."""
    return x is not None


def default(val, d):
    """Return `val` if not None, otherwise `d` (call `d()` if callable)."""
    if exists(val):
        return val
    return d() if callable(d) else d


def to_tensor(t: Optional[Any]) -> Optional[torch.Tensor]:
    """Convert a python scalar / array-like to a float32 tensor (or pass through tensors)."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.to(dtype=torch.float32)
    return torch.tensor(t, dtype=torch.float32)
    


class Scaler(nn.Module):
    """Feature scaler for diffusion inputs/targets (standard or min-max)."""
    def __init__(
            self,
            mode: str,
            eps: float = 1e-6,
            **kwargs: Dict[str, Any],
        ) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer('eps', to_tensor(eps))
        if mode == 'standard':
            self.register_buffer('mean', to_tensor(kwargs["mean"]))
            self.register_buffer('std', to_tensor(kwargs["std"]))
            self.register_buffer('target_mean', to_tensor(kwargs["target_mean"]))
            self.register_buffer('target_std', to_tensor(kwargs["target_std"]))
        elif mode == 'minmax':
            self.register_buffer('min', to_tensor(kwargs["min"]))
            self.register_buffer('max', to_tensor(kwargs["max"]))
            self.register_buffer('target_min', to_tensor(kwargs["target_min"]))
            self.register_buffer('target_max', to_tensor(kwargs["target_max"]))
        else:
            raise ValueError(f'mode must be one of [standard, minmax], not {mode}')
            

    def encode(self, batch = None, target = None):
        """Scale `batch` and/or `target` into the normalized space."""
        assert self.mode in ['standard', 'minmax']

        if target is None:
            if batch is None:
                raise ValueError('One of the batch and target must not be None')
            s_batch = getattr(self, f'{self.mode}_encode')(batch)
            return s_batch
        elif batch is None:
            t_batch = getattr(self, f'{self.mode}_encode_target')(target)
            return t_batch  
        else:
            s_batch = getattr(self, f'{self.mode}_encode')(batch)
            t_batch = getattr(self, f'{self.mode}_encode_target')(target)
            return s_batch, t_batch

    def decode(self, batch = None, target = None):
        """Inverse-scale `batch` and/or `target` back to original space."""
        assert self.mode in ['standard', 'minmax']
        
        if target is None:
            if batch is None:
                raise ValueError('One of the batch and target must not be None')
            s_batch = getattr(self, f'{self.mode}_decode')(batch)
            return s_batch
        elif batch is None:
            t_batch = getattr(self, f'{self.mode}_decode_target')(target)
            return t_batch  
        else:
            s_batch = getattr(self, f'{self.mode}_decode')(batch)
            t_batch = getattr(self, f'{self.mode}_decode_target')(target)
            return s_batch, t_batch

    def standard_encode(self, batch):
        if isinstance(batch, torch.Tensor):
            return (batch - self.mean) / (self.std + self.eps)
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            eps = self.eps.item()
            return (batch - mean) / (std + eps)
                
    def standard_encode_target(self, target):
        return (target - self.target_mean) / (self.target_std + self.eps)
    
    def minmax_encode(self, batch):
        if isinstance(batch, torch.Tensor):
            return (batch - self.min) / (self.max - self.min + self.eps)
        else:
            min, max = self.min.cpu().numpy(), self.max.cpu().numpy()
            eps = self.eps.item()
            return (batch - min) / (max - min + eps)

    def minmax_encode_target(self, target):
        return (target - self.target_min) / (self.target_max - self.target_min + self.eps)
    
    def standard_decode(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch * (self.std + self.eps) + self.mean
        else:
            mean, std = self.mean.cpu().numpy(), self.std.cpu().numpy()
            eps = self.eps.item()
            return batch * (std + eps) + mean
    
    def standard_decode_target(self, target):
        return target * (self.target_std + self.eps) + self.target_mean
    
    def minmax_decode(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch * (self.max - self.min + self.eps) + self.min
        else:
            min, max = self.min.cpu().numpy(), self.max.cpu().numpy()
            eps = self.eps.item()
            return batch * (max - min + eps) + min
    
    def minmax_decode_target(self, target):
        return target * (self.target_max - self.target_min + self.eps) + self.target_min
