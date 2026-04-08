# AGENTS.md — Development Guide for EGMOF

> Guide for AI agents working in this repository

## Project Overview

EGMOF (Efficient Generative Model for Metal-Organic Frameworks) is a Python project using PyTorch Lightning.

- **Python**: >=3.12 | **Package**: `src/egmof/`
- **Key deps**: PyTorch, Lightning, ASE, RDKit, molSimplify, selfies, omegaconf

---

## Commands

### Installation
```bash
pip install -e .
```

### Running Tests
```bash
# Direct run with indices (start/end for slicing)
python src/egmof/descriptors/test_get_rac.py 0 10

# With pytest
python -m pytest src/egmof/descriptors/test_*.py -v

# Specific test by name
python -m pytest src/egmof/descriptors/test_get_rac.py -v -k "test_name"
```

### Running Modules
```bash
# Descriptor calculation
python -m egmof.descriptors.get_all_descriptors --cif_dir /path/to/cifs --output out.csv

# desc2mof pretraining
python -m egmof.desc2mof.pretrain --config /path/to/config.yaml --accelerator gpu

# EGMOF main usage (via Python)
python -c "
from egmof import EGMOF
egmof = EGMOF()  # auto-loads desc2mof, mof2desc
egmof.generate(num_samples=100, target_value=150.0)
"
```

---

## Code Style

### Imports (three sections, sorted alphabetically)
```python
from __future__ import annotations  # always first

# Stdlib
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Local
from .data import Datamodule
from .prop2desc import Prop2Desc
```

### Type Hints (Python 3.12+)
```python
# Good: def func(x: str, y: int | None = None) -> dict[str, list[int]]:
# Avoid: def func(x: str, y: Optional[int] = None) -> Dict[str, List[int]]:
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `get_all_descriptors.py` |
| Classes | PascalCase | `class Datamodule:` |
| Functions/variables | snake_case | `def cal_wmse(...):` |
| Constants | UPPER_SNAKE | `MAX_BATCH_SIZE = 256` |
| Private methods | _snake_case | `def _load_config(self):` |

### Docstrings (Google-style)
```python
def generate(self, num_samples: int = 100, target: float | None = None) -> pd.DataFrame:
    """Generate MOF structures from target property.

    Args:
        num_samples: Number of MOFs to generate
        target: Target property value

    Returns:
        DataFrame with generated MOFs

    Raises:
        RuntimeError: If models not loaded
    """
```

### Error Handling
Use specific exceptions, informative messages. NEVER use bare `except:` or suppress errors.

```python
# Good
if cfg_path is None:
    raise ValueError("cfg_path must be provided")
assert model is not None, "Model should be loaded before training"

# Bad - NEVER do this
try:
    x = data["key"]
except:  # DON'T
    pass
```

---

## PyTorch/Lightning Patterns

```python
# Module pattern
class MyModule(LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

# Inference - ALWAYS use no_grad
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Reproducibility
from lightning import seed_everything
seed_everything(42)

# Checkpoint loading
model = MyModel.load_from_checkpoint("path/to/ckpt.ckpt")

# DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

---

## Project Structure
```
src/egmof/
├── __init__.py           # Package root
├── egmof.py              # Main orchestrator (EGMOF class)
├── train.py              # train_desc2mof, train_mof2desc, dataloader creators
├── utils.py              # create_scaler, load_config, load_feature_names
├── generate.py           # run_desc2mof, run_mof2desc_and_select, cal_wmse
├── egmof_backup.py       # Original implementation (reference)
├── data/                 # LightningDataModule + datasets
├── descriptors/          # RAC + Zeo++ calculator, tests
├── desc2mof/            # Descriptor → MOF model (Transformer)
├── mof2desc/            # MOF → Descriptor model (validation)
├── prop2desc/           # Property → Descriptor diffusion
└── builder/             # MOF building blocks
```

---

## Configuration
OmegaConf/YAML is used throughout:
```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.load("config.yaml", overrides={"model.lr": 1e-3})
```

---

## What NOT To Do
- **Type safety**: Never use `as any`, `@ts-ignore`, `@ts-expect-error`
- **Error suppression**: Never use empty catch blocks `except: pass`
- **Tests**: Never delete failing tests to "make them pass"
- **Commits**: Never commit unless explicitly requested