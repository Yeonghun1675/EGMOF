# AGENTS.md — Development Guide for EGMOF

> Guide for AI agents working in this repository

## Project Overview

EGMOF (Efficient Generative Model for Metal-Organic Frameworks) is a Python project using PyTorch Lightning.

- **Python**: >=3.12
- **Package**: `src/egmof/`
- **Key deps**: PyTorch, Lightning, ASE, pandas, numpy, omegaconf

---

## Commands

### Installation

```bash
pip install -e .
```

### Running Tests

Tests in `src/egmof/descriptors/test_*.py` (non-standard layout):

```bash
# Direct run with indices
python src/egmof/descriptors/test_get_rac.py <start> <end>

# With pytest
python -m pytest src/egmof/descriptors/test_*.py -v

# Specific test
python -m pytest src/egmof/descriptors/test_get_rac.py -v -k "test_name"
```

### Running Modules

```bash
python -m egmof.descriptors.get_all_descriptors --cif_dir /path/to/cifs --output out.csv
```

---

## Code Style

### Imports (three sections, alphabetically sorted)

```python
from __future__ import annotations  # always first

# Stdlib
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# Third-party
import joblib
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

### Type Hints

Use modern Python 3.12+ syntax:

```python
# Good: def func(x: str, y: int | None = None) -> dict[str, list[int]]:
# Avoid: def func(x: str, y: Optional[int] = None) -> Dict[str, List[int]]:
```

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `get_all_descriptors.py` |
| Classes | PascalCase | `class Datamodule:` |
| Functions/variables | snake_case | `def cal_wmse(...):` |
| Constants | UPPER_SNAKE | `MAX_BATCH_SIZE = 256` |
| Private methods | _snake_case | `def _load_config(self):` |

### Docstrings

Google-style for public APIs:

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

Use specific exceptions, informative messages, avoid bare `except:`.

```python
if cfg_path is None:
    raise ValueError("cfg_path must be provided")
assert model is not None, "Model should be loaded before training"
```

### PyTorch/Lightning Patterns

- Use `torch.no_grad()` for inference
- Use `LightningModule` and `LightningDataModule`
- Use `seed_everything(42)` for reproducibility

```python
class MyModule(LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
```

---

## Project Structure

```
src/egmof/
├── __init__.py           # Package root
├── egmof.py              # Main orchestrator
├── data/                 # LightningDataModule + datasets
├── descriptors/          # RAC + Zeo++ calculator, tests
├── desc2mof/             # Descriptor → MOF model
├── mof2desc/             # MOF → Descriptor model
├── prop2desc/            # Property → Descriptor diffusion
└── builder/              # MOF building blocks
```

---

## Configuration

OmegaConf/YAML:

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
```

---

## Recommendations

1. Add pytest: `[tool.pytest.ini_options]` in pyproject.toml
2. Add type checking: `[tool.mypy]` in pyproject.toml
3. Add linting: `[tool.ruff]` in pyproject.toml
4. Standardize tests: Move to `tests/` directory

---

## Common Patterns

```python
# Load checkpoint
model = MyModel.load_from_checkpoint("path/to/ckpt.ckpt")

# Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```
