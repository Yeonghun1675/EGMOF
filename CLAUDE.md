# CLAUDE.md — EGMOF Development Guide

> Claude Code agent guide for this repository. See AGENTS.md for the full reference.

## Project Overview

EGMOF (Efficient Generative Model for Metal-Organic Frameworks) — end-to-end generative pipeline for MOF structures using PyTorch Lightning.

**Pipeline**: `Property → [prop2desc] → Descriptors[183] → [desc2mof] → MOF tokens → [mof2desc] → Valid MOFs`

- **Python**: >=3.12 | **Package**: `src/egmof/`
- **Key deps**: PyTorch, Lightning, ASE, RDKit, molSimplify, selfies, omegaconf, pormake

---

## Commands

```bash
# Install
pip install -e .

# Descriptor calculation
python -m egmof.descriptors.get_all_descriptors --cif_dir /path/to/cifs --output out.csv

# desc2mof pretraining
python -m egmof.desc2mof.pretrain --config /path/to/config.yaml --accelerator gpu

# Tests
python -m pytest src/egmof/descriptors/test_*.py -v
python src/egmof/descriptors/test_get_rac.py 0 10

# Main usage
python -c "
from egmof import EGMOF
egmof = EGMOF()
egmof.generate(num_samples=100, target_value=150.0)
"
```

---

## Module Structure

```
src/egmof/
├── __init__.py           # Package root, exports EGMOF + download functions
├── egmof.py              # EGMOF class — main orchestrator
├── train.py              # train_desc2mof, train_mof2desc, dataloader creators
├── utils.py              # create_scaler, load_config, load_feature_names, download_*
├── generate.py           # run_desc2mof, run_mof2desc_and_select, cal_wmse
├── data/                 # LightningDataModule + CSVDataset, TextSplitDataset, JsonSplitDataset
├── prop2desc/            # Property → Descriptor diffusion model (UNet1D)
├── desc2mof/             # Descriptor → MOF tokens (Transformer Encoder-Decoder, beam search)
├── mof2desc/             # MOF tokens → Descriptor (Transformer Encoder, validation)
├── descriptors/          # CIF → 183-D RAC descriptors (molSimplify + Zeo++)
└── builder/              # MOF structure construction (PORMAKE)
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `EGMOF` | `egmof.py` | Top-level orchestrator |
| `Prop2Desc` | `prop2desc/model.py` | Diffusion model, condition on property |
| `Desc2MOF` | `desc2mof/model.py` | Seq2Seq transformer, beam search |
| `MOF2Desc` | `mof2desc/model/model.py` | Encoder for validation |
| `Datamodule` | `data/datamodule.py` | Lightning DataModule |

### Token Vocabulary (desc2mof)
- Special: `[PAD]=0, [SOS]=1, [EOS]=2, [SEP]=3`
- Topology → Metal Nodes → Metal Edges → SELFIES → CN tokens
- MOF name format: `topology+N1+N2+E1+E2`
- Descriptors: 183 RAC features

---

## Code Style

```python
from __future__ import annotations  # always first

# Stdlib → Third-party → Local (alphabetical within each section)
import os
from pathlib import Path

import torch
from lightning import LightningModule

from .data import Datamodule
```

- **Type hints**: Python 3.12+ style — `list[int]`, `str | None` (not `List`, `Optional`)
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants, `_prefix` private
- **Docstrings**: Google-style
- **Inference**: always `model.eval()` + `torch.no_grad()`
- **Error handling**: specific exceptions, never bare `except:`

---

## What NOT To Do

- Never use bare `except:` or empty `except: pass`
- Never delete failing tests
- Never commit unless explicitly requested
- Never suppress type errors with `# type: ignore` without justification
