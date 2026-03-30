# Builder

Tools for generating and managing MOF building blocks (BBs).

## make_bbs.py

Generate building block XYZ files from a single SELFIES string.

### Usage

```bash
# Single SELFIES string
python -m egmof.builder.make_bbs "[H][C][=C][Branch1][C][H][CH0][Branch1][C][Lr][=C][C][=Branch1][Branch1][=CH0][Ring1][Branch2][Lr][C][Branch1][C][H][Branch1][C][H][C][Ring1][Branch2][Branch1][C][H][H]" --engine xtb

# Specify output directory
python -m egmof.builder.make_bbs "[H][C][=C][Branch1][C][H][CH0][Branch1][C][Lr][=C][C][=Branch1][Branch1][=CH0][Ring1][Branch2][Lr][C][Branch1][C][H][Branch1][C][H][C][Ring1][Branch2][Branch1][C][H][H]" --run_dir /path/to/output --engine xtb
```

### Arguments

| Argument    | Description         | Default           |
| ----------- | ------------------- | ----------------- |
| `selfies`   | SELFIES string      | required          |
| `--run_dir` | Output directory    | `builder/new_bbs` |
| `--engine`  | Optimization engine | `xtb`             |

### Engine Options

- `xtb` — GFN-xTB tight optimization (recommended)
- `mmff` — MMFF94 force field (no external dependency)
- `uff` — UFF force field (no external dependency)

### Output Naming

- `[Lr] count = 2` → `Custom_E{n}.xyz` (Edge)
- `[Lr] count > 2` → `Custom_N{n}.xyz` (Node)

Counters auto-increment from existing files in `--run_dir`.

### xTB Auto-Install

If `xtb` engine is selected and the binary is not found, it will be auto-downloaded from GitHub.

## selfies2bb.py

Low-level SELFIES → XYZ conversion functions. Used by `make_bbs.py`.

### Functions

- `decode_selfies_to_xyz_opt()` — Main entry point
- `selfies_to_mol()` — SELFIES → RDKit molecule
- `run_xtb_tight()` — xTB geometry optimization
- `run_mmff94_opt()` — MMFF94 optimization
- `run_uff_opt()` — UFF optimization

### Usage

```python
from egmof.builder.make_bbs import make_bb

result = make_bb(
    "[H][C][=C][Branch1][C][H][CH0][Branch1][C][Lr][=C][C][=Branch1][Branch1][=CH0][Ring1][Branch2][Lr][C][Branch1][C][H][Branch1][C][H][C][Ring1][Branch2][Branch1][C][H][H]",
    #run_dir="./new_bbs/", default
    #engine="xtb",
    #xtb_bin = "./xtb-dist/bin/" or /your/xtb/path/
   
)
print(result)  #True of False, path to output XYZ
```
```bash
python make_bbs.py  "[H][C][=C][Branch1][C][H][CH0][Branch1][C][Lr][=C][C][=Branch1][Branch1][=CH0][Ring1][Branch2][Lr][C][Branch1][C][H][Branch1][C][H][C][Ring1][Branch2][Branch1][C][H][H]"
```

### Results
```python
import os
from egmof.desc2mof.preprocessing import read_extended_xyz,  build_rdkit_mol
from egmof import __root_dir__
run_dir = os.path.join(__root_dir__, "builder", "new_bbs")
atoms, bonds = read_extended_xyz(f'{run_dir}/Custom_E1.xyz')
mol  = build_rdkit_mol (atoms, bonds)

```