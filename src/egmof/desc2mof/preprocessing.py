"""
Preprocessing script for PORMAKE building blocks.
Use this script to preprocess the building blocks for the desc2mof model.

e.g.) python preprocessing.py --output 'test_bbs'
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import selfies
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdmolops
from egmof.desc2mof import __desc2mof_dir__

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def _find_pormake_bb_dir() -> Path | None:
    try:
        import importlib.util
        import pormake as pm
    except ImportError:
        return None
    spec = importlib.util.find_spec("pormake")
    if spec and spec.origin:
        bb_dir = Path(spec.origin).parent / "database" / "bbs"
        if bb_dir.exists():
            return bb_dir
    return Path(pm.__file__).parent / "database" / "bbs"


BB_DIR = _find_pormake_bb_dir()


BOND_MAP = {
    "S": Chem.BondType.SINGLE,
    "D": Chem.BondType.DOUBLE,
    "T": Chem.BondType.TRIPLE,
    "A": Chem.BondType.AROMATIC,
}
PLACEHOLDERS = "Lr"

METALS = [
    # Alkali metals
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Fr",
    # Alkaline earth metals
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ra",
    # Transition metals
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    # Lanthanides
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    # Actinides
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    # Post-transition (often treated as metals in materials contexts)
    "Al",
    "Ga",
    "In",
    "Sn",
    "Tl",
    "Pb",
    "Bi",
    # added like metal node
    "Ge",
    "Po",
    "Sb",
    "Si",
    "As",
    "B",
    "Te",
]

FAILED_IMPLICIT_LIST = {"E115", "E161", "E174", "E229"}


def bb_id_from_fname(fname: str) -> str:
    return os.path.splitext(fname)[0]


def fname_from_bb_id(bb_id: str) -> str:
    return f"{bb_id}.xyz"


def has_metal_atoms(
    atoms,
    metals=METALS,
    return_metals=False,
):
    found = sorted({a for a in atoms if a in metals})
    ok = len(found) > 0
    return (ok, found) if return_metals else ok


def count_X(atoms):
    return atoms.count("X")


def read_extended_xyz(path: str) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Read extended XYZ file format.

    Returns:
        atoms: list of element symbols
        bonds: list of (i, j, bond_type_char)
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    n_atoms = int(lines[0])

    atom_lines = lines[2 : 2 + n_atoms]
    bond_lines = lines[2 + n_atoms :]

    atoms = [l.split()[0] for l in atom_lines]

    bonds = []
    for l in bond_lines:
        i, j, b = l.split()
        bonds.append((int(i), int(j), b))

    return atoms, bonds


def build_rdkit_mol(atoms: list[str], bonds: list[tuple[int, int, str]]) -> Chem.Mol:
    """Build RDKit molecule from atoms and bonds.

    Args:
        atoms: List of element symbols
        bonds: List of (i, j, bond_type) tuples

    Returns:
        RDKit Mol object
    """
    mol = Chem.RWMol()

    # Add atoms
    for elem in atoms:
        if elem == "X":
            atom = Chem.Atom(PLACEHOLDERS)  # dummy -> Lr
        else:
            atom = Chem.Atom(elem)
        mol.AddAtom(atom)

    # Add bonds
    for i, j, btype in bonds:
        if btype not in BOND_MAP:
            raise ValueError(f"Unknown bond type: {btype}")

        bt = BOND_MAP[btype]
        mol.AddBond(i, j, bt)

        # Set aromatic flag on atoms for aromatic bonds
        if btype == "A":
            mol.GetAtomWithIdx(i).SetIsAromatic(True)
            mol.GetAtomWithIdx(j).SetIsAromatic(True)

    mol = mol.GetMol()

    # Sanitize without kekulize
    rdmolops.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
    )

    return mol


def preprocess_building_blocks(
    bb_dir: str | Path | None = None,
    save_dir: str | Path = "",
) -> None:
    """Process PORMAKE building blocks and save mappings.

    Args:
        bb_dir: Path to PORMAKE BB directory (auto-detected if None)
        save_dir: Directory to save output files (default:"",  for pretrain: {__desc2mof_dir__}/data/)
    """
    if bb_dir is None:
        if BB_DIR is None:
            raise ValueError(
                "PORMAKE not installed. Install with: pip install pormake\n"
                "Or provide bb_dir explicitly."
            )
        bb_dir = str(BB_DIR)

    results = []
    selfies2bb: dict[str, str] = {}
    bb2selfies: dict[str, str] = {}
    metal_bb_list: list[str] = []
    bb_cn_dict: dict[str, int] = {}
    bb_cn_dict["E0"] = 2
    failed: list[tuple[str, str]] = []

    xyz_files = sorted(f for f in os.listdir(bb_dir) if f.endswith(".xyz"))
    print(f"Found {len(xyz_files)} *.xyz files in {bb_dir}")

    for fname in xyz_files:
        try:
            bb_id = bb_id_from_fname(fname)
            path = os.path.join(bb_dir, fname)
            atoms, bonds = read_extended_xyz(path)
            cn = count_X(atoms)

            bb_cn_dict[bb_id] = cn

            if bb_id in FAILED_IMPLICIT_LIST:
                failed.append((fname, "implicit"))
                continue
            if has_metal_atoms(atoms):
                metal_bb_list.append(bb_id)
                continue
            mol = build_rdkit_mol(atoms, bonds)

            smiles = Chem.MolToSmiles(mol)

            selfies_str = selfies.encoder(smiles)
            tokens = list(selfies.split_selfies(selfies_str))

            results.append((fname, smiles, selfies_str, len(tokens), tokens))

            selfies2bb[selfies_str] = bb_id
            bb2selfies[bb_id] = selfies_str
        except Exception as e:
            failed.append((fname, str(e)))

    failed_bb_list = [bb[:-4] for bb, err in failed]

    token_set = set()
    for res in results:
        token_set.update(set(res[-1]))

    # Metal nodes (N*) and edges (E*)
    metal_node_list = [bb for bb in metal_bb_list if bb.startswith("N")]
    metal_node_list = sorted(metal_node_list, key=lambda x: int(x[1:]))

    metal_edge_list = ["E0"] + [bb for bb in metal_bb_list if bb.startswith("E")]
    metal_edge_list = sorted(metal_edge_list, key=lambda x: int(x[1:]))

    special_tkns = [f"[CN_{i}]" for i in set(bb_cn_dict.values())]

    # Save to files
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "cn.txt", "w") as g:
        g.write("\n".join(special_tkns))

    with open(save_dir / "metal_node.txt", "w") as g:
        g.write("\n".join(metal_node_list))

    with open(save_dir / "metal_edge.txt", "w") as g:
        g.write("\n".join(metal_edge_list))

    with open(save_dir / "selfies.txt", "w") as f:
        f.write("\n".join(sorted(token_set)))

    with open(save_dir / "failed_bb_list.txt", "w") as g:
        g.write("\n".join(failed_bb_list))

    with open(save_dir / "selfies2bb.pkl", "wb") as g:
        pickle.dump(selfies2bb, g)

    with open(save_dir / "bb2selfies.pkl", "wb") as g:
        pickle.dump(bb2selfies, g)

    with open(save_dir / "bb_cn_dict.pkl", "wb") as g:
        pickle.dump(bb_cn_dict, g)

    print(f"Processed {len(results)} BBs, {len(failed)} failed")
    print(f"Output saved to: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess PORMAKE building blocks")
    parser.add_argument(
        "--bb_dir",
        type=str,
        default=None,
        help="Path to PORMAKE BB directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory for processed files",
    )
    args = parser.parse_args()

    preprocess_building_blocks(bb_dir=args.bb_dir, save_dir=args.output)
