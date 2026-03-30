"""
code mofified from 'https://github.com/parkjunkil/MOFFUSION/blob/main/utils/build_materials.py'
"""

from typing import Optional, Union, List
import numpy as np
from pathlib import Path
import argparse
import pormake as pm
from egmof.builder import __builder_dir__
import warnings

warnings.filterwarnings("ignore")

# from pormake import *

pm.log.disable_print()
pm.log.disable_file_print()


def count_normal_atoms(bb):
    if bb is None:
        return 0
    else:
        return np.sum(bb.atoms.get_chemical_symbols() != np.array("X"))


def calculate_n_atoms_of_mof(_topology, _node_bbs, _edge_bbs):
    nt_counts = {}
    for nt in _topology.unique_node_types:
        n_nt = np.sum(_topology.node_types == nt)
        nt_counts[nt] = n_nt

    et_counts = {}
    for et in _topology.unique_edge_types:
        n_et = np.sum(np.all(_topology.edge_types == et[np.newaxis, :], axis=1))
        et_counts[tuple(et)] = n_et

    counts = 0
    for nt, bb in enumerate(_node_bbs):
        counts += nt_counts[nt] * count_normal_atoms(bb)

    for et, bb in _edge_bbs.items():
        counts += et_counts[et] * count_normal_atoms(bb)

    return counts


# Builder function
def name_to_mof(
    _mof_name: str,
    db: pm.Database | None = None,
    new_bb_dir: str | None = None,
    max_atoms: int = 5000,
):
    if db is None:
        db = pm.Database()
    if new_bb_dir is None:
        new_bb_dir = f"{__builder_dir__}/new_bbs"

    tokens: List[str] = _mof_name.split("+")
    _topo_name = tokens[0]

    _node_bb_names = []
    _edge_bb_names = []
    for bb in tokens[1:]:
        if bb.startswith("N") or bb.startswith("Custom_N"):
            _node_bb_names.append(bb)

        if bb.startswith("E") or bb.startswith("Custom_E"):
            _edge_bb_names.append(bb)

    _topology = db.get_topo(_topo_name)
    new_bb_db = pm.Database(bb_dir=new_bb_dir)

    def get_bb_with_fallback(name: str):
        if name.startswith("Custom"):
            return new_bb_db.get_bb(f"{name}.xyz")
        return db.get_bb(f"{name}.xyz")

    _node_bbs = [get_bb_with_fallback(n) for n in _node_bb_names]

    _edge_bbs = {
        tuple(et): None if n == "E0" else get_bb_with_fallback(n)
        for et, n in zip(_topology.unique_edge_types, _edge_bb_names)
    }

    # Check # of atoms of target MOF.
    n_atoms = calculate_n_atoms_of_mof(_topology, _node_bbs, _edge_bbs)
    if n_atoms > max_atoms:
        return "Many Atoms"

    # Check COF.
    has_metal = False
    for _bb in _node_bbs + list(_edge_bbs.values()):
        if _bb is None:
            continue
        if _bb.has_metal:
            has_metal = True
    if not has_metal:
        return "COF"

    _builder = pm.Builder()
    _mof = _builder.build_by_type(_topology, _node_bbs, _edge_bbs)

    return _mof


def build_materials(
    candidate_file: Union[str, Path],
    bb_dir: Optional[Union[str, Path]] = None,
    topo_dir: Optional[Union[str, Path]] = None,
    save_dir: Union[str, Path] | None = None,
    cutoff: float = 1e9,
):
    # Default save_dir
    if save_dir is None:
        save_dir = f"{__builder_dir__}/examples/cifs"

    # Basic settings for accessing database of pormake
    if isinstance(bb_dir, str):
        bb_dir = Path(bb_dir)
    if isinstance(topo_dir, str):
        topo_dir = Path(topo_dir)

    try:
        db = pm.Database(bb_dir=bb_dir, topo_dir=topo_dir)
    except:
        pass

    # Directory validation
    try:
        if not Path(candidate_file).resolve().exists():
            raise Exception("Error: candidate file does not exist!")
    except Exception as e:
        print(e)
        exit()

    Path(save_dir).resolve().mkdir(exist_ok=True, parents=True)

    # Read MOF names
    with open(candidate_file, "r") as f:
        mof_names = [line.strip() for line in f if line.strip()]

    print(f"Found {len(mof_names)} MOFs to generate")
    print(f"Output directory: {save_dir}")
    print("Start generation.")

    # Generate all MOFs
    success_count = 0
    for name in mof_names:
        print(f"Generating: {name}", end=" ")

        try:
            mof = name_to_mof(name, db)

            if isinstance(mof, str):
                print(f"-> Skipped: {mof}")
                continue

            min_cell_length = np.min(mof.atoms.cell.cellpar()[:3])
            if min_cell_length < 4.5:
                print("-> Too small cell. Skip.")
                continue

            max_cell_length = np.max(mof.atoms.cell.cellpar()[:3])
            if max_cell_length >= cutoff:
                print(
                    f"-> Cell length {max_cell_length:.1f} >= cutoff ({cutoff}). Skip."
                )
                continue

            cif_path = Path(save_dir) / f"{name}.cif"
            mof.write_cif(str(cif_path))
            print(f"-> Saved: {cif_path.name}")
            success_count += 1

        except Exception as e:
            print(f"-> Error: {e}")

    print(f"\nEnd generation. Success: {success_count}/{len(mof_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_MOFs")
    parser.add_argument(
        "candidates",
        nargs="?",
        default=None,
        help="Path to candidate file (one MOF name per line)",
    )
    parser.add_argument("-b", "--bb-dir", "--building-block-dir", default=None)
    parser.add_argument("-t", "--topo-dir", "--topology-dir", default=None)
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for CIF files (default: examples/cifs)",
    )
    parser.add_argument(
        "-co",
        "--cutoff",
        type=float,
        default=1e9,
        help="Max cell length threshold (default: 1e9, save all)",
    )

    args = parser.parse_args()

    if args.candidates is None:
        # Default: use generated_cif_list.txt
        args.candidates = f"{__builder_dir__}/examples/generated_cif_list.txt"

    build_materials(
        candidate_file=args.candidates,
        bb_dir=args.bb_dir,
        topo_dir=args.topo_dir,
        save_dir=args.save_dir,
        cutoff=args.cutoff,
    )
