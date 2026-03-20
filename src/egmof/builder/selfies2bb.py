import os
import sys
import math
import shutil
import selfies
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

# Placeholder atom symbol used in SELFIES (e.g., Lr)
# This is temporarily replaced by H during optimization

'''
Usage
- xtb: 
output = decode_selfies_to_xyz_opt(selfies_str, 
                                   run_dir="{your_directory}/tmp",
                                   out_xyz_name = 'e1_xtb.xyz',
                                   xtb_bin = '/home1/users/seunghh/xtb-dist/bin/xtb',
                                    engine="xtb")
- MMFF94:
output =decode_selfies_to_xyz_opt(selfies_str, run_dir="{your_directory}/tmp",
                                  out_xyz_name = 'e1_mmff.xyz',
                                   engine="mmff")
- UFF:
output = decode_selfies_to_xyz_opt(selfies_str, run_dir="{your_directory}/tmp",
                                   out_xyz_name = 'e1_uff.xyz',
                                    engine="uff")


'''


PLACEHOLDERS = 'Lr'


def selfies_to_mol(selfies_str):
    """
    Convert a SELFIES string to an RDKit molecule.
    Raises an error if SMILES decoding or sanitization fails.
    """   
    smiles = selfies.decoder(selfies_str)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return mol


def get_placeholder_indices(mol):
    """
    Return atom indices corresponding to placeholder atoms (e.g., Lr).
    These will be restored after geometry optimization.
    """
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in PLACEHOLDERS]


def replace_placeholder_with_H(mol):
    """
    Replace placeholder atoms with hydrogen.
    This improves stability for force-field and xTB optimizations.
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == PLACEHOLDERS:
            atom.SetAtomicNum(1)  # H
    Chem.SanitizeMol(mol)
    return mol


def restore_X(mol, placeholder_idxs):
    """
    Restore placeholder atoms as dummy atoms (atomic number 0, written as 'X').
    Called after optimization is complete.
    """    
    for idx in placeholder_idxs:
        mol.GetAtomWithIdx(idx).SetAtomicNum(0)  # dummy atom = X
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    return mol


def load_xyz_to_conformer(mol, xyz_path, confId):
    """
    Load atomic coordinates from an XYZ file into an existing RDKit conformer.
    Assumes atom ordering matches exactly.
    """    
    conf = mol.GetConformer(confId)
    with open(xyz_path, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    body = lines[2:2 + n]
    if n != mol.GetNumAtoms():
        raise ValueError(f"XYZ atom count ({n}) != mol atom count ({mol.GetNumAtoms()})")

    for i, line in enumerate(body):
        parts = line.split()
        x, y, z = map(float, parts[1:4])
        conf.SetAtomPosition(i, (x, y, z))


def mol_to_extended_xyz(mol, path, confId=None):
    """
    Write an extended XYZ file:
    - atom symbols and coordinates
    - additional bond connectivity lines
    Placeholder atoms are written as 'X'.
    """
    if confId is None:
        confId = 0
    conf = mol.GetConformer(confId)

    atoms = []
    for a in mol.GetAtoms():
        atoms.append('X' if a.GetAtomicNum() == 0 else a.GetSymbol())

    bonds = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()

        bt = b.GetBondType()
        if bt == Chem.BondType.SINGLE:
            c = 'S'
        elif bt == Chem.BondType.DOUBLE:
            c = 'D'
        elif bt == Chem.BondType.TRIPLE:
            c = 'T'
        elif bt == Chem.BondType.AROMATIC:
            c = 'A'
        else:
            continue
        bonds.append((i, j, c))

    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("generated from SELFIES\n")

        for i, elem in enumerate(atoms):
            p = conf.GetAtomPosition(i)
            f.write(f"{elem:2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}\n")

        for i, j, c in bonds:
            f.write(f"{i:4d} {j:4d} {c}\n")


def run_xtb_tight(
    xyz_filename,
    run_dir,
    xtb_bin="xtb",
    charge=0,
    uhf=0,
    gfn=2,
    solvation_model=None,  # None | "gbsa" | "alpb"
    solvent=None,          # e.g., "water", "methanol"
    cleanup = True
):
    """
    Run xTB geometry optimization (tight convergence).

    Parameters
    ----------
    solvation_model : None | "gbsa" | "alpb"
        Implicit solvation model.
    solvent : str
        Solvent name (e.g., "water"). Required if solvation_model is set.

    Returns
    -------
    Path to xtbopt.xyz
    """
    if shutil.which(xtb_bin) is None:
        raise FileNotFoundError("xtb executable not found in PATH.")

    cmd = [
        xtb_bin, xyz_filename,
        "--gfn", str(int(gfn)),
        "--opt", "tight",
        "--chrg", str(int(charge)),
        "--uhf", str(int(uhf)),
    ]

    if solvation_model is not None:
        m = solvation_model.lower()
        if m not in ("gbsa", "alpb"):
            raise ValueError("solvation_model must be one of: None, 'gbsa', 'alpb'")
        if not solvent:
            raise ValueError("When solvation_model is set, solvent must be provided (e.g., 'water').")
        # xTB: --gbsa water  or  --alpb methanol
        cmd += [f"--{m}", str(solvent)]

    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    out_path = os.path.join(run_dir, "xtb.out")
    with open(out_path, "w") as f:
        f.write(proc.stdout)

    xtbopt = os.path.join(run_dir, "xtbopt.xyz")
    if proc.returncode != 0 or not os.path.exists(xtbopt):
        raise RuntimeError(f"xTB optimization failed. See {out_path}")
    
    if cleanup:
        # NOTE: xtb.out is the captured stdout (we created it), and xtbopt.xyz is the main result.
        # You asked to delete both too; if you want to KEEP xtbopt.xyz, remove it from this list.
        files_to_remove = [
            ".xtboptok",
            "charges",
            "seed.input.xyz",
            "wbo",
            "xtb.out",
            "xtbopt.log",
   
            "xtbrestart",
            "xtbtopo.mol",
        ]

        for name in files_to_remove:
            p = os.path.join(run_dir, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
            except Exception:
                # Best-effort cleanup: ignore deletion errors (e.g., permissions, in-use)
                pass
    return xtbopt


def run_uff_opt(mol, confId=0, maxIters=2000):
    """
    Perform in-place UFF geometry optimization using RDKit.
    Used as a general-purpose fallback method.
    """
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=confId)
    if ff is None:
        raise RuntimeError("UFF force field could not be constructed (missing parameters?).")

    ff.Initialize()
    ff.Minimize(maxIts=int(maxIters))
    energy = float(ff.CalcEnergy())

    # RDKit UFFOptimizeMolecule는 수렴 여부를 int로 주기도 하는데
    # 여기선 간단히 에너지만 반환하고, 필요하면 추가로 체크 로직 넣을 수 있음
    return True, energy


def run_mmff94_opt(mol, confId=0, maxIters=2000, variant="MMFF94"):
    """
    Perform in-place MMFF94 or MMFF94s optimization.
    Recommended for organic molecules when parameters are available.
    """
    variant = str(variant)
    if variant not in ("MMFF94", "MMFF94s"):
        raise ValueError("MMFF variant must be 'MMFF94' or 'MMFF94s'")

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if props is None:
        raise RuntimeError("MMFF properties could not be assigned (missing parameters?).")

    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=confId)
    if ff is None:
        raise RuntimeError("MMFF force field could not be constructed.")

    ff.Initialize()
    ff.Minimize(maxIts=int(maxIters))
    energy = float(ff.CalcEnergy())
    return True, energy


def normalize_run_dir(run_dir: str) -> str:
    """
    Normalize run directory:
    - empty string → current working directory
    - relative path → absolute path
    """
    if run_dir is None or str(run_dir).strip() == "":
        run_dir = os.getcwd()
    else:
        run_dir = str(run_dir)
        if not os.path.isabs(run_dir):
            run_dir = os.path.abspath(run_dir)

    return run_dir


def set_dummy_X_bond_length(mol, dummy_idxs, target_len=0.75, confId=0):
    """
    After restore_X(), dummy atoms (atomic num 0, written as 'X') keep the optimized
    coordinates of the former H atoms. This function moves ONLY the dummy atom positions
    so that the distance between each X and its bonded neighbor becomes target_len (Å).

    Assumptions / Notes
    -------------------
    - Each dummy atom is expected to have exactly one neighbor (common in "attachment point" use).
    - If there are multiple neighbors, this function uses the first neighbor found.
      (If you have a real multi-bond dummy, tell me and I'll adapt it.)
    - The neighbor atom coordinates are kept fixed; only the dummy atom is moved.
    """
    conf = mol.GetConformer(confId)

    for x_idx in dummy_idxs:
        atom_x = mol.GetAtomWithIdx(x_idx)
        if atom_x.GetAtomicNum() != 0:
            # Not a dummy atom; skip silently
            continue

        nbrs = [n.GetIdx() for n in atom_x.GetNeighbors()]
        if len(nbrs) == 0:
            continue  # isolated dummy, nothing to do

        nbr_idx = nbrs[0]
        p_x = conf.GetAtomPosition(x_idx)
        p_n = conf.GetAtomPosition(nbr_idx)

        vx = p_x.x - p_n.x
        vy = p_x.y - p_n.y
        vz = p_x.z - p_n.z
        dist = math.sqrt(vx*vx + vy*vy + vz*vz)

        if dist < 1e-8:
            # Degenerate case: X exactly on neighbor; can't define direction
            continue

        scale = float(target_len) / dist
        new_p = Point3D(
            p_n.x + vx * scale,
            p_n.y + vy * scale,
            p_n.z + vz * scale
        )
        conf.SetAtomPosition(x_idx, new_p)

def decode_selfies_to_xyz_opt(
    selfies_str,
    run_dir = '',
    out_xyz_name="tmp.xyz",
    seed=0,
    charge=0,
    uhf=0,
    engine="xtb",          # "xtb" | "uff" | "mmff"
    xtb_bin="xtb",
    xtb_gfn=2,
    solvation_model=None,  # None | "gbsa" | "alpb" (xtb)
    solvent=None,          # e.g. "water" (xtb)
    uff_max_iters=2000,
    mmff_max_iters=2000,
    mmff_variant="MMFF94", # "MMFF94" | "MMFF94s"
    xtb_cleanup=True,
):
    """
    Main workflow:
    SELFIES
      → RDKit ETKDG seed conformer
      → Geometry optimization (UFF / MMFF / xTB)
      → Restore placeholders
      → Write extended XYZ
    """
    run_dir = normalize_run_dir(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    mol = selfies_to_mol(selfies_str)

    # placeholder(Lr) → H (최적화 안정성)
    placeholder_idxs = get_placeholder_indices(mol)
    mol = replace_placeholder_with_H(mol)

    # seed conformer 생성
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("RDKit EmbedMolecule failed.")

    engine = engine.lower().strip()
    result = {"engine": engine}

    if engine == "xtb":
        seed_xyz_name = "seed.input.xyz"
        seed_xyz_path = os.path.join(run_dir, seed_xyz_name)
        Chem.MolToXYZFile(mol, seed_xyz_path, confId=cid)

        xtbopt = run_xtb_tight(
            xyz_filename=seed_xyz_name,
            run_dir=run_dir,
            xtb_bin=xtb_bin,
            charge=charge,
            uhf=uhf,
            gfn=xtb_gfn,
            solvation_model=solvation_model,
            solvent=solvent,
            cleanup=xtb_cleanup,
        )
        load_xyz_to_conformer(mol, xtbopt, confId=cid)
        result["xtbopt"] = xtbopt

    elif engine == "uff":
        converged, e = run_uff_opt(mol, confId=cid, maxIters=uff_max_iters)
        result["uff_converged"] = converged
        result["uff_energy"] = e

    elif engine == "mmff":
        converged, e = run_mmff94_opt(
            mol, confId=cid,
            maxIters=mmff_max_iters,
            variant=mmff_variant
        )
        result["mmff_converged"] = converged
        result["mmff_energy"] = e
        result["mmff_variant"] = mmff_variant

    else:
        raise ValueError("engine must be 'xtb' or 'uff' or 'mmff'")

    # placeholder 복원 (H → X)
    mol = restore_X(mol, placeholder_idxs)

    # Adjust X–neighbor distance to a fixed bond length (Å)
    set_dummy_X_bond_length(mol, placeholder_idxs, target_len=0.75, confId=cid)

    # 결과 저장
    out_xyz_path = os.path.join(run_dir, out_xyz_name)
    mol_to_extended_xyz(mol, out_xyz_path, confId=cid)

    result["out"] = out_xyz_path
    result["mol"] = mol
    return result



# ==========================================
# Main Execution Block
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode SELFIES to Optimized XYZ (xTB/MMFF/UFF)")
    
    # 필수 인자
    parser.add_argument("selfies", type=str, help="SELFIES string to decode")
    
    # 공통 옵션
    parser.add_argument("--run_dir", type=str, default="tmp", help="Directory for execution and output")
    parser.add_argument("--out", type=str, default="output.xyz", help="Output XYZ filename")
    parser.add_argument("--engine", type=str, default="xtb", choices=["xtb", "mmff", "uff"], help="Optimization engine")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for conformer generation")
    
    # xTB 옵션
    parser.add_argument("--xtb_bin", type=str, default="xtb", help="Path to xTB executable")
    parser.add_argument("--gfn", type=int, default=2, help="GFN-xTB version")
    parser.add_argument("--charge", type=int, default=0, help="Total system charge")
    parser.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons")
    parser.add_argument("--solvation", type=str, default=None, choices=["gbsa", "alpb"], help="Solvation model")
    parser.add_argument("--solvent", type=str, default=None, help="Solvent name (e.g., water, methanol)")
    parser.add_argument("--no_cleanup", action="store_true", help="Do not remove xTB temporary files")
    
    # Force Field 옵션
    parser.add_argument("--iters", type=int, default=2000, help="Max iterations for UFF/MMFF")
    parser.add_argument("--mmff_variant", type=str, default="MMFF94", choices=["MMFF94", "MMFF94s"], help="MMFF variant")

    args = parser.parse_args()
    
    print(f"Start processing SELFIES: {args.selfies}")
    print(f"Engine: {args.engine.upper()}")

    try:
        # 함수 실행
        output = decode_selfies_to_xyz_opt(
            selfies_str=args.selfies,
            run_dir=args.run_dir,
            out_xyz_name=args.out,
            seed=args.seed,
            charge=args.charge,
            uhf=args.uhf,
            engine=args.engine,
            xtb_bin=args.xtb_bin,
            xtb_gfn=args.gfn,
            solvation_model=args.solvation,
            solvent=args.solvent,
            uff_max_iters=args.iters,
            mmff_max_iters=args.iters,
            mmff_variant=args.mmff_variant,
            xtb_cleanup=not args.no_cleanup  # cleanup 플래그 반전
        )
        
        print(f"\nSuccess! Output saved to: {output['out']}")
        if args.engine == "xtb":
            # xtbopt 키가 있는지 확인 후 출력
            print(f"xTB optimized geometry (intermediate): {output.get('xtbopt', 'N/A')}")
        elif args.engine == "mmff":
            print(f"MMFF Energy: {output.get('mmff_energy', 0):.4f} kcal/mol")
        elif args.engine == "uff":
            print(f"UFF Energy: {output.get('uff_energy', 0):.4f} kcal/mol")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)