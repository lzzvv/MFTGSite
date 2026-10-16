import os
import subprocess

import numpy as np
from Bio import pairwise2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_ORDER  = "ACDEFGHIKLMNPQRSTVWY"
_SS_ORDER  = "HBEGITSC"           # 8 DSSP secondary-structure codes
_RASA_STD  = [115, 135, 150, 190, 210, 75, 195, 175, 200,
               170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

_DSSP_EXE  = "./Software/dssp-2.0.4/mkdssp"
_PDB_DIR   = "../datasets/pro_pro_pdb"
_OUT_DIR   = "../dssp"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_dssp(dssp_path: str):
    with open(dssp_path) as fh:
        lines = fh.readlines()

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1

    seq, features = "", []
    for line in lines[p + 1 :]:
        aa = line[13]
        if aa in ("!", "*"):
            continue
        seq += aa

        ss = line[16]
        if ss == " ":
            ss = "C"
        ss_vec = np.zeros(9)
        ss_vec[_SS_ORDER.find(ss)] = 1

        phi = float(line[103:109].strip())
        psi = float(line[109:115].strip())
        acc = float(line[34:38].strip())
        aa_idx = _AA_ORDER.find(aa)
        asa = min(100, round(acc / _RASA_STD[aa_idx] * 100)) / 100 if aa_idx >= 0 else 0.0

        features.append(np.concatenate(([phi, psi, asa], ss_vec)))

    return seq, features


def _align_to_reference(dssp_seq: str, dssp_feats: list, ref_seq: str) -> list:
    alignments = pairwise2.align.globalxx(ref_seq, dssp_seq)
    aligned_ref, aligned_dssp = alignments[0].seqA, alignments[0].seqB

    unknown_ss = np.zeros(9)
    unknown_ss[-1] = 1
    gap_item = np.concatenate(([360.0, 360.0, 0.0], unknown_ss))

    dssp_iter = list(dssp_feats)
    expanded = []
    for aa in aligned_dssp:
        expanded.append(gap_item if aa == "-" else dssp_iter.pop(0))

    matched = [expanded[i] for i, aa in enumerate(aligned_ref) if aa != "-"]
    return matched


def _transform_angles(features: list) -> np.ndarray:
    arr    = np.array(features)
    angles = arr[:, :2]
    rest   = arr[:, 2:]
    rad    = angles * (np.pi / 180)
    return np.concatenate([np.sin(rad), np.cos(rad), rest], axis=1)


def get_dssp(pdb_id: str, ref_seq: str) -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)

    pdb_path    = f"{_PDB_DIR}/{pdb_id}.pdb"
    dssp_output = f"{_OUT_DIR}/{pdb_id}.dssp"
    npy_output  = f"{_OUT_DIR}/{pdb_id}.npy"

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not os.path.exists(_DSSP_EXE):
        raise FileNotFoundError(f"DSSP binary not found: {_DSSP_EXE}")

    cmd = f"{_DSSP_EXE} -i {pdb_path} -o {dssp_output}"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dssp_output):
        raise FileNotFoundError(
            f"DSSP failed for {pdb_id}.\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}"
        )

    dssp_seq, dssp_feats = _parse_dssp(dssp_output)
    if dssp_seq != ref_seq:
        dssp_feats = _align_to_reference(dssp_seq, dssp_feats, ref_seq)

    np.save(npy_output, _transform_angles(dssp_feats))
    os.remove(dssp_output)   # clean up intermediate file
