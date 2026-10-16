import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F


def _euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.sum((x - y) ** 2)))


def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _threshold_adjacency(dist_matrix: np.ndarray, radius: float) -> np.ndarray:
    mask = (dist_matrix >= 0) & (dist_matrix <= radius)
    return mask.astype(np.int32)


def _symmetric_normalise(adj: np.ndarray) -> np.ndarray:
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.where(rowsum == 0, 0.0, rowsum ** -0.5)
    D = np.diag(d_inv_sqrt)
    return D @ adj @ D


def _pad_adjacency(adj: np.ndarray, target_size: int) -> torch.Tensor:
    t = torch.from_numpy(adj).float()
    pad = target_size - t.size(0)
    return F.pad(t, (0, pad, 0, pad), "constant", 0)


def prepare_adj(pdb_id: str, max_len: int = 869) -> None:
    sc_path = f"../SC_position/{pdb_id}_psepos_SC.pkl"
    with open(sc_path, "rb") as fh:
        coords = joblib.load(fh)          # np.ndarray [L, 3]

    dist_matrix = _build_distance_matrix(coords)
    adj_binary  = _threshold_adjacency(dist_matrix, radius=17.0)
    adj_norm    = _symmetric_normalise(adj_binary)
    adj_padded  = _pad_adjacency(adj_norm, max_len)

    os.makedirs("../feature_combined/", exist_ok=True)
    torch.save(adj_padded, f"../feature_combined/{pdb_id}_adj.tensor")
