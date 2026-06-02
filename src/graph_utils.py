from __future__ import annotations

import random
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def row_normalize_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    row_sum = x.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    return x / row_sum


def build_random_walk_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return sparse D^-1(A + A^T + I) as a coalesced PyTorch sparse tensor on CPU."""
    try:
        from torch_geometric.utils import to_scipy_sparse_matrix
    except ImportError as exc:
        raise ImportError("torch-geometric is required for building adjacency matrices. Install requirements.txt first.") from exc

    adj = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes).tocsr()
    adj = (adj + adj.T).tocsr()
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="csr")

    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    inv = np.power(rowsum, -1.0, where=rowsum != 0)
    inv[rowsum == 0] = 0.0
    norm_adj = sp.diags(inv.astype(np.float32), format="csr").dot(adj).tocoo()

    indices = torch.tensor(np.vstack((norm_adj.row, norm_adj.col)), dtype=torch.long)
    values = torch.tensor(norm_adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).coalesce()


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if int(mask.sum()) == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return float((pred[mask] == labels[mask]).float().mean().item() * 100.0)


def infer_num_classes(labels: torch.Tensor) -> int:
    labels = labels.detach().cpu().long()
    return int(labels.max().item() + 1)


def resolve_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
