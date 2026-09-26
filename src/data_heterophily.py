from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import HeterophilousGraphDataset, LINKXDataset, WikipediaNetwork
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

DATASETS = ["chameleon", "squirrel", "amazon_ratings", "penn94"]
ALIASES = {
    "chameleon": "chameleon",
    "squirrel": "squirrel",
    "amazon_ratings": "amazon_ratings",
    "amazon-ratings": "amazon_ratings",
    "penn94": "penn94",
    "penn-94": "penn94",
}


@dataclass
class GraphBundle:
    name: str
    data: Data
    num_classes: int
    source_path: str


def canonical_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key not in ALIASES:
        raise ValueError(f"Unsupported dataset {name!r}. Supported: {DATASETS}")
    return ALIASES[key]


def _standardize(data: Data) -> Data:
    data = data.clone()
    data.x = data.x.float()
    data.y = data.y.view(-1).long()
    data.edge_index = data.edge_index.long()
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    return T.NormalizeFeatures()(data)


def load_dataset(name: str, root: str = "data") -> GraphBundle:
    """Load the four heterophilous graphs used by the ODINN few-shot experiment.

    The data directory is assumed to exist. PyG-compatible cached layouts are
    reused when present; PyG may download missing raw files into these folders.
    """
    name = canonical_name(name)
    root = Path(root).expanduser()
    hetero_root = root / "heterophily"
    hetero_root.mkdir(parents=True, exist_ok=True)

    if name in {"chameleon", "squirrel"}:
        base = hetero_root / "wikipedia_network"
        ds = WikipediaNetwork(root=str(base), name=name, geom_gcn_preprocess=True)
        data = ds[0]
    elif name == "amazon_ratings":
        base = hetero_root / "heterophilous_graphs"
        ds = HeterophilousGraphDataset(root=str(base), name="Amazon-ratings")
        data = ds[0]
    elif name == "penn94":
        base = hetero_root / "linkx"
        ds = LINKXDataset(root=str(base), name="penn94")
        data = ds[0]
    else:
        raise AssertionError(name)

    data = _standardize(data)
    valid = data.y[data.y >= 0]
    if valid.numel() == 0:
        raise RuntimeError(f"{name}: no labeled nodes")
    return GraphBundle(name, data, int(valid.max().item()) + 1, str(base))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fewshot_split(y: torch.Tensor, shots: int, seed: int) -> Dict[str, torch.Tensor]:
    """Use exactly ``shots`` training nodes per class; split the rest 1:9 into val/test."""
    y = y.detach().cpu().view(-1).long()
    valid = y >= 0
    classes = torch.unique(y[valid], sorted=True)
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.zeros(y.numel(), dtype=torch.bool)
    train_parts = []

    for c in classes.tolist():
        idx = ((y == int(c)) & valid).nonzero(as_tuple=False).view(-1)
        if idx.numel() < shots:
            raise ValueError(f"class {c}: {idx.numel()} nodes < shots={shots}")
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        selected = idx[:shots]
        chosen[selected] = True
        train_parts.append(selected)

    train = torch.cat(train_parts)
    remain = (valid & ~chosen).nonzero(as_tuple=False).view(-1)
    remain = remain[torch.randperm(remain.numel(), generator=generator)]
    n_val = max(1, int(round(remain.numel() * 0.1)))
    n_val = min(n_val, max(1, remain.numel() - 1))
    val, test = remain[:n_val], remain[n_val:]

    def mask(idx):
        m = torch.zeros(y.numel(), dtype=torch.bool)
        m[idx] = True
        return m

    return {"train": mask(train), "val": mask(val), "test": mask(test)}
