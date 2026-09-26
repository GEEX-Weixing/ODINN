from __future__ import annotations

"""Local-first loaders for the four heterophilous graphs used by ODINN.

The loader first reuses compatible caches from the layouts supported by the
original ODINN repository.  If no cache is found, PyG downloads/processes the
dataset under ``<root>/heterophily``.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import HeterophilousGraphDataset, LINKXDataset, WikipediaNetwork
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

DATASETS = ["chameleon", "squirrel", "amazon_ratings", "penn94"]
ALIASES = {
    "chameleon": "chameleon",
    "cham": "chameleon",
    "squirrel": "squirrel",
    "squi": "squirrel",
    "amazon_ratings": "amazon_ratings",
    "amazon-ratings": "amazon_ratings",
    "amazon ratings": "amazon_ratings",
    "penn94": "penn94",
    "penn_94": "penn94",
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


def _unique(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        path = Path(path).expanduser()
        key = str(path.absolute())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _first_existing(candidates: List[Path], relative_files: List[str]) -> Optional[Path]:
    """Return the first cache root containing processed or raw dataset files."""
    for base in candidates:
        if any((base / rel).is_file() for rel in relative_files):
            return base
    return None


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
    """Load an ODINN heterophilous benchmark using the original local-first layout."""
    name = canonical_name(name)
    root_path = Path(root).expanduser()
    canonical_root = root_path / "heterophily"
    canonical_root.mkdir(parents=True, exist_ok=True)

    if name in {"chameleon", "squirrel"}:
        candidates = _unique([
            canonical_root / "wikipedia_network",
            root_path / "WikipediaNetwork",
            root_path / "wikipedia_network",
            root_path / "pyg",
            root_path,
        ])
        base = _first_existing(candidates, [
            f"{name}/geom_gcn/processed/data.pt",
            f"{name}/geom_gcn/raw/out1_node_feature_label.txt",
            f"{name}/geom_gcn/raw/out1_graph_edges.txt",
            f"{name}/geom_gcn/raw/{name}_split_0.6_0.2_0.npz",
        ])
        if base is None:
            base = canonical_root / "wikipedia_network"
            print(f"[DOWNLOAD] {name} -> {base}", flush=True)
        else:
            print(f"[CACHE] {name} -> {base}", flush=True)
        ds = WikipediaNetwork(root=str(base), name=name, geom_gcn_preprocess=True)
        data = ds[0]

    elif name == "amazon_ratings":
        candidates = _unique([
            canonical_root / "heterophilous_graphs",
            root_path / "heterophilous_graphs",
            root_path / "pyg" / "heterophilous_graphs",
            root_path / "pyg",
            root_path,
        ])
        base = _first_existing(candidates, [
            "amazon_ratings/processed/data.pt",
            "amazon_ratings/raw/amazon_ratings.npz",
        ])
        if base is None:
            base = canonical_root / "heterophilous_graphs"
            print(f"[DOWNLOAD] {name} -> {base}", flush=True)
        else:
            print(f"[CACHE] {name} -> {base}", flush=True)
        ds = HeterophilousGraphDataset(root=str(base), name="Amazon-ratings")
        data = ds[0]

    elif name == "penn94":
        candidates = _unique([
            canonical_root / "linkx",
            root_path / "linkx",
            root_path / "pyg" / "linkx",
            root_path / "pyg",
            root_path,
        ])
        base = _first_existing(candidates, [
            "penn94/processed/data.pt",
            "penn94/raw/data.mat",
            "penn94/raw/fb100-Penn94-splits.npy",
        ])
        if base is None:
            base = canonical_root / "linkx"
            print(f"[DOWNLOAD] penn94 -> {base}", flush=True)
        else:
            print(f"[CACHE] penn94 -> {base}", flush=True)
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

    def mask(idx: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(y.numel(), dtype=torch.bool)
        m[idx] = True
        return m

    return {"train": mask(train), "val": mask(val), "test": mask(test)}
