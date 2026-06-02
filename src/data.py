from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


SUPPORTED_DATASETS = ("pubmed", "wiki_cs", "photo", "ogbn-arxiv")


@dataclass
class LoadedDataset:
    name: str
    data: object
    official_split: Optional[Dict[str, torch.Tensor]] = None


def canonical_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "pubmed": "pubmed",
        "wiki": "wiki_cs",
        "wikics": "wiki_cs",
        "wiki_cs": "wiki_cs",
        "amazon_photo": "photo",
        "photo": "photo",
        "ogbn_arxiv": "ogbn-arxiv",
        "ogbn-arxiv": "ogbn-arxiv",
        "arxiv": "ogbn-arxiv",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {', '.join(SUPPORTED_DATASETS)}")
    return aliases[key]


def load_dataset(name: str, root: str = "data") -> LoadedDataset:
    """Load PubMed, WikiCS, Amazon Photo, or OGBN-Arxiv.

    The datasets are downloaded/cached by PyTorch Geometric or OGB under ``root``.
    """
    dataset_name = canonical_dataset_name(name)
    try:
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Amazon, Planetoid, WikiCS
    except ImportError as exc:
        raise ImportError("torch-geometric is required. Install dependencies from requirements.txt.") from exc

    if dataset_name == "pubmed":
        dataset = Planetoid(root=f"{root}/pyg/planetoid", name="PubMed", transform=T.NormalizeFeatures())
        data = dataset[0]
        return LoadedDataset(dataset_name, _prepare_data(data))

    if dataset_name == "wiki_cs":
        dataset = WikiCS(root=f"{root}/pyg/wiki_cs", transform=T.NormalizeFeatures())
        data = dataset[0]
        return LoadedDataset(dataset_name, _prepare_data(data))

    if dataset_name == "photo":
        dataset = Amazon(root=f"{root}/pyg/amazon", name="Photo", transform=T.NormalizeFeatures())
        data = dataset[0]
        return LoadedDataset(dataset_name, _prepare_data(data))

    if dataset_name == "ogbn-arxiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise ImportError("ogb is required for ogbn-arxiv. Install dependencies from requirements.txt.") from exc
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}/ogb", transform=T.ToUndirected())
        data = dataset[0]
        data.y = data.y.squeeze().long()
        split = {k: v.long() for k, v in dataset.get_idx_split().items()}
        return LoadedDataset(dataset_name, _prepare_data(data), official_split=split)

    raise AssertionError("unreachable")


def _prepare_data(data):
    data.y = data.y.squeeze().long()
    data.x = data.x.float()
    return data


def split_indices_to_masks(split: Dict[str, torch.Tensor], num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    train[split["train"]] = True
    val[split["valid"] if "valid" in split else split["val"]] = True
    test[split["test"]] = True
    return train, val, test


def generate_stratified_splits(
    labels: torch.Tensor,
    folds: int,
    label_rate: float,
    val_ratio: float,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate few-shot stratified splits.

    ``label_rate`` is interpreted as samples per class. For compatibility with older
    scripts, fractional values are rounded up to at least one sample per class.
    """
    labels_np = labels.detach().cpu().numpy()
    num_nodes = labels_np.shape[0]
    classes = np.unique(labels_np)
    per_class = max(1, int(np.ceil(label_rate)))

    splits = []
    for fold in range(folds):
        rng = np.random.default_rng(seed + fold)
        train_indices = []
        for cls in classes:
            cls_indices = np.where(labels_np == cls)[0]
            if len(cls_indices) == 0:
                continue
            take = min(per_class, len(cls_indices))
            train_indices.extend(rng.choice(cls_indices, size=take, replace=False).tolist())

        all_indices = np.arange(num_nodes)
        remaining = np.setdiff1d(all_indices, np.array(train_indices, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)

        val_size = int(round(len(remaining) * val_ratio))
        val_indices = remaining[:val_size]
        test_indices = remaining[val_size:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        splits.append((train_mask, val_mask, test_mask))

    return splits


def get_splits(
    data,
    folds: int,
    label_rate: float,
    val_ratio: float,
    seed: int,
    official_split: Optional[Dict[str, torch.Tensor]] = None,
    split_mode: str = "generated",
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if split_mode == "official":
        if official_split is None:
            raise ValueError("Official split is only available for datasets that provide it, such as ogbn-arxiv.")
        return [split_indices_to_masks(official_split, data.num_nodes)]
    if split_mode != "generated":
        raise ValueError("split_mode must be 'generated' or 'official'")
    return generate_stratified_splits(data.y, folds=folds, label_rate=label_rate, val_ratio=val_ratio, seed=seed)
