from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models import build_model
from src.data import get_splits, load_dataset
from src.graph_utils import accuracy, build_random_walk_adj, infer_num_classes, resolve_device, row_normalize_features, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved ODINN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint saved by train.py.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name; otherwise read from checkpoint.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root; otherwise read from checkpoint or use data.")
    parser.add_argument("--fold", type=int, default=None, help="Fold to evaluate; otherwise read from checkpoint.")
    parser.add_argument("--split", type=str, default=None, choices=["generated", "official"], help="Split mode; otherwise read from checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example 'cuda:0' or 'cpu'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    saved_args = checkpoint.get("args", {})

    dataset = args.dataset or saved_args.get("dataset", "pubmed")
    root = args.root or saved_args.get("root", "data")
    split = args.split or saved_args.get("split", "generated")
    fold = checkpoint.get("fold", 0) if args.fold is None else args.fold
    seed = int(saved_args.get("seed", 42))
    label_rate = float(saved_args.get("label_rate", 1))
    val_ratio = float(saved_args.get("val_ratio", 0.1))

    set_random_seed(seed)
    loaded = load_dataset(dataset, root=root)
    data = loaded.data.to(device)
    features = data.x if saved_args.get("no_feature_norm", False) else row_normalize_features(data.x)
    norm_adj = build_random_walk_adj(data.edge_index.detach().cpu(), data.num_nodes).to(device)

    splits = get_splits(
        data=data,
        folds=max(fold + 1, int(saved_args.get("folds", 1))),
        label_rate=label_rate,
        val_ratio=val_ratio,
        seed=seed,
        official_split=loaded.official_split,
        split_mode=split,
    )
    _, _, test_mask = [mask.to(device) for mask in splits[fold]]

    model = build_model(
        model_name=saved_args.get("model_name", "DeGroot"),
        input_dim=int(checkpoint.get("input_dim", features.shape[1])),
        num_classes=int(checkpoint.get("num_classes", infer_num_classes(data.y))),
        num_layers=int(saved_args.get("num_layers", 10)),
        hid_units=int(saved_args.get("hid_units", 64)),
        mlp_units=int(saved_args.get("mlp_units", 64)),
        dropout=float(saved_args.get("dropout", 0.6)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    labels = data.y.to(device)
    with torch.no_grad():
        logits = model(features, norm_adj)
        loss = F.cross_entropy(logits[test_mask], labels[test_mask]).item()
        acc = accuracy(logits, labels, test_mask)

    print(f"checkpoint={args.checkpoint}")
    print(f"dataset={dataset} split={split} fold={fold}")
    print(f"test_loss={loss:.4f} test_acc={acc:.2f}")


if __name__ == "__main__":
    main()
