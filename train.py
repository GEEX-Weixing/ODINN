from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import build_model
from src.data import SUPPORTED_DATASETS, get_splits, load_dataset
from src.graph_utils import accuracy, build_random_walk_adj, infer_num_classes, resolve_device, row_normalize_features, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate ODINN on citation/coauthor/product graph datasets.")
    parser.add_argument("--dataset", type=str, default="photo", choices=SUPPORTED_DATASETS, help="Dataset name.")
    parser.add_argument("--root", type=str, default="data", help="Directory used by PyG/OGB to cache datasets.")
    parser.add_argument("--model_name", type=str, default="DeGroot", help="DeGroot or Friedkin_Johnsen/FJ.")
    parser.add_argument("--split", type=str, default="generated", choices=["generated", "official"], help="Use generated few-shot splits or the official split when available.")
    parser.add_argument("--label_rate", type=float, default=1, help="Samples per class for generated splits. Fractional values are rounded up.")
    parser.add_argument("--folds", type=int, default=30, help="Number of generated splits/folds.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio among non-training nodes for generated splits.")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of ODINN propagation layers.")
    parser.add_argument("--hid_units", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--mlp_units", type=int, default=64, help="MLP hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=400, help="Maximum training epochs per fold.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience based on validation loss.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example 'cuda:0' or 'cpu'. Default auto-detects CUDA.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory.")
    parser.add_argument("--no_feature_norm", action="store_true", help="Disable row-normalization of node features in this script.")
    return parser.parse_args()


def train_one_fold(args: argparse.Namespace, data, features: torch.Tensor, norm_adj: torch.Tensor, fold: int, masks, device: torch.device):
    train_mask, val_mask, test_mask = [mask.to(device) for mask in masks]
    labels = data.y.to(device)

    set_random_seed(args.seed + fold)
    model = build_model(
        model_name=args.model_name,
        input_dim=features.shape[1],
        num_classes=infer_num_classes(data.y),
        num_layers=args.num_layers,
        hid_units=args.hid_units,
        mlp_units=args.mlp_units,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    checkpoint_path = Path(args.save_dir) / f"{args.dataset}_{args.model_name}_fold{fold}.pt"
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, norm_adj)
        loss_train = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, norm_adj)
            loss_val = F.cross_entropy(logits[val_mask], labels[val_mask])
            train_acc = accuracy(logits, labels, train_mask)
            val_acc = accuracy(logits, labels, val_mask)

        print(
            f"Fold {fold + 1} | Epoch {epoch:03d} | "
            f"train_loss={loss_train.item():.4f} val_loss={loss_val.item():.4f} "
            f"train_acc={train_acc:.2f} val_acc={val_acc:.2f}"
        )

        if loss_val.item() <= best_val_loss:
            best_val_loss = loss_val.item()
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "fold": fold,
                    "input_dim": int(features.shape[1]),
                    "num_classes": infer_num_classes(data.y),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(features, norm_adj)
        test_loss = F.cross_entropy(logits[test_mask], labels[test_mask]).item()
        test_acc = accuracy(logits, labels, test_mask)

    elapsed = time.time() - start_time
    print(
        f"Fold {fold + 1} finished | best_epoch={best_epoch} best_val_acc={best_val_acc:.2f} "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f} time={elapsed:.2f}s"
    )
    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "seconds": elapsed,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    set_random_seed(args.seed)

    loaded = load_dataset(args.dataset, root=args.root)
    data = loaded.data
    data = data.to(device)
    features = data.x if args.no_feature_norm else row_normalize_features(data.x)
    norm_adj = build_random_walk_adj(data.edge_index.detach().cpu(), data.num_nodes).to(device)

    splits = get_splits(
        data=data,
        folds=args.folds,
        label_rate=args.label_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        official_split=loaded.official_split,
        split_mode=args.split,
    )

    results = []
    for fold, masks in enumerate(splits):
        results.append(train_one_fold(args, data, features, norm_adj, fold, masks, device))

    test_accs = [item["test_acc"] for item in results]
    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "split": args.split,
        "label_rate": args.label_rate,
        "num_layers": args.num_layers,
        "folds": len(results),
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "mean_seconds": float(np.mean([item["seconds"] for item in results])),
        "results": results,
    }

    stem = f"{args.dataset}_{args.model_name}_{args.split}_L{args.num_layers}_label{args.label_rate}"
    json_path = Path(args.log_dir) / f"{stem}.json"
    csv_path = Path(args.log_dir) / f"{stem}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Test Accuracy: {summary['mean_test_acc']:.2f} ± {summary['std_test_acc']:.2f}")
    print(f"Saved logs to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
