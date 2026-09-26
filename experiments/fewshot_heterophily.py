from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_heterophily import DATASETS, fewshot_split, load_dataset, set_seed
from src.odinn_heterophily import build_odinn

FIELDS = [
    "dataset", "model", "shots", "hops", "seed",
    "val_acc", "test_acc", "best_epoch", "train_seconds", "status",
]


def append_csv(path: Path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_ok(path: Path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((
                    row["dataset"], row["model"], int(row["shots"]),
                    int(row["hops"]), int(row["seed"]),
                ))
    return done


def accuracy(logits, y, mask):
    return float((logits[mask].argmax(-1) == y[mask]).float().mean().item() * 100.0)


def train_one(model, data, masks, device, epochs, patience, lr, weight_decay):
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)
            val_loss = F.cross_entropy(val_logits[masks["val"]], y[masks["val"]]).item()

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        val_acc = accuracy(logits, y, masks["val"])
        test_acc = accuracy(logits, y, masks["test"])
    return val_acc, test_acc, best_epoch, elapsed


def summarize(runs: Path, out_dir: Path, folds: int) -> None:
    if not runs.exists():
        return
    df = pd.read_csv(runs)
    df = df[df.status.eq("ok")].copy()
    if df.empty:
        return
    agg = (
        df.groupby(["dataset", "model", "shots", "hops"])
        .agg(
            val_mean=("val_acc", "mean"),
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", lambda x: x.std(ddof=0)),
            n=("test_acc", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "odinn_hop_summary.csv", index=False)
    eligible = agg[agg.n >= folds].copy()
    if not eligible.empty:
        best = (
            eligible.sort_values(
                ["dataset", "model", "shots", "val_mean"],
                ascending=[True, True, True, False],
            )
            .groupby(["dataset", "model", "shots"], as_index=False)
            .first()
        )
        best.to_csv(out_dir / "odinn_best_summary.csv", index=False)
        print(best.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="ODINN few-shot experiment on heterophilous graphs")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=["ODINN-DG", "ODINN-FJ"])
    ap.add_argument("--shots", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--hops", nargs="*", type=int, default=[10, 20, 40])
    ap.add_argument("--folds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out_dir", default="results/fewshot_heterophily")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "odinn_all_runs.csv"
    done = completed_ok(runs)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name, args.root)
        for model_name in args.models:
            for shots in args.shots:
                for hops in args.hops:
                    for i in range(args.folds):
                        seed = args.seed0 + i
                        key = (bundle.name, model_name, shots, hops, seed)
                        if key in done:
                            continue
                        set_seed(seed)
                        masks = fewshot_split(bundle.data.y, shots, seed)
                        model = build_odinn(
                            model_name,
                            bundle.data.num_features,
                            args.hidden,
                            bundle.num_classes,
                            dropout=args.dropout,
                            hops=hops,
                        )
                        try:
                            val, test, best_epoch, secs = train_one(
                                model, bundle.data, masks, device,
                                args.epochs, args.patience, args.lr, args.weight_decay,
                            )
                            row = {
                                "dataset": bundle.name, "model": model_name,
                                "shots": shots, "hops": hops, "seed": seed,
                                "val_acc": val, "test_acc": test,
                                "best_epoch": best_epoch, "train_seconds": secs,
                                "status": "ok",
                            }
                        except RuntimeError as exc:
                            status = "oom" if "out of memory" in str(exc).lower() else "error"
                            row = {
                                "dataset": bundle.name, "model": model_name,
                                "shots": shots, "hops": hops, "seed": seed,
                                "val_acc": "", "test_acc": "", "best_epoch": "",
                                "train_seconds": "", "status": status,
                            }
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        append_csv(runs, row)
                        if row["status"] == "ok":
                            done.add(key)

    summarize(runs, out, args.folds)


if __name__ == "__main__":
    main()
