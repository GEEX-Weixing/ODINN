from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .odinn_weight_init import build_gpr_style_adam


@dataclass
class TrainResult:
    seed: int
    best_epoch: int
    val_acc: float
    test_acc: float
    train_seconds: float
    status: str = "ok"


def accuracy(logits, y, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return float((logits[mask].argmax(-1) == y[mask]).float().mean().item() * 100.0)


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_odinn(
    model,
    data,
    masks,
    w_lazy,
    device,
    epochs=1000,
    lr=0.005,
    weight_decay=5e-4,
    patience=50,
    dynamics_weight_decay=0.0,
):
    x = data.x.to(device)
    y = data.y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}
    model = model.to(device)
    w_lazy = w_lazy.to(device)

    optimizer = build_gpr_style_adam(
        model,
        lr=lr,
        weight_decay=weight_decay,
        dynamics_weight_decay=dynamics_weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    _cuda_sync(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, w_lazy)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(
                model(x, w_lazy)[masks["val"]], y[masks["val"]]
            ).item()

        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    _cuda_sync(device)
    elapsed = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, w_lazy)
        val_acc = accuracy(logits, y, masks["val"])
        test_acc = accuracy(logits, y, masks["test"])

    return TrainResult(0, best_epoch, val_acc, test_acc, elapsed), model
