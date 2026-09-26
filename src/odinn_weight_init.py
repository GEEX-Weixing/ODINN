from __future__ import annotations

from typing import List

import torch


def ppr_init(
    length: int,
    a: float = 0.1,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """PPR initialization used for ODINN-DG aggregation weights alpha.

    For ``n`` coefficients, entries 0..n-2 are ``a(1-a)^k`` and the final
    coefficient is ``(1-a)^(n-1)``.  This is only an initialization; alpha is
    unconstrained during training.
    """
    n = int(length)
    if n < 0:
        raise ValueError("length must be non-negative")
    if n == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if not (0.0 < float(a) < 1.0):
        raise ValueError("PPR parameter a must lie in (0, 1)")
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)

    out = torch.empty(n, device=device, dtype=dtype)
    k = torch.arange(n - 1, device=device, dtype=dtype)
    base = torch.tensor(1.0 - float(a), device=device, dtype=dtype)
    out[:-1] = float(a) * torch.pow(base, k)
    out[-1] = (1.0 - float(a)) ** (n - 1)
    return out


def balanced_init(
    length: int,
    center: float = 0.5,
    std: float = 0.01,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Balanced initialization used for ODINN-FJ stage weights beta.

    ``beta_k = center + epsilon_k`` with ``epsilon_k ~ N(0, std^2)``.  No
    depth-dependent trend is imposed.  This is only an initialization; beta is
    unconstrained during training.  Reproducibility follows PyTorch's global
    RNG, so call the project ``set_seed`` before model construction.
    """
    n = int(length)
    if n < 0:
        raise ValueError("length must be non-negative")
    if float(std) < 0.0:
        raise ValueError("std must be non-negative")
    if n == 0:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.empty(n, device=device, dtype=dtype).normal_(
        mean=float(center), std=float(std)
    )


def dynamics_parameter_names(model: torch.nn.Module) -> List[str]:
    """Names of ODINN aggregation parameters that receive zero weight decay."""
    names: List[str] = []
    for name, _ in model.named_parameters():
        if name.endswith("alpha_raw") or name.endswith("beta_raw"):
            names.append(name)
    return names


def split_dynamics_parameters(model: torch.nn.Module):
    dynamics = []
    ordinary = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("alpha_raw") or name.endswith("beta_raw"):
            dynamics.append(p)
        else:
            ordinary.append(p)
    if not dynamics:
        raise RuntimeError(
            "No ODINN dynamics parameter found. Expected alpha_raw or beta_raw."
        )
    return ordinary, dynamics


def build_gpr_style_adam(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    dynamics_weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Adam with separate (default zero) weight decay on alpha/beta."""
    ordinary, dynamics = split_dynamics_parameters(model)
    return torch.optim.Adam(
        [
            {"params": ordinary, "lr": float(lr), "weight_decay": float(weight_decay)},
            {"params": dynamics, "lr": float(lr), "weight_decay": float(dynamics_weight_decay)},
        ]
    )
