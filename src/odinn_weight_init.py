from __future__ import annotations

import math
from typing import Iterable, List

import torch

INIT_CHOICES = ("ppr", "nppr", "random")


def canonical_init(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "ppr": "ppr",
        "positive_ppr": "ppr",
        "nppr": "nppr",
        "negative_ppr": "nppr",
        "signed_ppr": "nppr",
        "random": "random",
        "rand": "random",
    }
    if key not in aliases:
        raise ValueError(f"Unknown ODINN weight initialization {name!r}; choose from {INIT_CHOICES}")
    return aliases[key]


def gpr_like_init(
    length: int,
    init: str,
    *,
    ppr_alpha: float = 0.1,
    nppr_alpha: float = -0.5,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a GPR-GNN-inspired initialization for an ODINN hop-weight vector.

    The three presets intentionally mirror three initializers present in the
    original GPR-GNN implementation:

    ppr:
        Positive PPR-like coefficients.  With n coefficients, entries 0..n-2
        are a(1-a)^k and the final entry is (1-a)^(n-1), so the coefficients
        sum to one.  We use a=0.1 by default, a standard value evaluated by
        GPR-GNN.  This is an initialization only; no sum-to-one constraint is
        applied during ODINN training.

    nppr:
        Signed / Negative-PPR coefficients rho^k, L1-normalized exactly in the
        style of GPR-GNN's NPPR initializer.  The default rho=-0.5 gives a
        decaying alternating-sign starting point and is useful for checking
        whether unconstrained ODINN weights remain in a signed basin.

    random:
        Uniform U[-sqrt(3/n), sqrt(3/n)] followed by L1 normalization, matching
        GPR-GNN's Random initializer.  Randomness follows PyTorch's global RNG,
        so calling the project's set_seed(seed) before model construction makes
        it fully reproducible.

    Important: these are *initializations*, not constraints.  After model
    construction alpha and beta are ordinary real-valued Parameters.
    """
    n = int(length)
    if n < 0:
        raise ValueError("length must be non-negative")
    if n == 0:
        return torch.empty(0, device=device, dtype=dtype)

    kind = canonical_init(init)
    if kind == "ppr":
        a = float(ppr_alpha)
        if not (0.0 < a < 1.0):
            raise ValueError("ppr_alpha must lie in (0,1)")
        out = torch.empty(n, device=device, dtype=dtype)
        if n == 1:
            out[0] = 1.0
        else:
            k = torch.arange(n - 1, device=device, dtype=dtype)
            out[:-1] = a * torch.pow(torch.tensor(1.0 - a, device=device, dtype=dtype), k)
            out[-1] = (1.0 - a) ** (n - 1)
        return out

    if kind == "nppr":
        rho = float(nppr_alpha)
        if rho == 0.0:
            raise ValueError("nppr_alpha must be non-zero; a negative value is recommended")
        k = torch.arange(n, device=device, dtype=dtype)
        out = torch.pow(torch.tensor(rho, device=device, dtype=dtype), k)
        return out / out.abs().sum().clamp_min(torch.finfo(dtype).eps)

    bound = math.sqrt(3.0 / n)
    out = torch.empty(n, device=device, dtype=dtype).uniform_(-bound, bound)
    denom = out.abs().sum()
    if float(denom) == 0.0:  # practically impossible, but keep reset robust
        out.zero_()
        out[0] = 1.0
        return out
    return out / denom


def dynamics_parameter_names(model: torch.nn.Module) -> List[str]:
    """Names of ODINN hop-weight Parameters that should receive zero WD."""
    names: List[str] = []
    for name, _ in model.named_parameters():
        if name.endswith("alpha_raw") or name.endswith("beta_raw"):
            names.append(name)
    return names


def split_dynamics_parameters(model: torch.nn.Module):
    """Split trainable params into ordinary network params and hop weights."""
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
            "No ODINN dynamics parameter found. Expected a Parameter named alpha_raw or beta_raw."
        )
    return ordinary, dynamics


def build_gpr_style_adam(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    dynamics_weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Adam with GPR-GNN-style zero weight decay on propagation weights.

    The hop weights use the same learning rate as the rest of the network, as
    in GPR-GNN.  Only their L2/weight-decay coefficient is separated.  Keeping
    the same LR avoids adding another tuning dimension to the requested sweep.
    """
    ordinary, dynamics = split_dynamics_parameters(model)
    return torch.optim.Adam(
        [
            {"params": ordinary, "lr": float(lr), "weight_decay": float(weight_decay)},
            {"params": dynamics, "lr": float(lr), "weight_decay": float(dynamics_weight_decay)},
        ]
    )
