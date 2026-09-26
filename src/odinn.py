from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_ops import spmm
from .odinn_weight_init import balanced_init, ppr_init


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.6):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.lin2(x))


class ODINNDG(nn.Module):
    """DeGroot-inspired ODINN with unconstrained real-valued alpha weights."""

    def __init__(
        self,
        in_dim,
        num_classes,
        hops=20,
        hidden=64,
        dropout=0.6,
        ppr_alpha=0.1,
    ):
        super().__init__()
        self.hops = int(hops)
        self.encoder = FeatureMLP(in_dim, hidden, hidden, dropout)
        self.ppr_alpha = float(ppr_alpha)
        self.alpha_raw = nn.Parameter(torch.empty(self.hops))
        self.reset_dynamics_parameters()
        self.classifier = nn.Linear(hidden, num_classes)

    def reset_dynamics_parameters(self):
        with torch.no_grad():
            self.alpha_raw.copy_(
                ppr_init(
                    self.hops,
                    self.ppr_alpha,
                    device=self.alpha_raw.device,
                    dtype=self.alpha_raw.dtype,
                )
            )

    def effective_alpha(self):
        return self.alpha_raw

    def forward(self, x, w_lazy):
        z = self.encoder(x)
        h = spmm(w_lazy, z)
        agg = self.alpha_raw[0] * h
        for k in range(1, self.hops):
            h = spmm(w_lazy, h)
            agg = agg + self.alpha_raw[k] * h
        return self.classifier(agg)


class ODINNFJ(nn.Module):
    """Friedkin--Johnsen-inspired ODINN with unconstrained beta weights."""

    def __init__(
        self,
        in_dim,
        num_classes,
        hops=20,
        hidden=64,
        dropout=0.6,
        beta_center=0.5,
        beta_std=0.01,
    ):
        super().__init__()
        self.hops = int(hops)
        self.encoder = FeatureMLP(in_dim, hidden, hidden, dropout)
        self.beta_center = float(beta_center)
        self.beta_std = float(beta_std)
        self.beta_raw = nn.Parameter(torch.empty(max(0, self.hops - 1)))
        self.reset_dynamics_parameters()
        self.classifier = nn.Linear(hidden, num_classes)

    @property
    def beta_logits(self):
        # Compatibility alias only; beta is not transformed by sigmoid.
        return self.beta_raw

    def reset_dynamics_parameters(self):
        if self.beta_raw.numel() == 0:
            return
        with torch.no_grad():
            self.beta_raw.copy_(
                balanced_init(
                    self.beta_raw.numel(),
                    center=self.beta_center,
                    std=self.beta_std,
                    device=self.beta_raw.device,
                    dtype=self.beta_raw.dtype,
                )
            )

    def effective_beta(self):
        return self.beta_raw

    def forward(self, x, w_lazy):
        z = self.encoder(x)
        h = spmm(w_lazy, z)
        if self.hops > 1:
            for beta in self.beta_raw:
                wh = spmm(w_lazy, h)
                h = (1.0 - beta) * wh + beta * h
        return self.classifier(h)


def build_odinn(
    name: str,
    in_dim: int,
    num_classes: int,
    hops: int,
    hidden: int = 64,
    dropout: float = 0.6,
):
    key = name.lower().replace("-", "_")
    if key in {"dg", "degroot", "odinn_dg"}:
        return ODINNDG(
            in_dim, num_classes, hops, hidden, dropout,
            ppr_alpha=0.1,
        )
    if key in {"fj", "friedkin_johnsen", "odinn_fj"}:
        return ODINNFJ(
            in_dim, num_classes, hops, hidden, dropout,
            beta_center=0.5, beta_std=0.01,
        )
    raise ValueError(name)
