from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_ops import spmm
from .odinn_weight_init import gpr_like_init

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
        return F.relu(self.lin2(x))  # non-negative opinion/node embedding

class ODINNDG(nn.Module):
    """ODINN-DG with unconstrained real-valued alpha hop weights."""
    def __init__(self, in_dim, num_classes, hops=20, hidden=64, dropout=0.6,
                 weight_init="ppr", ppr_alpha=0.1, nppr_alpha=-0.5):
        super().__init__()
        self.hops = int(hops)
        self.encoder = FeatureMLP(in_dim, hidden, hidden, dropout)
        self.weight_init = str(weight_init)
        self.ppr_alpha = float(ppr_alpha)
        self.nppr_alpha = float(nppr_alpha)
        self.alpha_raw = nn.Parameter(torch.empty(self.hops))
        self.reset_dynamics_parameters()
        self.classifier = nn.Linear(hidden, num_classes)

    def reset_dynamics_parameters(self):
        with torch.no_grad():
            self.alpha_raw.copy_(gpr_like_init(
                self.hops, self.weight_init,
                ppr_alpha=self.ppr_alpha, nppr_alpha=self.nppr_alpha,
                device=self.alpha_raw.device, dtype=self.alpha_raw.dtype,
            ))

    def effective_alpha(self):
        return self.alpha_raw

    def forward(self, x, w_lazy):
        z = self.encoder(x)
        a = self.effective_alpha()
        h = spmm(w_lazy, z)
        agg = a[0] * h
        for k in range(1, self.hops):
            h = spmm(w_lazy, h)
            agg = agg + a[k] * h
        return self.classifier(agg)


class ODINNFJ(nn.Module):
    """ODINN-FJ with unconstrained real-valued beta hop weights.

    beta is used directly in Eq. (18); there is no sigmoid, clamp, softmax or
    other numerical constraint.  The attribute beta_logits is retained as a
    read-only compatibility alias for older experiment scripts.
    """
    def __init__(self, in_dim, num_classes, hops=20, hidden=64, dropout=0.6,
                 weight_init="ppr", ppr_alpha=0.1, nppr_alpha=-0.5):
        super().__init__()
        self.hops = int(hops)
        self.encoder = FeatureMLP(in_dim, hidden, hidden, dropout)
        self.weight_init = str(weight_init)
        self.ppr_alpha = float(ppr_alpha)
        self.nppr_alpha = float(nppr_alpha)
        self.beta_raw = nn.Parameter(torch.empty(max(0, self.hops - 1)))
        self.reset_dynamics_parameters()
        self.classifier = nn.Linear(hidden, num_classes)

    @property
    def beta_logits(self):
        # Compatibility with older analysis code; this is NOT a logit anymore.
        return self.beta_raw

    def reset_dynamics_parameters(self):
        if self.beta_raw.numel() == 0:
            return
        with torch.no_grad():
            self.beta_raw.copy_(gpr_like_init(
                self.beta_raw.numel(), self.weight_init,
                ppr_alpha=self.ppr_alpha, nppr_alpha=self.nppr_alpha,
                device=self.beta_raw.device, dtype=self.beta_raw.dtype,
            ))

    def effective_beta(self):
        return self.beta_raw

    def forward(self, x, w_lazy):
        z = self.encoder(x)
        h = spmm(w_lazy, z)
        if self.hops > 1:
            for b in self.effective_beta():
                wh = spmm(w_lazy, h)
                h = (1.0 - b) * wh + b * h
        return self.classifier(h)

def build_odinn(name: str, in_dim: int, num_classes: int, hops: int, hidden: int=64,
                dropout: float=0.6, weight_init: str="ppr", ppr_alpha: float=0.1,
                nppr_alpha: float=-0.5):
    n = name.lower().replace('-', '_')
    kwargs = dict(weight_init=weight_init, ppr_alpha=ppr_alpha, nppr_alpha=nppr_alpha)
    if n in {'dg','degroot','odinn_dg'}:
        return ODINNDG(in_dim, num_classes, hops, hidden, dropout, **kwargs)
    if n in {'fj','friedkin_johnsen','odinn_fj'}:
        return ODINNFJ(in_dim, num_classes, hops, hidden, dropout, **kwargs)
    raise ValueError(name)
