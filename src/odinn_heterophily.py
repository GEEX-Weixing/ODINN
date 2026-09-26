from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

from .odinn_weight_init import gpr_like_init


def row_propagate(x: torch.Tensor, edge_index: torch.Tensor, add_loops: bool = False) -> torch.Tensor:
    n = x.size(0)
    ei = edge_index
    if add_loops:
        ei, _ = add_self_loops(ei, num_nodes=n)
    src, dst = ei
    deg = degree(dst, n, dtype=x.dtype).clamp_min(1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] / deg[dst].unsqueeze(-1))
    return out


def lazy_propagate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + row_propagate(x, edge_index, add_loops=True))


class ODINNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.l2(h))


class ODINNDG(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.6, hops=20,
                 weight_init="ppr", ppr_alpha=0.1, nppr_alpha=-0.5):
        super().__init__()
        self.enc = ODINNEncoder(in_dim, hidden, dropout)
        self.hops = int(hops)
        self.weight_init = weight_init
        self.ppr_alpha = float(ppr_alpha)
        self.nppr_alpha = float(nppr_alpha)
        self.alpha_raw = nn.Parameter(torch.empty(self.hops))
        self.reset_dynamics_parameters()
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def reset_dynamics_parameters(self):
        with torch.no_grad():
            self.alpha_raw.copy_(gpr_like_init(
                self.hops, self.weight_init,
                ppr_alpha=self.ppr_alpha, nppr_alpha=self.nppr_alpha,
                device=self.alpha_raw.device, dtype=self.alpha_raw.dtype,
            ))

    def forward(self, x, edge_index):
        z = self.enc(x)
        h = lazy_propagate(z, edge_index)
        agg = self.alpha_raw[0] * h
        for k in range(1, self.hops):
            h = lazy_propagate(h, edge_index)
            agg = agg + self.alpha_raw[k] * h
        return self.out(F.dropout(agg, p=self.dropout, training=self.training))


class ODINNFJ(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.6, hops=20,
                 weight_init="ppr", ppr_alpha=0.1, nppr_alpha=-0.5):
        super().__init__()
        self.enc = ODINNEncoder(in_dim, hidden, dropout)
        self.hops = int(hops)
        self.weight_init = weight_init
        self.ppr_alpha = float(ppr_alpha)
        self.nppr_alpha = float(nppr_alpha)
        self.beta_raw = nn.Parameter(torch.empty(max(0, self.hops - 1)))
        self.reset_dynamics_parameters()
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def reset_dynamics_parameters(self):
        if self.beta_raw.numel() == 0:
            return
        with torch.no_grad():
            self.beta_raw.copy_(gpr_like_init(
                self.beta_raw.numel(), self.weight_init,
                ppr_alpha=self.ppr_alpha, nppr_alpha=self.nppr_alpha,
                device=self.beta_raw.device, dtype=self.beta_raw.dtype,
            ))

    def forward(self, x, edge_index):
        h = lazy_propagate(self.enc(x), edge_index)
        for beta in self.beta_raw:
            wh = lazy_propagate(h, edge_index)
            h = (1.0 - beta) * wh + beta * h
        return self.out(F.dropout(h, p=self.dropout, training=self.training))


def build_odinn(name, in_dim, hidden, out_dim, dropout=0.6, hops=20):
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"dg", "odinndg"}:
        return ODINNDG(in_dim, hidden, out_dim, dropout, hops)
    if key in {"fj", "odinnfj"}:
        return ODINNFJ(in_dim, hidden, out_dim, dropout, hops)
    raise ValueError(f"Unknown ODINN variant: {name}")
