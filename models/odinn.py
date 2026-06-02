from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class DeGrootLayer(nn.Module):
    """ODINN DeGroot propagation block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_dim: int,
        num_layers: int,
        activation=F.elu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.activation = activation
        self.mlp = MLP(num_layers=2, input_dim=input_dim, hidden_dim=mlp_dim, output_dim=output_dim, dropout=dropout)
        self.alpha = nn.Parameter(torch.ones(num_layers), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        self.num_layers = num_layers

    def forward(self, inputs: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h0 = self.activation(self.mlp(inputs))
        alpha = self.activation(self.alpha)

        h = torch.sparse.mm(norm_adj, h0)
        agg = alpha[0] * h
        for i in range(1, self.num_layers):
            h = torch.sparse.mm(norm_adj, h)
            agg = agg + alpha[i] * h

        return self.activation(agg + self.bias)


class FriedkinJohnsenLayer(nn.Module):
    """ODINN Friedkin-Johnsen propagation block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_dim: int,
        num_layers: int,
        activation=F.elu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.activation = activation
        self.mlp = MLP(num_layers=2, input_dim=input_dim, hidden_dim=mlp_dim, output_dim=output_dim, dropout=dropout)
        self.stub = nn.Parameter(torch.ones(num_layers), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(num_layers), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        self.norms = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, inputs: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h0 = self.activation(self.mlp(inputs))
        stub = torch.sigmoid(self.stub)
        alpha = F.relu(self.alpha)

        propagated = torch.sparse.mm(norm_adj, h0)
        h = (1.0 - stub[0]) * propagated + stub[0] * h0
        h = self.norms[0](h)
        agg = alpha[0] * h

        for i in range(1, self.num_layers):
            propagated = torch.sparse.mm(norm_adj, h)
            h = (1.0 - stub[i]) * propagated + stub[i] * h0
            h = self.norms[i](h)
            agg = agg + alpha[i] * h

        return self.activation(agg + self.bias)


class ODINN_DeGroot(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_layers: int, hid_units: int, mlp_units: int, dropout: float):
        super().__init__()
        self.encoder = DeGrootLayer(input_dim, hid_units, mlp_units, num_layers, activation=F.elu, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_units, num_classes)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, norm_adj)
        h = self.dropout(h)
        return self.classifier(h)


class ODINN_FJ(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_layers: int, hid_units: int, mlp_units: int, dropout: float):
        super().__init__()
        self.encoder = FriedkinJohnsenLayer(input_dim, hid_units, mlp_units, num_layers, activation=F.elu, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid_units, num_classes)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, norm_adj)
        h = self.dropout(h)
        return self.classifier(h)


def build_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    num_layers: int,
    hid_units: int,
    mlp_units: int,
    dropout: float,
) -> nn.Module:
    normalized_name = model_name.lower().replace("-", "_")
    if normalized_name in {"degroot", "de_groot"}:
        return ODINN_DeGroot(input_dim, num_classes, num_layers, hid_units, mlp_units, dropout)
    if normalized_name in {"fj", "friedkin_johnsen", "friedkinjohnsen"}:
        return ODINN_FJ(input_dim, num_classes, num_layers, hid_units, mlp_units, dropout)
    raise ValueError("model_name must be one of: DeGroot, Friedkin_Johnsen, FJ")
