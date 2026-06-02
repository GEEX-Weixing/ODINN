import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A small MLP used before ODINN propagation."""

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.layers[-1](h)
