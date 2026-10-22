"""Plain MLP. Architecture is held fixed across methods so only the loss mechanism varies."""

from __future__ import annotations

import torch
import torch.nn as nn


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,  # PINN default: smooth higher derivatives
    "sin": Sin,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "relu": nn.ReLU,  # second derivative is zero a.e. -- unusable for AD residuals
}


class MLP(nn.Module):
    """(in_dim) -> width x depth hidden layers -> (out_dim)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
        final_bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; have {sorted(ACTIVATIONS)}")
        act = ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, out_dim, bias=final_bias))
        self.net = nn.Sequential(*layers)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.reset_parameters(activation)

    def reset_parameters(self, activation: str = "tanh") -> None:
        """Xavier for saturating activations, Kaiming otherwise; zero biases."""
        gain = nn.init.calculate_gain("tanh") if activation in ("tanh", "sin") else 1.0
        for m in self.net:
            if isinstance(m, nn.Linear):
                if activation in ("tanh", "sin", "softplus"):
                    nn.init.xavier_normal_(m.weight, gain=gain)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
