import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference semantics for y = x + alpha * residual."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return x + self.alpha * residual
