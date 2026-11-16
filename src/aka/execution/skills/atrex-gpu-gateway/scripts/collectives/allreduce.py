"""NCCL smoke candidate; replace forward with your allreduce implementation."""
from __future__ import annotations

import torch
import torch.distributed as dist


class Model(torch.nn.Module):
    def forward(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        dist.all_reduce(tensor, group=group)
        return tensor
