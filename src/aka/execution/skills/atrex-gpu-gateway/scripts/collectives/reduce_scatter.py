"""NCCL smoke candidate; sum all inputs and return this rank's contiguous shard."""
from __future__ import annotations

import torch
import torch.distributed as dist


class Model(torch.nn.Module):
    def forward(self, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        output = torch.empty(tensor.numel() // dist.get_world_size(group),
                             dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter_tensor(output, tensor, group=group)
        return output
