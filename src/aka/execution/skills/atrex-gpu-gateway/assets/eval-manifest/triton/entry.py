import torch
from .kernel import add_one


class Model(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        add_one[(triton_grid(x.numel()),)](x, out, x.numel(), BLOCK=256)
        return out


def triton_grid(size):
    return (size + 255) // 256
