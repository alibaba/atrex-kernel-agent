import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y
