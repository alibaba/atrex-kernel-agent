import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1
