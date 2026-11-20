import torch


def _make_inputs(n):
    return {"x": torch.randn(n, dtype=torch.float32, device="cuda")}
