import torch


def _make_inputs(N=1024, device="cuda", **kwargs):
    return {"x": torch.randn(N, device=device, dtype=torch.float32)}
