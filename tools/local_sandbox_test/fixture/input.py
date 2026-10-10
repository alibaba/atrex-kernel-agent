import torch


def _make_inputs(n: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(n, dtype=torch.float32, device="cuda"),
        "y": torch.randn(n, dtype=torch.float32, device="cuda"),
    }
