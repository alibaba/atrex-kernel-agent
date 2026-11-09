import torch


def _make_inputs(numel: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(numel, dtype=torch.bfloat16, device="cuda"),
        "residual": torch.randn(numel, dtype=torch.bfloat16, device="cuda"),
    }
