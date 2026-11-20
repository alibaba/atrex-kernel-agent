import torch


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _make_inputs(n, dtype="float32", device="cuda"):
    tensor_dtype = _DTYPES[dtype]
    x = torch.randn((n,), device=device, dtype=tensor_dtype)
    y = torch.randn((n,), device=device, dtype=tensor_dtype)
    return {"x": x, "y": y}
