import torch
from torch.utils.cpp_extension import load

_extension = load(
    name="agate_manifest_add_one",
    sources=[__file__.rsplit("/", 1)[0] + "/" + name for name in ("binding.cpp", "kernel.cu")],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        return _extension.add_one(x)
