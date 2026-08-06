import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)


def run(x, out):
    _copy_kernel[(1,)](x, out, n=x.numel(), BLOCK=triton.next_power_of_2(x.numel()))
