import triton
import triton.language as tl


@triton.jit
def add_one(x, out, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < size, other=0)
    tl.store(out + offsets, values + 1, offsets < size)
