"""Deterministic GPU/JIT/JSON-output probe executed through the real sandbox."""
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def add_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, values, mask=mask)


def main():
    torch.manual_seed(7)
    cases = []
    reference_latencies = {}
    for n in (512, 1025, 4096):
        x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x + y, atol=0, rtol=0)
        cases.append(n)
        reference_latencies[str(len(cases) - 1)] = float(do_bench(lambda: x + y)) * 1000
    result = {"status": "PASS", "cases": cases, "device": torch.cuda.get_device_name(),
              "reference_latency_us_by_shape": reference_latencies,
              "reference_measurement": "triton.testing.do_bench(torch.add), milliseconds * 1000"}
    Path("smoke-result.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
