# --reference-dir: atrex-bench operator packages (`agate eval --backend=atrex`)

**Use this path for custom operators.** Unlike SOL, you upload the problem (reference implementation, input generator and shape list)
with the request, so evaluation is not limited to a predefined problem catalog.

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
```

`--candidate` supplies the implementation to evaluate. `--reference-dir` supplies the correctness reference and shapes.

## Directory contents

Provide `reference.py`, `input.py` and `shapes.json`. `metadata.json` and `roofline.json` are optional;
the gateway supplies a default metadata file required upstream:

```
data/rms_norm/
├── reference.py     # Reference: class Model(nn.Module)
├── input.py         # Input generator: _make_inputs(**input_kwargs) -> dict[str, Tensor]
├── shapes.json      # Shapes to evaluate
├── metadata.json    # Optional operator metadata; gateway supplies defaults
└── roofline.json    # Optional; required to calculate sol.pct
```

Minimal structures for required files and optional metadata appear below.
With repository access, you can also consult
[`examples/rms_norm/`](https://code.alibaba-inc.com/tre-infra/atrex-gpu-gateway/tree/master/examples/rms_norm),
but repository access is not a prerequisite for using this Skill.

### reference.py

Expose a top-level `class Model(nn.Module)`, constructed with `Model(**init_kwargs)` and invoked with `forward(**inputs)`:

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * weight
```

**The Python `--candidate` must also expose `class Model` with matching signatures.**
Candidate, reference, input generator and ABBA baseline may use ordinary variables/parameters named `input`,
such as `forward(self, *, input)`. Calling the interactive built-in `input()` is still prohibited.

### input.py

Define top-level `_make_inputs` with parameter names matching `shapes.json`'s `input_kwargs`, returning a tensor dictionary:

```python
import torch


def _make_inputs(token_count: int, hidden_size: int) -> dict[str, torch.Tensor]:
    x = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02 + 1.0
    return {'x': x, 'weight': weight}
```

Returned keys must match `Model.forward` argument names. Construct domain constraints here, such as legal block counts,
masks and cache layouts; that is why this input is Python code rather than a declaration.

The default source limit is **16 KiB = 16384 UTF-8 bytes**, including comments and whitespace, not characters or generated tensor size.
Deployments may independently adjust `max_input_source_bytes`, effective after restart.
Check `limits.max_input_source_bytes` in `GET /v1/env/{gpu}/capabilities` for the actual limit.
Excess is rejected before queueing with `400 invalid_source`, identifying actual/allowed byte counts for `reference.input_py`.
Candidate and `reference.py` each default to 128 KiB under separate `limits.max_source_bytes`; they do not enlarge the input-generator budget.

### shapes.json

The top-level dictionary maps string shape IDs to constructor arguments and input arguments:

```json
{
  "0": {
    "init_kwargs": { "eps": 1e-06 },
    "input_kwargs": { "token_count": 2500, "hidden_size": 1024 }
  }
}
```

`init_kwargs` is passed to `Model(**init_kwargs)`; `input_kwargs` is passed to `_make_inputs(**input_kwargs)`.
Add more keys for additional shapes.

### metadata.json

Contains operator metadata such as category, dtype and provenance.
**Optional for callers, but the upstream gate requires the file**, so the gateway always materializes it. Supplying `{}` is valid.

### roofline.json (optional)

With this file, each shape's result includes `sol.pct`, attainment relative to your supplied roofline:

```json
{ "shapes": { "0": { "SOL_time_ms": { "NVIDIA L20": 0.0123 } } } }
```

Hardware names must match `environment.gpu_name` in the result. If it cannot be calculated,
`sol.pct` is null with an explanation in `sol.reason`; zero is not fabricated.

Here `sol` means **speed of light (performance-limit attainment)**. It is unrelated to the SOL-ExecBench catalog selected by `--backend=sol`;
`profile --level sol` is another use of the same term.

## Reading results

Results contain `compile`, `correctness` and `performance` sections.
**Use `passed`, not `error`, to decide success**: upstream may leave `error=None` on compilation failure.

Each shape also reports:

- `timing_stats.cv`: dispersion. A large CV makes comparisons unreliable.
  **Null means only one sample and unknown dispersion**, not exceptional stability. `runs` is the actual sample count.
- `environment.clock_sm_pct_of_max` and `clock_throttle_reasons`: measured clock state for judging measurement confidence.

## Choosing the backend

| Situation | Backend |
|---|---|
| Your own operator with a reference implementation | `--backend=atrex` (this document; default) |
| An implementation for one of SOL's 235 existing problems | `--backend=sol`; see [solution-json.md](solution-json.md) |
