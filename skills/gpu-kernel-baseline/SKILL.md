---
name: gpu-kernel-baseline
description: Learn the target framework from gpu-wiki and implement a baseline GPU kernel. Use this skill to understand compute semantics, determine the target platform and framework, search reference implementations, and produce a correct V0 baseline with performance records for later profile-driven optimization.
---

# GPU Kernel Baseline

## When to Use

Use this skill when the user provides PyTorch logic or a kernel demo and asks to:

- Write a GPU kernel for the target platform.
- Build a baseline from scratch.
- Prepare `kernel.py`, `reference.py`, `test_kernel.py`, and `baseline_report.md` for later profile-driven optimization.

## Workflow

This stage first understands the PyTorch semantics, then learns the framework APIs (CuteDSL or FlyDSL) through `<gpu-wiki>/README.md`, implements `kernel.py` and `test_kernel.py`, validates correctness, records performance, writes `baseline_report.md`, and writes `memory/v0.json`.

The orchestrator exposes the knowledge base at `./gpu-wiki/` inside each campaign workspace,
referenced below as `<gpu-wiki>/`.

## Phase 1: Understand PyTorch Semantics

1. Read the user-provided PyTorch logic and `kernel_demo`.
2. Extract and record:
   - Compute pattern, such as `GEMM`, `Decode Attention`, `Reduction`, or `Elementwise`.
   - Input/output shape, stride, dtype, layout, and device.
   - Data dependencies, broadcasting, masks, boundary handling, and write-back semantics.
   - Accuracy requirements, tolerance, accumulation dtype, and special-value handling.
3. Determine target platform and framework:
   - H100/H20/H200 -> Hopper -> `CuteDSL`
   - MI300X/MI308X -> CDNA3 -> `FlyDSL`
   - MI355X -> CDNA4 -> `FlyDSL`
4. If the PyTorch logic is ambiguous, first create a minimal runnable reference, then continue.

## Phase 2: Learn Framework APIs from gpu-wiki

1. Read `<gpu-wiki>/README.md`, then use the natural-language front door:
   ```bash
   python3 gpu-wiki/tools/query_nl.py "<your description>" --brief
   ```
2. State the true target product and authoritative runtime architecture exactly as supplied. Explicitly
   request the complete product specification and relevant architecture/ISA facts, plus the framework,
   operator, shapes, dtypes, intended implementation, and uncertainties. Do not substitute another
   hardware identity or reduce the description to keywords.
3. Results contain a top-level `query_id`, `records`, and `notes`. Read each id-keyed record's own
   canonical `wiki_id`, `store`, independent `payload`, `source`, `type`, and `match.arch`; internal
   mapping keys use the `internal_gpu_wiki::` namespace. Copy emitted attribution ids exactly and use
   `--max-bytes` when a hard context limit is needed.
4. Prefer records with the same framework and compute pattern. Record the stable ids and the constraints
   they established in `plans/v0_plan.md`.

## Phase 3: Implement Baseline Kernel and Correctness Tests

1. Implement a correct baseline `kernel.py` based on PyTorch semantics and the learned framework APIs.Not only must the functionality be correct, but the framework implementation must also be correct, using either CuteDSL or FlyDSL.
2. Write `test_kernel.py` using PyTorch logic directly as the correctness reference.
3. Cover representative inputs, including normal shapes, boundary shapes, and relevant dtype or stride cases.
4. Example correctness check:

```python
ref = pytorch_reference(inputs)
out = kernel_v1(inputs)
rel_err = (out.float() - ref).norm() / ref.norm()
assert rel_err < 0.01
```

5. The default BF16 threshold is `rel_err < 0.01`; lower precision formats may use task-specific relaxed thresholds.
6. Add per-case timeout guard in `test_kernel.py` to prevent hanging:

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test case exceeded timeout limit")

signal.signal(signal.SIGALRM, timeout_handler)

TIMEOUT_SEC = int(os.environ.get("TEST_TIMEOUT_SEC", "30"))

for case in test_cases:
    signal.alarm(TIMEOUT_SEC)
    try:
        run_test(case)
    except TimeoutError:
        record_failure(case, "TIMEOUT_FAIL")
    finally:
        signal.alarm(0)
```
6. If API, compilation, accuracy, performance, or hardware issues appear, query `<gpu-wiki>/` again with
   the exact failure and measured evidence, and then fix the implementation.
7. Record the baseline configuration, including tile size, thread organization, grid/block design, and major data-movement patterns.

## Phase 4: Performance, Correctness, and Quality Gate

1. Run exactly one full-workload base-seed V0 measurement through the mandatory sandbox. Do not pass
   `--multi-seed` and do not launch a separate robustness run for V0:

```bash
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version v0 --no-memory
```

   Parse the emitted `[test_kernel] RESULT_JSON=...`, use its performance result and accompanying
   correctness status for `memory/v0.json`, and avoid repeating the expensive baseline workload.

   - Each individual test case must complete within **30 seconds** (configurable via `TEST_TIMEOUT_SEC` env var).
   - If a case exceeds the timeout, mark it as `TIMEOUT_FAIL`, kill the process, and record the failure in `baseline_report.md`.
   - Common timeout causes: infinite loops in index calculation, deadlocks in synchronization, or excessive compilation time. Query gpu-wiki with the failure mode to diagnose.

2. Verify all correctness cases pass and record max `rel_err` plus PASS/FAIL.
3. Measure baseline performance and record:

```text
latency(us) | TFLOPS | bandwidth(GB/s) | TFLOPS peak utilization(%) | bandwidth peak utilization(%)
```

4. Use `compute_utilization.py` to calculate TFLOPS and bandwidth utilization:

```bash
python tools/compute_utilization.py   --gpu <gpu> --dtype <dtype>   --flops-expr '<expr>' --bytes-expr '<expr>'   --time-ms <ms> --grid-blocks <blocks>
```

5. Every theoretical peak, bandwidth, and utilization calculation must cite the gpu-wiki spec sources registered in Step 0.
6. Write `baseline_report.md` with:
   - Baseline kernel path
   - Correctness test path
   - PyTorch reference logic description
   - Stable gpu-wiki record ids consulted
   - Baseline configuration summary
   - Correctness results: case list, max `rel_err`, PASS/FAIL (include any TIMEOUT_FAIL cases)
   - Baseline performance: latency(us), TFLOPS, bandwidth(GB/s), and peak utilization percentages
7. Write baseline iteration data to `memory/v0.json` using `tools/memory_manager.py`:

   ```bash
   # Create the iteration file
   python tools/memory_manager.py create --workspace kernel_opt_<name> --version v0

   # Fill in performance and metadata
   python tools/memory_manager.py update --workspace kernel_opt_<name> --version v0 \
       --set 'performance.latency_us=<value>' \
       --set 'performance.tflops=<value>' \
       --set 'performance.bandwidth_gbps=<value>' \
       --set 'performance.tflops_peak_utilization_pct=<value>' \
       --set 'performance.bandwidth_peak_utilization_pct=<value>' \
       --set 'optimization.action_category=baseline' \
       --set 'optimization.action_description=<summary>' \
       --set 'correctness.rel_err=<value>' \
       --set 'correctness.status=PASS' \
       --set 'quality_gate.result=PASS'
   ```

   For array fields (`pitfalls_and_fixes`, `references`), update the JSON file directly or use `read` + manual edit + write-back. Fill in:
   - `pitfalls_and_fixes`: any errors encountered during implementation
   - `references`: stable gpu-wiki record ids and other docs referenced during learning

8. After the quality gate passes, commit:

```bash
git add kernel.py test_kernel.py baseline_report.md memory/v0.json README.md
git commit -m "V0: baseline kernel"
```

## memory/ Requirements

Each iteration produces a `memory/v<N>.json` file following the schema defined in `reference/v_iteration.schema.json`. The JSON structure captures performance data, optimization actions, profile evidence, correctness results, ISA metric progress, search logs, pitfalls and fixes, and references.

Key rules:
- The `masked` field defaults to `false`. When set to `true`, the file is skipped during reads.
- ISA optimization target thresholds are stored in `README.md` and must be derived from `<gpu-wiki>/` best practices, hardware specs, and Step 0 Roofline conclusions. Do not fabricate thresholds from experience.

## Deliverables

- Runnable and correct(using either CuteDSL or FlyDSL) `kernel.py`
- PyTorch `reference.py`
- `test_kernel.py`
- `baseline_report.md`
- Created `memory/v0.json`
- Git commit

## Appendix: Prohibited Actions

- Do not use unspecified programming frameworks or import external projects.
