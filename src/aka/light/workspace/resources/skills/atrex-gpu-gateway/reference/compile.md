# Compilation checks: agate check and compile jobs

Use this for quick import/construction diagnostics. Use `eval` for correctness or performance conclusions and `profile` for hotspot analysis.

## Entry point and inputs

```bash
agate check my_kernel.py --gpu L20D
agate check my_kernel.py --gpu L20D --init-kwargs init_kwargs.json
agate check my_kernel.py --gpu L20D -r requirements.txt --no-wait
agate get <job_id> --wait
```

The CLI subcommand is `check`, the job kind is `compile`, and the HTTP endpoint is `POST /v1/jobs/compile`.
The candidate is a single Python file exposing a top-level `class Model`. `--gpu` is required: execution happens in the target worker environment, not as a local static check.
AST validation runs at submission; rejected requests return 400 without first queueing for a GPU.

`init_kwargs.json` must be a JSON object, defaulting to `{}`, and must not exceed 64 KiB after compact encoding. For example:

```json
{"hidden_size": 1024, "eps": 0.000001}
```

It is passed to `Model(**init_kwargs)`, not read automatically from `shapes.json` or supplied to `forward`.
These are ordinary job inputs; do not include passwords or tokens.
`check` does not read `reference.py`, `input.py` or `shapes.json`. Its candidate uses the separate source-code limit.
Use `-r` for extra dependencies; the default `freeze_installed` reuses packages already satisfied by the base environment.

## Execution scope

The worker imports the candidate, constructs `Model(**init_kwargs)` and collects compilation artifacts from the job's isolated cache.
AOT compilation triggered during construction runs, but the driver does not automatically invoke `forward` or generate test inputs.

Implementations such as Triton that JIT-compile on first launch may only complete import and construction, with no kernel artifacts.
In that case, `ok=true` means the executed path produced no diagnostic errors; it does not mean all kernels compiled, passed correctness or met performance requirements.
Use `eval`/`profile` for real inputs. `disassemble` shares this import/construction behavior and does not synthesize JIT inputs either.

## Reading results

Inspect these fields together in the job's `result`:

| Field | Meaning |
|---|---|
| `ok` | Whether import/construction completed without diagnostic errors; not an evaluation-pass flag |
| `diagnostics[]` | Bounded compilation/import diagnostics with `file`, `line`, `severity` and `message` |
| `ptxas` | Statistics such as `regs`, `spills` and `smem` when artifacts exist; empty means unavailable, not zero resource use |
| `asm_available`, `instruction_histogram` | Whether assembly was found and its instruction counts; not a substitute for latency or throughput measurements |
| `notes[]` | Diagnostic limits such as untriggered JIT, unavailable tools or architecture coverage |

When statistics exist for multiple Triton kernels, `ptxas` reports the heaviest kernel, ranked first by spills and then register count,
and `kernels_compiled` reports the number found. These are not averages across kernels.
If the driver cannot parse a result, it returns `ok=false` with diagnostics; do not interpret that as successful compilation.

## Architecture and sanitizers

```bash
agate check my_kernel.py --gpu H20 --arch sm_90
agate check my_kernel.py --gpu H20 --sanitize memcheck
```

`--arch` maps to toolchain-supported `TRITON_OVERRIDE_ARCH`/`TORCH_CUDA_ARCH_LIST` settings.
Not every framework reads them; for example, CuteDSL selects architecture through its own compiler arguments.
Use `arch_requested`, `arch_enforced_by` and `notes` to determine actual coverage. Untriggered JIT supplies no compilation evidence.

`--sanitize` accepts `memcheck`, `racecheck`, `initcheck` or `synccheck` and invokes the target worker's vendor tools with a longer execution budget.
It still does not invoke `forward` automatically. A kernel can be checked only if the candidate itself executes it during import/construction.
Inspect `sanitize.checked` and `sanitize.note`; `checked=false` means there was nothing to check, not that memory or race checks passed.
