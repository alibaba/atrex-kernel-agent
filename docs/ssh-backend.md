# Light OpenSSH backend

Light can execute native Atrex-Bench Evaluate, same-allocation ABBA, Check,
Profile, Disassemble, Env and custom Dev commands on an SSH host. The Supervisor
and coding agent remain on the coordinator. Sources and the selected evaluator
are uploaded for each allocation; no gateway service or credentials are required
on the GPU host. Source trees and native CUDA manifests use the same candidate
snapshots and acceptance evidence as Gateway execution.

Each campaign selects one GPU on one machine. Different campaigns can target
NVIDIA, AMD or a provisioned PPU runtime independently. This is not a scheduler
for a campaign spanning different machines or GPU architectures.

## Start a campaign

```bash
aka optimize --repo-root "$PWD" -- \
  --op-dir /path/to/atrex-bench/operators/my_operator \
  --platform MI300X --sandbox-hardware MI300X --framework Triton \
  --sandbox-ssh gpu-worker --sandbox-ssh-gpu 5 \
  --sandbox-ssh-runtime-bind /opt/venv \
  --sandbox-ssh-runtime-bind /opt/rocm \
  --sandbox-ssh-init 'export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH; export LD_LIBRARY_PATH=/opt/rocm/lib' \
  --workspace /path/to/runs --max-iters 20
```

Use an OpenSSH host or `~/.ssh/config` alias. Authentication must work without
interactive input. The evaluator checkout is inferred from the operator directory.
The coordinator needs `ssh` and `scp`; the host needs Bash,
Python 3.10+, `tar`, `flock`, Bubblewrap and the selected GPU toolchain. Set
`ATREX_SANDBOX_SSH_BWRAP` to an absolute remote Bubblewrap executable when it is
not on the host's default PATH.

Runtime directories outside `/usr` need explicit read-only binds. Bind both the
alias and the canonical path if libraries contain absolute symlinks, for example
`/opt/rocm` and `/opt/rocm-7.2.0`. Python environments can be remapped with
`REMOTE_PATH=SANDBOX_PATH`; an environment under a home directory must be named
`.venv`, `venv`, or `env`, or be a direct child of a Conda `envs` directory.

## GPU selection and isolation

The default `ATREX_SANDBOX_SSH_GPU_ISOLATION=device` exposes only the assigned
GPU's device nodes. NVIDIA indices resolve to UUIDs and actual device minors;
AMD indices resolve through accessible KFD topology to UUIDs and render minors.
Do not assume `renderD128 + index` on partition-capable AMD machines.

On a trusted AMD machine where ROCm initializes all accessible render devices
before applying its visibility mask, explicitly select:

```bash
export ATREX_SANDBOX_SSH_GPU_ISOLATION=runtime
```

This retains the filesystem, process and network namespace boundaries, but
exposes the driver's render nodes and selects the assigned GPU with
`ROCR_VISIBLE_DEVICES`. It does **not** prevent candidate code from accessing
other devices. Select it only when that weaker device boundary is appropriate.
No failure silently changes the isolation mode.

The selected GPU visibility is applied after environment activation and custom
environment variables. An account-scoped physical-GPU lock serializes AKA
allocations, including all ABBA runs. Locks do not coordinate unrelated programs.

PPU or other compatible CUDA runtimes can supply an explicit, supervisor-owned
`ATREX_SANDBOX_SSH_DEVICE_MAP` JSON object keyed by physical index. Each entry
contains `vendor`, a stable `uuid`, the exact character-device `devices` list,
and the runtime visibility `environment` mapping. Use the host's real driver
nodes and selectors; the backend does not infer proprietary device names.
These settings are captured with the selected profile for children and recovery.

## Operations and evidence

| Operation | SSH behavior |
| --- | --- |
| Evaluate | Native evaluator, full or correctness-only mode; selected shape batches run serially |
| ABBA | One allocation per shape batch; native A/B/B/A blocks, complete per-run and per-shape evidence |
| Check | Import/JIT and one real launch; NVIDIA sanitizer modes require compute-sanitizer |
| Profile | NVIDIA NCU, AMD rocprofv3, or provisioned PPU ACU; unavailable vendor-specific controls fail explicitly |
| Disassemble | NVIDIA SASS/PTX, AMD ISA, or artifacts exposed by the provisioned PPU toolchain |
| Env | Actual runtime architecture, available frameworks, profiler counters and allocation limits |
| Dev | Existing allowlisted command upload, timeout handling and output synchronization |

ABBA repeat counts and acceptance thresholds are unchanged. Completed shape
batches retain their existing checkpoints. Request identity includes the SSH
endpoint, GPU assignment, runtime configuration and isolation mode, so evidence
from a different allocation configuration cannot be reused. Transport loss or a
missing result is not a candidate correctness verdict.

NVIDIA clock control is applied when available. Unsupported clock control is
recorded as not applied; AMD performance levels are not reported as fixed clocks.

Python dependencies must already be installed in the bound environment. Native
CUDA manifests support provisioned pip/system dependencies and clean, pinned
Git repositories bound at `/opt/aka-dependencies/<id>`. The declared Git ref and
optional commit must match that checkout. Dependencies are never downloaded
inside the network-isolated worker. Native CUDA remains a CUDA toolchain
contract; this does not make CUDA source portable to an AMD HIP compiler.

SOL does not provide Light's native ABBA contract; use the Legacy workflow for
SOL verification. Each SSH allocation owns one GPU; distributed evaluator
contracts and MIG selection require a separate allocation implementation.

## Validation

```bash
PYTHONPATH=src python3 tools/check_ssh_backend.py
PYTHONPATH=src python3 tools/check_ssh_backend.py \
  --host gpu-worker --gpu 5 --hardware MI300X \
  --bench /path/to/atrex-bench \
  --runtime-bind /opt/venv --runtime-bind /opt/rocm \
  --init 'export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH; export LD_LIBRARY_PATH=/opt/rocm/lib' \
  --output /tmp/aka-ssh-validation
```

The GPU smoke suite checks source-tree Evaluate/ABBA, diagnostics, device
visibility, synchronized Dev outputs, an incorrect candidate and timeout handling.
It runs only on an explicitly provided host and GPU.
