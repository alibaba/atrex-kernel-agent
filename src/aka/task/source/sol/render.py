"""Original SOL reference wrapper and solution rendering."""
import re

def _build_kernel(defn: dict) -> str:
    """V0 kernel: inline the reference (rename run->_ref) + a DPS wrapper.

    Staging copies only solution.sources, so kernel.py must be self-contained
    (it cannot `import reference`). Args are definition.inputs then
    definition.outputs (DPS order); the wrapper writes outputs in place.
    """
    inputs = list(defn["inputs"].keys())
    outputs = list(defn["outputs"].keys())
    params = ", ".join(inputs + outputs)
    call = ", ".join(inputs)
    if len(outputs) == 1:
        assign = f"    {outputs[0]}[:] = _out\n"
    else:
        assign = "    if not isinstance(_out, (tuple, list)):\n        _out = (_out,)\n"
        for i, o in enumerate(outputs):
            assign += f"    {o}[:] = _out[{i}]\n"
    header = (
        "# V0 baseline: self-contained destination-passing-style (DPS) wrapper around the\n"
        "# verbatim SOL reference logic. Correct + directly submittable. The optimization\n"
        "# loop rewrites the body of run() toward the target framework; keep it DPS and keep\n"
        "# the argument names/order = definition.inputs then definition.outputs.\n"
        "{REF}\n\n"
        f"def run({params}):\n"
        f"    _out = _ref({call})\n"
        f"{assign}"
    )
    return header


def _render_kernel(defn: dict, reference_src: str) -> str:
    ref = re.sub(r"(?m)^def run\(", "def _ref(", reference_src)
    return _build_kernel(defn).replace("{REF}", ref)


def _solution_json(defn: dict, name: str, framework: str, platform: str) -> dict:
    hw = []
    # SOL-ExecBench's schema only accepts B200 or LOCAL here.  The optimizer's
    # logical platform may be an inventory alias (for example pro5000); actual
    # remote scheduling is selected independently by --sandbox-hardware.
    if platform.upper() == "B200":
        hw.append("B200")
    hw.append("LOCAL")
    # V0 is always a pure-PyTorch wrapper (guaranteed correct + submittable).
    # The loop migrates the body to `framework` and updates languages/dependencies then.
    return {
        "name": f"{name}_v0_pytorch",
        "definition": defn["name"],
        "author": "atrex-aka",
        "description": f"V0 baseline (DPS PyTorch wrapper around the reference); target framework: {framework}",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": hw,
            "entry_point": "kernel.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": True,
        },
        # No inline content: the evaluator reads the live kernel.py from disk, so
        # kernel.py stays the single source of truth across iterations.
        "sources": [{"path": "kernel.py"}],
    }


def _readme(name: str, defn: dict, framework: str, platform: str, gpu_wiki: str, n_workloads: int) -> str:
    return (
        f"# kernel_opt_{name}\n\n"
        f"Profile-driven optimization of SOL-ExecBench op **{defn['name']}**.\n\n"
        f"{defn.get('description', '').strip()}\n\n"
        "## Goal\n\n"
        "**Minimize the GEOMEAN of per-workload kernel latency** "
        "(`performance.latency_us` in `memory/v<N>.json`), while keeping ALL workloads "
        "correct under their own SOL tolerances. A version that passes `test_kernel.py` "
        "is directly submittable to SOL-ExecBench.\n\n"
        "## Config\n\n"
        f"- Target platform: `{platform}`\n"
        f"- Target framework: `{framework}` (V0 starts as PyTorch; migrate the body of `run()` in `kernel.py`)\n"
        f"- gpu_wiki_path: `{gpu_wiki}`\n"
        f"- Workloads (shape set): {n_workloads} in `workload.jsonl` (ground truth — do NOT edit)\n\n"
        "## Ground truth (immutable)\n\n"
        "- `definition.json`, `reference.py`, `workload.jsonl` — copied verbatim from the op dir.\n"
        "- `test_kernel.py` — the SOL evaluator harness (immutable methodology).\n\n"
        "## Workflow\n\n"
        "- Edit `kernel.py` only (DPS `run()`; args = definition.inputs then definition.outputs).\n"
        "- When migrating framework, also update `solution.json` `spec.languages` / `dependencies`.\n"
        "- Validate + bench every iteration with `python test_kernel.py --version v<N>`.\n"
    )

