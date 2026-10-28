"""Trusted, workspace-independent SSH runtime probes (never import candidates)."""

from __future__ import annotations

import shlex

GPU_HEALTH_SOURCE = (
    "import torch; "
    "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
    "x = torch.ones(4, device='cuda'); "
    "y = x * x + 1; torch.cuda.synchronize(); "
    "assert y.cpu().tolist() == [2.0] * 4, 'GPU compute check failed'; "
    "p = torch.cuda.get_device_properties(0); "
    "print(getattr(p, 'gcnArchName', '') or torch.cuda.get_device_capability(0))"
)
DEFAULT_SSH_HEALTH_COMMAND = shlex.join(["python", "-c", GPU_HEALTH_SOURCE])


def runtime_health_command(*, sol: bool, framework: str) -> str:
    """Check selected installed tooling without executing operator-owned code.

    Native Atrex-Bench sources are bundled per job, not installed on the host;
    their operator-specific contract is still checked by the real evaluator.
    """
    imports = {
        "triton": "triton",
        "gluon": "triton.experimental.gluon",
        "cutedsl": "cutlass.cute",
        "tilelang": "tilelang",
        "flydsl": "flydsl",
    }
    lines = ["import importlib, pathlib, shlex, shutil, subprocess"]
    module = imports.get(framework.casefold())
    if module:
        lines.append(f"importlib.import_module({module!r})")
    if framework.casefold() == "cuda":
        lines.append("assert shutil.which('nvcc'), 'CUDA framework requires nvcc on PATH'")
    if sol:
        # Use the same interpreter as the evaluator console script, not merely
        # the shell's `python`; two environments on PATH can otherwise disagree.
        probe = (
            GPU_HEALTH_SOURCE
            + "; from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype; "
            + "assert dtype_str_to_torch_dtype('float32') == torch.float32"
        )
        lines += [
            "cli = shutil.which('sol-execbench')",
            "if cli:",
            "    with pathlib.Path(cli).open() as stream: header = stream.readline().strip()",
            "    assert header.startswith('#!') and 'python' in header, "
            "'SOL preflight requires a Python sol-execbench console script'",
            "    interpreter = shlex.split(header[2:])",
            "else:",
            "    assert shutil.which('uv'), 'sol-execbench CLI and uv are unavailable'",
            "    interpreter = ['uv', 'run', 'python']",
            f"subprocess.run([*interpreter, '-c', {probe!r}], check=True)",
        ]
    return shlex.join(["python", "-c", "\n".join(lines)])


def combined_health_command(health: str, runtime: str) -> str:
    return f"({health}) && ({runtime})" if runtime else health
