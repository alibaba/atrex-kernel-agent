#!/usr/bin/env python3
"""Self-contained source-tree diagnostics uploaded to a GPU allocation.

No Runtime installation or ncu_report Python package is required on the worker.
Diagnostics are not correctness or performance Gate evidence.
"""

from __future__ import annotations

import copy
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

DIAGNOSTIC_PREFIX = "__ATREX_RUNTIME_SOURCE_DIAGNOSTIC__="
TEXT_LIMIT = 512 * 1024
# Agate Dev retains only the last 1 MiB of stdout. Budget the complete encoded
# record (including framing), with headroom for any surrounding tool output.
RESPONSE_LIMIT = 768 * 1024
LOG_TEXT_LIMIT = 4 * 1024


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _target(request_path: Path) -> int:
    """One isolated JIT/launch probe; no hidden inputs are printed by this driver."""
    request = json.loads(request_path.read_text())
    stage = request_path.parent
    root = stage / "candidate"
    sys.path[:0] = [str(root), str(root / request["package_root"])]
    os.chdir(root)
    clocks = _load(stage / "__atrex_abba.py", "_atrex_clock_support")
    clocks._check_requirements(request["runtime_requirements"])
    if request.get("requirements"):
        from importlib.metadata import version

        from packaging.requirements import Requirement

        for raw in request["requirements"]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if requirement.url or version(requirement.name) not in requirement.specifier:
                raise RuntimeError("diagnostic dependency is not provisioned in the GPU image")
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("source-tree diagnostics require an accessible GPU")
    hip = getattr(torch.version, "hip", None)
    roctx = None
    if hip and request['operation'] == 'profile':
        roctx = ctypes.CDLL('librocprofiler-sdk-roctx.so')
        roctx.roctxProfilerPause(0)
    arch = request["parameters"].get("arch")
    if arch:
        actual = (torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] if hip
                  else ''.join(map(str, torch.cuda.get_device_capability())))
        requested = re.fullmatch(r'(?:sm_|compute_)?(\d+)([af]?)', arch.replace('.', '')) if not hip else None
        matches = arch == actual if hip else requested is not None and requested.group(1) == actual and not requested.group(2)
        if not matches:
            raise ValueError("arch must match the allocated GPU; cross-compilation is unsupported")
    torch.manual_seed(0)
    model_module = _load(root / request["entrypoint"], "kernel")
    input_module = _load(stage / "input.py", "_atrex_diagnostic_input")
    shape = next(iter(request["shapes"].values()))
    model = model_module.Model(**(shape.get("init_kwargs") or {})).eval()

    def inputs() -> dict[str, Any]:
        value = input_module._make_inputs(**(shape.get("input_kwargs") or {}))
        if not isinstance(value, dict):
            raise TypeError("_make_inputs must return a dictionary")
        return value

    with torch.no_grad():
        if request["operation"] != "check":
            model(**inputs())  # JIT and setup are outside the profiled NVTX range.
            torch.cuda.synchronize()
        values = inputs()  # Fresh buffers, including mutable output/state buffers.
        torch.cuda.synchronize()
        if not hip:
            torch.cuda.nvtx.range_push("ATREX_CANDIDATE")
        elif roctx is not None:
            roctx.roctxProfilerResume(0)
        try:
            model(**values)
            torch.cuda.synchronize()
        finally:
            if not hip:
                torch.cuda.nvtx.range_pop()
            elif roctx is not None:
                roctx.roctxProfilerPause(0)
    return 0


class DiagnosticError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _execute(argv: list[str], root: Path, deadline: float, log: list[dict[str, Any]]) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DiagnosticError("timeout", "diagnostic time budget exhausted")
    with subprocess.Popen(
        argv,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            log.append(_command_log(argv, stdout, stderr, timeout=True))
            raise DiagnosticError("timeout", "diagnostic command timed out") from error
        code = process.returncode
    log.append(_command_log(argv, stdout, stderr, exit_code=code))
    if code:
        raise DiagnosticError("execution", "compiler, GPU tool or candidate execution failed")
    return stdout


def _tool(name: str) -> str:
    result = shutil.which(name)
    if result is None:
        raise DiagnosticError("environment", f"required GPU tool is not installed: {name}")
    return result


def _export(text: str, *, limit: int = TEXT_LIMIT, tail: bool = False) -> dict[str, Any]:
    raw = text.encode("utf-8")
    selected = (raw[-limit:] if tail else raw[:limit]) if limit else b""
    return {
        "text": selected.decode("utf-8", errors="ignore"),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > limit,
    }


def _command_log(argv: list[str], stdout: str, stderr: str, **outcome: Any) -> dict[str, Any]:
    # Successful output is already represented by parsed metrics/exports. Do
    # not duplicate hundreds of KiB of SASS/CSV inside a JSON-escaped log.
    success = outcome.get("exit_code") == 0
    out = _export(stdout, limit=0 if success else LOG_TEXT_LIMIT, tail=True)
    if success:
        out.pop("text")
        out.pop("truncated")
        out["content_omitted"] = True
    return {
        "command": argv,
        **outcome,
        "stdout": out,
        "stderr": _export(stderr, limit=LOG_TEXT_LIMIT, tail=True),
    }


def _text_records(value: Any, path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if isinstance(value.get("text"), str) and "sha256" in value:
            yield path, value
        else:
            for key, item in value.items():
                yield from _text_records(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _text_records(item, f"{path}[{index}]")


def _encode_result(result: dict[str, Any]) -> str:
    """Keep the verdict intact; shrink optional evidence before emitting one record."""
    value = copy.deepcopy(result)
    records = [
        *list(_text_records(value.get("exports"), "exports")),
        *list(_text_records(value.get("logs"), "logs")),
    ]
    truncated = {path for path, item in records if item.get("truncated")}
    budget_applied = False

    def encode() -> str:
        value["output_summary"] = {
            "truncated": bool(truncated),
            "reason": (
                "diagnostic_response_size_limit"
                if budget_applied
                else "diagnostic_text_size_limit"
                if truncated
                else None
            ),
            "response_limit_bytes": RESPONSE_LIMIT,
            "truncated_fields": sorted(truncated),
            "message": (
                "Diagnostic verdict and failure summary are complete. Optional evidence was "
                "truncated to fit Runtime output limits and prevent Agate stdout truncation. "
                "Truncation itself is not a Kernel failure; use passed/status for the verdict. "
                "Text sizes and SHA-256 describe the original content. "
                "Use a narrower kernel filter/launch range to obtain a smaller export."
                if truncated
                else "Diagnostic summary and exports are complete."
            ),
        }
        return DIAGNOSTIC_PREFIX + json.dumps(value, separators=(",", ":")) + "\n"

    encoded = encode()
    while len(encoded.encode("utf-8")) > RESPONSE_LIMIT:
        budget_applied = True
        # Cut previews first, preserving all original size/hash metadata. Size
        # is rechecked after JSON escaping, including non-ASCII/control chars.
        available = [(path, item) for path, item in records if item["text"]]
        if available:
            path, item = max(available, key=lambda pair: len(json.dumps(pair[1]["text"])))
            text = item["text"]
            item["text"] = (
                text[len(text) // 2 + 1 :] if path.startswith("logs") else (text[: len(text) // 2])
            )
            item["truncated"] = True
            truncated.add(path)
        elif any("metrics" in k for k in value.get("kernels", [])):
            for kernel in value["kernels"]:
                kernel.pop("metrics", None)
            truncated.add("kernels.metrics")
        elif value.get("kernels"):
            kernels = value["kernels"]
            value.setdefault("collected_kernel_count", len(kernels))
            value["kernels"] = kernels[: len(kernels) // 2]
            truncated.add("kernels")
        else:
            # Last resort for oversized log metadata: retain the fixed driver
            # verdict/error/clock/scope fields, never slice serialized JSON.
            if not any(key in value for key in ("logs", "exports")):
                # The driver's remaining fixed summary is small. Fail closed if
                # a future extension breaks that invariant, rather than looping.
                raise ValueError("fixed diagnostic summary exceeds response budget")
            for key in ("logs", "exports"):
                if key in value:
                    value.pop(key)
                    truncated.add(key)
            records = []
        encoded = encode()
    return encoded


def _kernels(text: str) -> list[dict[str, Any]]:
    # Shared stdlib parser is shipped alongside this remote driver.
    if __package__:
        from aka.legacy.evaluation.ncu_profile import kernels
    else:
        from ncu_profile import kernels
    return kernels(text)


def _ncu_command(
    ncu: str,
    request: dict[str, Any],
    report: Path,
    target: list[str],
) -> list[str]:
    parameters = request["parameters"]
    level = parameters.get("level", "sol")
    argv = [
        ncu,
        "--force-overwrite",
        "--export",
        str(report),
        "--clock-control",
        "none",
        "--nvtx",
        "--nvtx-include",
        "ATREX_CANDIDATE/",
        "--launch-skip",
        str(parameters.get("launch_skip") or 0),
        "--launch-count",
        str(parameters.get("launch_count") or 10),
        "--kernel-name-base",
        "demangled",
    ]
    if request["operation"] == "disassemble" or level == "survey":
        argv += ["--section", "LaunchStats"]
    elif level == "deep":
        argv += ["--set", "full"]
    else:
        argv += ["--section", "SpeedOfLight"]
    if parameters.get("source"):
        argv += ["--section", "SourceCounters", "--import-source", "yes"]
    metrics = ["gpu__time_duration.sum", *(parameters.get("counters") or [])]
    argv += ["--metrics", ",".join(dict.fromkeys(metrics))]
    name = parameters.get("kernel_name")
    regex = parameters.get("kernel_regex")
    if name:
        argv += ["--kernel-name", f"regex:^{re.escape(name)}$"]
    elif regex:
        argv += ["--kernel-name", f"regex:{regex}"]
    return argv + target


def _collect_ppu(request, root, deadline, log, target):
    parameters = request['parameters']
    if parameters.get('profiler') not in (None, 'acu', 'ncu'):
        raise DiagnosticError('capability', 'PPU profiling requires ACU')
    if parameters.get('source'):
        raise DiagnosticError('capability', 'PPU source correlation is unavailable')
    if request['operation'] == 'disassemble':
        _execute(target, root, deadline, log)
        fmt = parameters.get('fmt', 'auto')
        if fmt not in ('auto', 'ptx', 'sass'):
            raise DiagnosticError('capability', 'PPU CUDA toolchains expose PTX or SASS')
        fmt = 'ptx' if fmt == 'auto' else fmt
        if fmt == 'ptx':
            text = '\n'.join(path.read_text() for path in sorted((root / '.caches').rglob('*.ptx')))
        else:
            text = '\n'.join(_execute([_tool('cuobjdump'), '--dump-sass', str(path)], root, deadline, log)
                             for path in sorted((root / '.caches').rglob('*.cubin')))
        if not text:
            raise DiagnosticError('collection', f'The PPU compiler produced no {fmt} artifacts')
        return {'format': fmt, 'exports': {fmt + '.txt': _export(text)}}
    report = root / 'metrics.csv'
    command = [sys.executable, str(root / 'profile_ppu.py'), '--csv', '--log-file', str(report),
               '--nvtx', '--nvtx-include', 'ATREX_CANDIDATE/',
               '--launch-skip', str(parameters.get('launch_skip') or 0),
               '--launch-count', str(parameters.get('launch_count') or 10),
               '--metrics', ','.join(['gpu__time_duration.sum', *(parameters.get('counters') or [])])]
    level = parameters.get('level', 'sol')
    command += ['--set', 'full'] if level == 'deep' else [
        '--section', 'LaunchStats' if level == 'survey' else 'SpeedOfLight']
    if parameters.get('kernel_name'):
        command += ['--kernel-name', parameters['kernel_name']]
    elif parameters.get('kernel_regex'):
        command += ['--kernel-name', 'regex:' + parameters['kernel_regex']]
    _execute([*command, *target], root, deadline, log)
    kernels = _kernels(report.read_text())
    if not kernels:
        raise DiagnosticError('collection', 'PPU profiler produced no kernel measurements')
    result = {'profiler': 'acu', 'level': level, 'kernels': kernels,
              'exports': {'metrics.csv': _export(report.read_text())}}
    if parameters.get('top_kernels'):
        result['collected_kernel_count'] = len(kernels)
        result['kernels'] = sorted(kernels, key=lambda item: item.get('duration_us', 0),
                                   reverse=True)[:parameters['top_kernels']]
    return result


def _collect_amd(request, root, deadline, log, target):
    parameters = request['parameters']
    if parameters.get('profiler') not in (None, 'rocprofv3'):
        raise DiagnosticError('capability', 'AMD profiling requires rocprofv3')
    if parameters.get('fmt', 'auto') not in ('auto', 'isa'):
        raise DiagnosticError('capability', 'AMD disassembly supports auto or isa')
    if request['operation'] == 'disassemble':
        _execute(target, root, deadline, log)
        binaries = sorted((root / '.caches').rglob('*.hsaco'))
        if not binaries:
            raise DiagnosticError('collection', 'No candidate AMD code objects were produced')
        objdump = shutil.which('llvm-objdump') or '/opt/rocm/llvm/bin/llvm-objdump'
        assembly = '\n'.join(f'; {path.relative_to(root)}\n' + _execute(
            [objdump, '--disassemble', '--demangle', str(path)], root, deadline, log)
            for path in binaries)
        return {'format': 'isa', 'exports': {'isa.txt': _export(assembly)}}
    directory = root / 'rocprof'
    command = [_tool('rocprofv3'), '--kernel-trace', '--marker-trace', '--output-format', 'csv',
               '--output-directory', str(directory)]
    counters = parameters.get('counters') or []
    if counters:
        command += ['--pmc', *counters]
    if parameters.get('source'):
        raise DiagnosticError('capability', 'rocprofv3 does not provide NCU source correlation')
    _execute([*command, '--', *target], root, deadline, log)
    kernels, exports = [], {}
    pattern = parameters.get('kernel_regex')
    if parameters.get('kernel_name'):
        pattern = '^' + re.escape(parameters['kernel_name']) + '$'
    for path in sorted(directory.rglob('*.csv')):
        exports[str(path.relative_to(directory))] = _export(path.read_text())
        with path.open() as stream:
            for row in csv.DictReader(stream):
                name = row.get('Kernel_Name') or row.get('KernelName')
                if not name or pattern and not re.search(pattern, name):
                    continue
                start, end = row.get('Start_Timestamp'), row.get('End_Timestamp')
                if start is not None and end is not None:
                    kernels.append({'name': name, 'duration_us': (float(end) - float(start)) / 1000,
                                    'metrics': {key: row[key] for key in counters if key in row}})
    skip = parameters.get('launch_skip') or 0
    count = parameters.get('launch_count')
    kernels = kernels[skip:skip + count if count else None]
    if not kernels:
        raise DiagnosticError('collection', 'rocprofv3 produced no matching kernel measurements')
    top = parameters.get('top_kernels')
    if top:
        kernels = sorted(kernels, key=lambda item: item['duration_us'], reverse=True)[:top]
    return {'profiler': 'rocprofv3', 'level': parameters.get('level', 'sol'),
            'kernels': kernels, 'exports': exports}


def _collect(
    request: dict[str, Any],
    root: Path,
    deadline: float,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    operation, parameters = request["operation"], request["parameters"]
    target = [
        sys.executable,
        str(root / "__atrex_diagnostic.py"),
        "--target",
        str(root / "request.json"),
    ]
    if operation == "check":
        sanitizer = parameters.get("sanitize")
        if sanitizer:
            import torch
            if getattr(torch.version, 'hip', None):
                raise DiagnosticError('capability', 'compute-sanitizer is unavailable on AMD')
            target = [
                _tool("compute-sanitizer"),
                "--tool",
                sanitizer,
                "--error-exitcode",
                "86",
                "--target-processes",
                "all",
                *target,
            ]
        _execute(target, root, deadline, log)
        return {
            "compile_ok": True,
            "launch_ok": True,
            "sanitize": sanitizer,
            "sanitizer_passed": True if sanitizer else None,
            "scope": "one_shape_launch_probe",
            "correctness_checked": False,
        }

    import torch
    if getattr(torch.version, 'hip', None):
        return _collect_amd(request, root, deadline, log, target)
    if request.get('vendor') == 'ppu':
        return _collect_ppu(request, root, deadline, log, target)
    ncu = _tool("ncu")
    report = root / "diagnostic.ncu-rep"
    _execute(_ncu_command(ncu, request, report, target), root, deadline, log)
    if not report.is_file():
        raise DiagnosticError(
            "collection", "NCU produced no report; check kernel filter/launch range"
        )
    # The raw CSV page already exports canonical metric names. NCU restricts
    # --print-metric-name to the details page and rejects it for raw exports.
    raw = _execute(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        root,
        deadline,
        log,
    )
    kernels = _kernels(raw)
    if not kernels:
        raise DiagnosticError(
            "collection", "no kernels collected; check kernel filter/launch range"
        )
    result: dict[str, Any] = {
        "profiler": "ncu",
        "kernels": kernels,
        "exports": {"metrics.csv": _export(raw)},
    }
    if operation == "profile":
        result["level"] = parameters.get("level", "sol")
        top = parameters.get("top_kernels")
        if top:
            result["collected_kernel_count"] = len(kernels)
            result["kernels"] = sorted(
                kernels, key=lambda k: k.get("duration_us", 0), reverse=True
            )[:top]
    if operation == "disassemble" or parameters.get("source"):
        if __package__:
            from aka.legacy.evaluation.ncu_profile import has_sass
        else:
            from ncu_profile import has_sass
        fmt = parameters.get("fmt", "auto")
        fmt = "sass" if fmt == "auto" else fmt
        view = "cuda,sass" if operation == "profile" else fmt
        assembly = _execute(
            [ncu, "--import", str(report), "--page", "source", "--print-source", view],
            root,
            deadline,
            log,
        )
        if view == "cuda,sass" and not has_sass(assembly.encode()):
            view = "sass"
            assembly = _execute(
                [ncu, "--import", str(report), "--page", "source", "--print-source", view],
                root, deadline, log,
            )
            result["warnings"] = ["CUDA/source correlation unavailable; exported SASS-only evidence from the same report"]
        # Tool banners alone are not evidence that assembly was obtained.
        has_assembly = (bool(re.search(r"(?:\.version|\.target|\.entry)\b", assembly))
                        if fmt == "ptx" else has_sass(assembly.encode()))
        if not has_assembly:
            raise DiagnosticError(
                "assembly",
                "assembly unavailable in NCU report; try sass or check toolchain support",
            )
        result["exports"][f"{fmt}.txt"] = _export(assembly)
        result["format"] = fmt
        if operation == "profile":
            result["source_view"] = view
    return result


def _driver(request_path: Path) -> int:
    root = request_path.resolve().parent
    request = json.loads(request_path.read_text())
    parameters = request["parameters"]
    deadline = time.monotonic() + float(request["timeout_s"])
    log: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "operation": request["operation"],
        "shape_id": next(iter(request["shapes"])),
        "passed": False,
        "status": "error",
    }
    try:
        # Always override caller caches; each job compiles the exact uploaded tree.
        for name, folder in (
            ("CUTE_DSL_CACHE_DIR", "cute"),
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCH_EXTENSIONS_DIR", "torch"),
            ("TMPDIR", "tmp"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("CUTE_DSL_DUMP_DIR", "dump"),
        ):
            path = root / ".caches" / folder
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        if parameters.get("fmt") == "ptx":
            os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
            os.environ["CUTE_DSL_KEEP_PTX"] = "1"
        clocks = _load(root / "__atrex_abba.py", "_atrex_clock_support")
        import torch
        clock_supported = not getattr(torch.version, 'hip', None) and request.get('vendor') != 'ppu'
        with clocks._clock_lock(bool(request["lock_clocks"]) and clock_supported) as report:
            result.update(_collect(request, root, deadline, log))
            result["clock_lock"] = dict(report, requested=bool(request['lock_clocks']))
        result.update(passed=True, status="passed")
    except DiagnosticError as error:
        result.update(error=str(error), failure_stage=error.stage)
    except Exception as error:
        log.append({"error": _export(f"{type(error).__name__}: {error}", limit=LOG_TEXT_LIMIT)})
        result.update(
            error="diagnostic environment or clock setup failed", failure_stage="environment"
        )
    if request["operation"] == "check" and not result["passed"]:
        result.update(
            compile_ok=None,
            launch_ok=False,
            sanitize=parameters.get("sanitize"),
            sanitizer_passed=False if parameters.get("sanitize") else None,
        )
    # Kept in trusted raw evidence; normal private-result projection removes logs.
    result["logs"] = log
    sys.stdout.write(_encode_result(result))
    sys.stdout.flush()
    # A failed diagnostic is a recorded tool result, not a lost Dev transport job.
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--target":
        return _target(Path(sys.argv[2]).resolve())
    return _driver(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
