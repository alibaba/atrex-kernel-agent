"""Standalone single-GPU integration tests for the public localhost sandbox."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


MODEL_KEYS = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
SUITE = Path(__file__).resolve().parent
SOURCE_ROOT = SUITE.parents[1]

GPU_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH",
            "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCM_PATH")


def clean_environment(root: Path, model: bool = True) -> dict[str, str]:
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/opt/rocm/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home), "TMPDIR": str(root / "tmp"),
        "PYTHONNOUSERSITE": "1", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
        "DISABLE_AUTOUPDATER": "1", "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "TRITON_CACHE_DIR": str(root / "triton-cache"),
        "ATREX_SANDBOX_GPU": "local", "ATREX_SANDBOX_URL": "http://127.0.0.1:8000",
        "ATREX_SANDBOX_QUEUE_WAIT_GRACE": "600",
        "LANG": "C.UTF-8",
    }
    (root / "tmp").mkdir(exist_ok=True)
    for name in GPU_KEYS + (MODEL_KEYS if model else ()):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def redact(value: str) -> str:
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        secret = os.environ.get(name, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: int) -> str:
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        log.write_text(redact(output), encoding="utf-8")
        raise
    output = redact(output)
    log.write_text(output, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{log.stem}: exit {process.returncode}\n{output[-8000:]}")
    return output


def positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def archived_candidate(workspace: Path, attempt: dict) -> dict[str, bytes]:
    episode = workspace / ".atrex_long_horizon/episodes/e0001"
    archived = episode / "archive"
    git_info = json.loads((archived / "git.json").read_text())
    if (git_info.get("head") != attempt.get("episode_head")
            or git_info.get("base_commit") != attempt.get("base_commit")):
        raise RuntimeError("V2 archive differs from the recorded episode")
    files = {}
    for name in ("kernel.py", "solution.json"):
        path = archived / "worktree_files" / name
        content = path.read_bytes() if path.is_file() and not path.is_symlink() else subprocess.check_output(
            ["git", "show", f"{git_info['head']}:{name}"], cwd=workspace)
        files[name] = content
    return files


def episode_evaluation(workspace: Path, attempt: dict, independent_receipt: dict | None = None) -> dict:
    fingerprints = {name: hashlib.sha256(content).hexdigest()
                    for name, content in archived_candidate(workspace, attempt).items()}
    if independent_receipt is not None:
        receipts = [independent_receipt]
    else:
        path = workspace / ".atrex_long_horizon/episodes/e0001/episode_runtime/evaluations.jsonl"
        receipts = [json.loads(line) for line in path.read_text().splitlines()]
    for receipt in reversed(receipts):
        result = receipt.get("result") or {}
        latencies = result.get("latency_us_by_shape") or {}
        if (type(receipt.get("schema_version")) is int and receipt["schema_version"] >= 3
                and receipt.get("kernel_sha256") == fingerprints["kernel.py"]
                and receipt.get("solution_sha256") == fingerprints["solution.json"]
                and result.get("all_pass") is True
                and result.get("performance_objective") == "shape_speedup_arithmetic_mean"
                and set(latencies) == {"0", "1", "2"}
                and all(positive(value) for value in latencies.values())
                and positive(result.get("latency_us_geomean"))
                and positive(result.get("performance_score"))):
            return result
    raise RuntimeError("V2 has no complete passing candidate evaluation for its final kernel/solution hashes")


def campaign_attempt(workspace: Path) -> dict:
    for number in range(3):
        path = f"memory/v{number}.json"
        memory = json.loads((workspace / path).read_text())
        committed = json.loads(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=workspace, text=True))
        if memory != committed or memory.get("version") != f"v{number}":
            raise RuntimeError(f"{path}: canonical record is missing or differs from Git HEAD")
        if number < 2 and (memory.get("correctness", {}).get("status") != "PASS"
                           or not positive(memory.get("performance", {}).get("latency_us"))):
            raise RuntimeError(f"{path}: baseline has no passing correctness/latency result")
    state = json.loads((workspace / ".atrex_long_horizon/state.json").read_text())
    attempts = state.get("attempts", [])
    if len(attempts) != 1:
        raise RuntimeError("expected exactly one completed fast episode")
    attempt = attempts[0]
    if (attempt.get("mode") != "fast" or attempt.get("status") not in {"candidate_ready", "pivot"}
            or attempt.get("violation")):
        raise RuntimeError("V2 has no complete passing candidate evaluation: " + str(attempt.get("violation") or attempt.get("status")))
    return attempt


def validate_campaign(workspace: Path, independent_receipt: dict | None = None) -> dict:
    attempt = campaign_attempt(workspace)
    # Verify the final archived bytes, including when a pivot restored a prior
    # candidate. Canonical V2 incumbent metrics never certify those bytes.
    evaluation = episode_evaluation(workspace, attempt, independent_receipt)
    if not (workspace / "kernel.py").is_file():
        raise RuntimeError("final kernel.py is missing")
    # A correctly evaluated candidate may lose the performance comparison. This
    # gate checks availability, so promotion is deliberately not required.
    return {"versions": ["v0", "v1", "v2"], "candidate_correctness": "PASS",
            "candidate_promoted": attempt.get("accepted", False),
            "candidate_latency_us": evaluation["latency_us_geomean"]}


def verify_archived_candidate(workspace: Path, artifacts: Path, env: dict, sandbox: Path) -> dict:
    attempt = campaign_attempt(workspace)
    files = archived_candidate(workspace, attempt)
    originals = {}
    for name in files:
        path = workspace / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"final workspace {name} must be a regular file")
        originals[name] = path.read_bytes()
    saved = artifacts / "v2-final-candidate"
    saved.mkdir()
    for name, content in files.items():
        (saved / name).write_bytes(content)
    # The optimization process has exited. Reuse its trusted harness, swapping
    # only the two candidate files and restoring the canonical workspace even
    # if the independent evaluation fails. Do not edit the episode archive.
    try:
        for name, content in files.items():
            (workspace / name).write_bytes(content)
        output = run([sys.executable, str(sandbox), "--hardware", "local", "--url", "http://127.0.0.1:8000",
                      "--workspace", str(workspace), "--kind", "run", "--timeout", "600", "--no-sync", "--",
                      sys.executable, "test_kernel.py", "--no-memory"],
                     workspace, env, artifacts / "v2-final-evaluation.log", 900)
        if any((workspace / name).is_symlink() or (workspace / name).read_bytes() != content
               for name, content in files.items()):
            raise RuntimeError("V2 candidate changed during independent evaluation")
        results = [json.loads(line.split("RESULT_JSON=", 1)[1]) for line in output.splitlines()
                   if "[test_kernel] RESULT_JSON=" in line]
        if not results:
            raise RuntimeError("V2 independent evaluation returned no result")
        receipt = {"schema_version": 3, "gateway_kind": "run", "verification": "ci-independent-final-candidate",
                   "kernel_sha256": hashlib.sha256(files["kernel.py"]).hexdigest(),
                   "solution_sha256": hashlib.sha256(files["solution.json"]).hexdigest(), "result": results[-1]}
        (artifacts / "v2-final-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        episode_evaluation(workspace, attempt, receipt)
        return receipt
    finally:
        for name, content in originals.items():
            path = workspace / name
            if path.is_symlink():
                path.unlink()
            path.write_bytes(content)


def collect_campaign(workspace: Path, artifacts: Path) -> None:
    for name in ("kernel.py", "solution.json", "agent_problem.json", "README.md"):
        if (workspace / name).is_file():
            shutil.copy2(workspace / name, artifacts)
    for source, target in (("memory", "memory"), (".atrex_long_horizon/episodes", "attempts")):
        if (workspace / source).is_dir():
            shutil.copytree(workspace / source, artifacts / target, dirs_exist_ok=True)
    state = workspace / ".atrex_long_horizon/state.json"
    if state.is_file():
        shutil.copy2(state, artifacts / "episode-state.json")


def prepare_fixture(root: Path, bench: Path) -> Path:
    # The native optimizer discovers its evaluator from the operator's ancestor
    # directories. Preserve that public layout in a disposable view rather than
    # adding the fixture or measured metadata to the caller's bench checkout.
    evaluator = bench / "scripts/run_eval.py"
    package = bench / "src/atrex_bench"
    if not evaluator.is_file() or not package.is_dir():
        raise RuntimeError("bench checkout must contain scripts/run_eval.py and src/atrex_bench")
    view = root / "bench"
    (view / "scripts").mkdir(parents=True)
    shutil.copy2(evaluator, view / "scripts/run_eval.py")
    shutil.copytree(package, view / "src/atrex_bench",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    fixture = view / "operators/vector_add"
    shutil.copytree(SUITE / "fixture", fixture)
    return fixture


def execute(root: Path, bench: Path, platform: str, artifacts: Path, timeout: int, fault: str = "") -> dict:
    candidate = SOURCE_ROOT
    artifacts.mkdir(exist_ok=True)
    env = clean_environment(root)
    gateway_env = clean_environment(root, model=False)
    # The pinned evaluator uses a src-layout package and a thin CLI wrapper.
    # Expose only that declared public checkout, never inherited PYTHONPATH.
    gateway_env["PYTHONPATH"] = str(bench / "src")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, env=env, text=True).strip()
    fixture_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((SUITE / "fixture").iterdir()) if path.is_file()}
    if not (candidate / "3rdparty/ncu-report-skill/SKILL.md").is_file():
        raise RuntimeError("required public ncu-report-skill submodule is missing")
    if not shutil.which("claude", path=env["PATH"]):
        raise RuntimeError("public Claude CLI is missing")
    if not env.get("ANTHROPIC_AUTH_TOKEN") or not env.get("ANTHROPIC_MODEL"):
        raise RuntimeError("model configuration is missing")
    if os.getuid() == 0:
        raise RuntimeError("public unattended Claude CLI must run as the image's non-root user")
    # GPU imports occur only after entering the clean process environment.
    probe = run([sys.executable, "-c", "import json,torch,triton; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.ones(1,device='cuda'); torch.cuda.synchronize(); print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'hip':torch.version.hip,'cuda':torch.version.cuda}))"],
                candidate, env, artifacts / "preflight.log", 120)
    fixture = prepare_fixture(root, bench)
    run([sys.executable, "-c", "import atrex_bench.cli.run_eval; print('pinned public evaluator import passed')"],
        candidate, gateway_env, artifacts / "evaluator-preflight.log", 120)
    gateway_log = (artifacts / "gateway.log").open("w")
    gateway = subprocess.Popen([sys.executable, str(candidate / "tools/local_gateway.py"), "serve", "--host", "127.0.0.1", "--port", "8000", "--state-dir", str(root / "gateway-state"), "--workers", "1", "--atrex-bench-root", str(bench)],
                               cwd=candidate, env=gateway_env, start_new_session=True, stdout=gateway_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while True:
            if gateway.poll() is not None:
                raise RuntimeError("localhost gateway exited: " + (artifacts / "gateway.log").read_text()[-4000:])
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("localhost gateway did not become healthy within 120s")
                time.sleep(1)
        smoke = root / "smoke"
        smoke.mkdir()
        shutil.copy2(SUITE / "gpu_smoke.py", smoke)
        if fault == "missing-module":
            with (smoke / "gpu_smoke.py").open("a") as stream:
                stream.write("\nimport atrex_integration_missing_module\n")
        # A bare worker Python must retain the gateway's dependency environment;
        # an absolute interpreter would conceal a broken virtualenv PATH.
        run([sys.executable, str(candidate / "tools/sandbox.py"), "--hardware", "local", "--url", "http://127.0.0.1:8000", "--workspace", str(smoke), "--kind", "run", "--timeout", "600", "--sync", "smoke-result.json", "--", "python", "gpu_smoke.py"],
            smoke, gateway_env, artifacts / "smoke.log", 900)
        smoke_result = json.loads((smoke / "smoke-result.json").read_text())
        if smoke_result.get("status") != "PASS" or smoke_result.get("cases") != [512, 1025, 4096]:
            raise RuntimeError("sandbox did not return the expected GPU result artifact")
        shutil.copy2(smoke / "smoke-result.json", artifacts)
        latencies = smoke_result.get("reference_latency_us_by_shape") or {}
        if set(latencies) != {"0", "1", "2"} or not all(positive(value) for value in latencies.values()):
            raise RuntimeError("missing measured PyTorch reference latencies")
        metadata = json.loads((fixture / "metadata.json").read_text())
        for shape_id, latency in latencies.items():
            metadata["shapes"][shape_id]["production_performance"] = {"performance_us": latency}
        # The native public scoring contract requires positive reference timing.
        # Measure it on this GPU through the gateway; never invent benchmark data.
        (fixture / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(fixture / "metadata.json", artifacts / "measured-fixture-metadata.json")
        run([sys.executable, "orchestrator/optimize.py", "--op-dir", str(fixture), "--workspace", str(root / "runs"),
             "--platform", platform, "--framework", "Triton", "--sandbox-hardware", "local", "--sandbox-url", "http://127.0.0.1:8000",
             "--agent-cli", "claude", "--optimization-mode", "production", "--framework-baseline", "always",
             "--max-iters", "2", "--fast-episodes", "1", "--fast-trials", "1", "--sandbox-timeout", "600",
             "--setup-timeout", str(timeout), "--framework-baseline-timeout", str(timeout)], candidate, env, artifacts / "campaign.log", timeout)
        workspaces = list((root / "runs").glob("kernel_opt_*"))
        if len(workspaces) != 1:
            raise RuntimeError("expected exactly one campaign workspace")
        workspace = workspaces[0]
        collect_campaign(workspace, artifacts)
        final_env = dict(gateway_env, ATREX_PRIVATE_REFERENCE_DIR=str(fixture))
        receipt = verify_archived_candidate(workspace, artifacts, final_env, candidate / "tools/sandbox.py")
        campaign = validate_campaign(workspace, receipt)
        output = run([sys.executable, str(candidate / "tools/sandbox.py"), "--hardware", "local", "--url", "http://127.0.0.1:8000", "--workspace", str(workspace), "--kind", "run", "--timeout", "600", "--no-sync", "--", sys.executable, "test_kernel.py", "--no-memory", "--correctness-only"],
                     workspace, final_env, artifacts / "final-evaluation.log", 900)
        results = [json.loads(line.split("RESULT_JSON=", 1)[1]) for line in output.splitlines() if "[test_kernel] RESULT_JSON=" in line]
        if not results or results[-1].get("all_pass") is not True:
            raise RuntimeError("final kernel failed independent sandbox reevaluation")
        return {"status": "PASS", "source_commit": source_commit, "fixture_sha256": fixture_hashes,
                "environment": json.loads(probe.splitlines()[-1]), "campaign": campaign}
    finally:
        workspaces = list((root / "runs").glob("kernel_opt_*"))
        if len(workspaces) == 1:
            try:
                collect_campaign(workspaces[0], artifacts)
            except Exception as exc:
                (artifacts / "campaign-evidence-error.log").write_text(redact(f"{type(exc).__name__}: {exc}"))
        # This fixture is public. Keep the gateway's structured evaluator errors
        # for CI diagnostics, even when the agent-facing client masks cases.
        if gateway.poll() is None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/v1/jobs?limit=200", timeout=5) as response:
                    evidence = json.load(response)
                (artifacts / "gateway-jobs.json").write_text(redact(json.dumps(evidence, indent=2)) + "\n")
            except Exception as exc:
                (artifacts / "gateway-evidence-error.log").write_text(redact(f"{type(exc).__name__}: {exc}"))
        if gateway.poll() is None:
            os.killpg(gateway.pid, signal.SIGTERM)
            try:
                gateway.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(gateway.pid, signal.SIGKILL)
                gateway.wait()
        gateway_log.close()


def main() -> int:
    def interrupted(signum, frame):
        raise InterruptedError(f"integration test interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="new, disposable test directory")
    parser.add_argument("--bench-root", type=Path, required=True, help="prepared public atrex-bench checkout")
    parser.add_argument("--platform", required=True, help="optimizer platform, for example H20 or MI308X")
    parser.add_argument("--output-dir", type=Path, help="evidence directory (default: WORK_DIR/artifacts)")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--fault", choices=["", "missing-module"], default="", help="negative test; must exit nonzero")
    args = parser.parse_args()
    root = args.work_dir.resolve()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if root.exists() and any(root.iterdir()):
        parser.error("--work-dir must be empty; existing test evidence is never overwritten")
    root.mkdir(parents=True, exist_ok=True)
    artifacts = args.output_dir.resolve() if args.output_dir else root / "artifacts"
    if artifacts.exists() and any(artifacts.iterdir()):
        parser.error("--output-dir must be empty")
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        result = execute(root, args.bench_root.resolve(), args.platform, artifacts, args.timeout, args.fault)
        code = 0
    except Exception as exc:
        result = {"status": "FAIL", "error": redact(str(exc))}
        code = 1
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    for path in artifacts.rglob("*"):
        if path.is_file():
            try:
                path.write_text(redact(path.read_text()), encoding="utf-8")
            except UnicodeError:
                pass
    (artifacts / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
