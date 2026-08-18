"""Single-operator optimization campaign: baseline setup, episode loop, promotion policy."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import agent_runtime as _agent_runtime
from .constants import (
    AGENT_PROBLEM_GENERATION_PROMPT,
    ATREX_PRIVATE_REFERENCE_ENV,
    ATREX_BENCH_HARNESS,
    DEFAULT_CONVERT_AFTER,
    DEFAULT_HANDOFF_RESUMES,
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_VERIFY_REPEATS,
    DEFAULT_VERIFY_RUN_TIMEOUT,
    DEPENDENCY_REVIEW_PROMPT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    DEPENDENCY_REVIEW_TIMEOUT_S,
    FRAMEWORK_BASELINE_CATEGORY,
    FRAMEWORK_BASELINE_FILE,
    FRAMEWORK_BASELINE_TIMEOUT_S,
    FRAMEWORK_BASELINE_VERSION,
    IMMUTABLE_BASELINE_PATHS,
    PROFILE_DRIVER,
    PROMPTS_DIR,
    REPO_ROOT,
    SOL_SEED,
    WORKSPACE_INIT,
)
from .hardware import hardware_directive, kernel_is_gluon
from .optimization_policy import (
    DependencyReviewSignal,
    install_workspace_policy,
    optimization_mode_directive,
    production_kernel_violations,
)
from .plan_reviewers import discover_plan_reviewers, plan_reviewer_environment
from .session_io import (
    SessionResult,
    _dependency_review_candidate_paths,
    _dependency_review_digest,
    _record_local_test_result,
    _render,
    _sandbox_command,
    _test_result_from_stdout,
    _validate_dependency_review,
    run_session,
    sandbox_directive,
)
from .operator_layout import (
    AGENT_PROBLEM_FILENAME,
    agent_visible_operator_files,
    has_agent_problem,
    is_sol_op,
    validate_agent_problem,
    validate_generated_agent_problem,
    validate_private_shapes,
)
from .workspace_runtime import (
    _agent_runtime_directive,
    _baseline_driver_directive,
    link_runtime,
)
from .workspace_state import (
    git_head,
    git_kernel_blob,
    git_path_blob,
    git_worktree_blob,
    head_kernel_is_initial_baseline,
    latest_version,
    read_memory,
    resolve_framework_baseline_commit,
    v0_baseline_commit,
    write_stall,
)


_LONG_REVIEWER_SESSION_ENV = {
    "codex": "ATREX_CODEX_REVIEW_SESSION_FILE",
}


@dataclass
class Campaign:
    name: str
    kernel_demo: str
    platform: str
    framework: str
    notes: str = "none"
    arch: str = ""  # real runtime GPU arch e.g. "sm_103" / "gfx942"; auto-detected
    work_dir: str = ""  # explicit working directory; "" = Path.cwd() (backward compat)
    workspace_suffix: str = ""  # internal auto-dispatch suffix, e.g. triton_h20
    max_iters: int = 20
    token_budget: int = 0  # 0 = no token cap (max-iters still bounds the run)
    target_util: float = 90.0
    setup_timeout: int = 7200  # 120 min for the baseline session
    max_stall: int = 0  # 0 = disabled; >0 = stop after N unpromoted episodes
    convert_after: int = (
        DEFAULT_CONVERT_AFTER  # triton-only: mandatory Gluon conversion threshold
    )
    sandbox_hardware: str = (
        ""  # agate scheduler token, e.g. REMOTE_GPU (may differ from platform)
    )
    sandbox_profile: str = ""  # pre/prod; empty preserves normal agate URL resolution
    sandbox_url: str = ""  # explicit endpoint, e.g. http://127.0.0.1:8000
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    atrex_bench_root: str = ""  # native evaluator checkout owning run_eval.py
    agent_cli: str = "claude"  # episode backend: claude, qodercli, codex, or pi
    optimization_mode: str = (
        "leaderboard"  # permissive contest flow or strict production gate
    )
    framework_baseline: str = (
        "auto"  # auto = production only; always | never override it
    )
    framework_baseline_timeout: int = FRAMEWORK_BASELINE_TIMEOUT_S
    handoff_resumes: int = DEFAULT_HANDOFF_RESUMES
    verify_repeats: int = DEFAULT_VERIFY_REPEATS
    verify_run_timeout: int = DEFAULT_VERIFY_RUN_TIMEOUT
    min_improvement_pct: float = 0.0
    long_reviewer_session: str = ""
    tokens_spent: int = field(default=0, init=False)
    _dependency_review_cache: dict[str, tuple[str, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _generated_agent_problem_digest: str = field(
        default="", init=False, repr=False, compare=False
    )
    _plan_reviewer_environment: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.long_reviewer_session and self.long_reviewer_session not in (
            _LONG_REVIEWER_SESSION_ENV
        ):
            raise NotImplementedError(
                f"long reviewer sessions are not implemented for "
                f"{self.long_reviewer_session}"
            )

    @property
    def campaign_name(self) -> str:
        suffix = f"_{self.workspace_suffix}" if self.workspace_suffix else ""
        return f"{self.name}{suffix}"

    @property
    def private_reference_dir(self) -> Path | None:
        """Return evaluator-only native inputs behind a generalized public problem."""
        op_dir = Path(self.kernel_demo).resolve().parent
        use_generalized = (
            self.optimization_mode == "production"
            and not is_sol_op(op_dir)
            and (op_dir / "shapes.json").is_file()
        )
        if not self.atrex_bench_root or not use_generalized:
            return None
        shapes_path = op_dir / "shapes.json"
        validate_private_shapes(shapes_path)
        if has_agent_problem(op_dir):
            validate_agent_problem(
                op_dir / AGENT_PROBLEM_FILENAME,
                private_shapes_path=shapes_path,
            )
        return op_dir

    def _ensure_agent_problem(self) -> None:
        """Materialize the public contract before any production optimization session.

        A user-authored contract is copied verbatim. When production receives only detailed
        evaluator shapes, a dedicated clean AKA session derives the public contract in a temporary
        directory; the later baseline/optimization sessions never receive ``shapes.json``.
        """
        private_dir = self.private_reference_dir
        if private_dir is None:
            return
        destination = self.workspace / AGENT_PROBLEM_FILENAME
        shapes_path = private_dir / "shapes.json"
        provided = private_dir / AGENT_PROBLEM_FILENAME
        if provided.is_file():
            validate_agent_problem(provided, private_shapes_path=shapes_path)
            shutil.copy2(provided, destination)
            print(
                f"[orchestrator] generalized problem: using user-provided {provided}",
                flush=True,
            )
            return
        if destination.is_file():
            validate_generated_agent_problem(
                destination,
                private_shapes_path=shapes_path,
            )
            self._generated_agent_problem_digest = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            print(
                "[orchestrator] generalized problem: reusing workspace-generated "
                f"{destination}",
                flush=True,
            )
            return
        if self.optimization_mode != "production":
            raise RuntimeError(
                "a generalized non-production campaign requires a user-provided "
                f"{AGENT_PROBLEM_FILENAME}"
            )

        print(
            "[orchestrator] generalized problem: production received detailed shapes only; "
            "starting AKA problem-authoring session",
            flush=True,
        )
        validation_error = ""
        with tempfile.TemporaryDirectory(
            prefix="aka-generalize-problem-"
        ) as raw_staging:
            staging = Path(raw_staging)
            for name in ("reference.py", "input.py", "shapes.json", "metadata.json"):
                source = private_dir / name
                if source.is_file():
                    shutil.copy2(source, staging / name)
            for attempt in range(2):
                repair_context = (
                    "The current agent_problem.json failed orchestrator validation. Replace it "
                    f"with a corrected file. Validation error: {validation_error}"
                    if validation_error
                    else "Create agent_problem.json now."
                )
                prompt = _render(
                    AGENT_PROBLEM_GENERATION_PROMPT,
                    REPAIR_CONTEXT=repair_context,
                )
                result = run_session(
                    staging,
                    prompt,
                    timeout=min(self.setup_timeout, 1_800),
                    agent_cli=self.agent_cli,
                    reasoning_effort="max",
                )
                self._account(result, f"agent problem generation attempt {attempt + 1}")
                generated = staging / AGENT_PROBLEM_FILENAME
                try:
                    validate_generated_agent_problem(
                        generated,
                        private_shapes_path=shapes_path,
                    )
                except ValueError as exc:
                    validation_error = str(exc)
                    if attempt == 0:
                        continue
                    detail = result.stderr_tail or result.stdout_tail
                    raise RuntimeError(
                        "AKA could not generate a valid generalized production problem after "
                        f"two attempts: {validation_error}"
                        + (f"; agent output: {detail}" if detail else "")
                    ) from exc
                shutil.copy2(generated, destination)
                self._generated_agent_problem_digest = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()
                print(
                    f"[orchestrator] generalized problem: generated {destination}",
                    flush=True,
                )
                return

    def agent_environment(self) -> dict[str, str]:
        private_dir = self.private_reference_dir
        environment = dict(self._plan_reviewer_environment)
        if self.long_reviewer_session:
            env_name = _LONG_REVIEWER_SESSION_ENV[self.long_reviewer_session]
            state_file = (
                self.workspace
                / f".atrex_long_horizon/{self.long_reviewer_session}_reviewer_session.json"
            )
            environment[env_name] = str(state_file.resolve())
        if private_dir is not None:
            environment[ATREX_PRIVATE_REFERENCE_ENV] = str(private_dir)
        return environment

    def ensure_plan_reviewer_availability(self) -> None:
        """Probe optional plan reviewers once and reuse the campaign-local decision."""
        if self._plan_reviewer_environment:
            return
        value, reused = discover_plan_reviewers(
            self.workspace,
            agent_cli=self.agent_cli,
        )
        self._plan_reviewer_environment = plan_reviewer_environment(value)
        statuses = []
        for name in ("codex", "qoder"):
            record = value["reviewers"][name]
            status = "available" if record["available"] else "disabled"
            statuses.append(f"{name}={status} ({record['reason']})")
        source = "cached" if reused else "startup probe"
        print(
            f"[orchestrator] plan reviewers ({source}): " + "; ".join(statuses),
            flush=True,
        )

    def _generalized_memory_coverage_problem(self, memory: dict | None) -> str:
        """Require successful canonical memory to cover every private shape by opaque id."""
        private_dir = self.private_reference_dir
        if private_dir is None or memory is None:
            return ""
        try:
            shapes = validate_private_shapes(private_dir / "shapes.json")
        except ValueError as exc:
            return f"cannot read private evaluator shape ids: {type(exc).__name__}"
        performance = memory.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        measured = performance.get("latency_us_by_shape")
        measured = measured if isinstance(measured, dict) else {}
        expected_ids = {str(value) for value in shapes}
        measured_ids = {str(value) for value in measured}
        if measured_ids != expected_ids:
            return (
                "canonical memory lacks complete real-shape performance coverage "
                f"({len(measured_ids)}/{len(expected_ids)})"
            )
        return ""

    def _generalized_contract_commit_problem(self) -> str:
        """Require the public problem to be part of the immutable V0 history."""
        if self.private_reference_dir is None:
            return ""
        head = git_head(self.workspace)
        if head and not git_path_blob(self.workspace, head, AGENT_PROBLEM_FILENAME):
            return "agent_problem.json is not tracked by the V0 baseline commit"
        return ""

    def _assert_generalized_inputs_are_private(self) -> None:
        """Fail closed if exact evaluator artifacts appear in the agent workspace."""
        private_dir = self.private_reference_dir
        if private_dir is None:
            return
        public_problem = self.workspace / AGENT_PROBLEM_FILENAME
        if not public_problem.is_file():
            raise RuntimeError(
                "generalized Atrex-Bench workspace is missing agent_problem.json; "
                "start a fresh workspace"
            )
        try:
            provided_problem = private_dir / AGENT_PROBLEM_FILENAME
            if provided_problem.is_file():
                validate_agent_problem(
                    public_problem,
                    private_shapes_path=private_dir / "shapes.json",
                )
                if public_problem.read_bytes() != provided_problem.read_bytes():
                    raise ValueError(
                        "workspace agent_problem.json differs from the user-provided contract"
                    )
            else:
                validate_generated_agent_problem(
                    public_problem,
                    private_shapes_path=private_dir / "shapes.json",
                )
                if (
                    self._generated_agent_problem_digest
                    and hashlib.sha256(public_problem.read_bytes()).hexdigest()
                    != self._generated_agent_problem_digest
                ):
                    raise ValueError(
                        "workspace agent_problem.json was modified after automatic generation"
                    )
        except ValueError as exc:
            raise RuntimeError(
                f"generalized Atrex-Bench workspace has an invalid public problem: {exc}; "
                "start a fresh workspace"
            ) from exc
        leaked = [
            name
            for name in ("shapes.json", "metadata.json", "roofline.json", "valid.py")
            if (self.workspace / name).exists()
        ]
        if leaked:
            raise RuntimeError(
                "generalized Atrex-Bench workspace exposes evaluator-only files: "
                + ", ".join(leaked)
                + "; start a fresh workspace"
            )

    @property
    def workspace(self) -> Path:
        base = Path(self.work_dir) if self.work_dir else Path.cwd()
        return base / f"kernel_opt_{self.campaign_name}"

    def _account(self, res: SessionResult, label: str) -> None:
        self.tokens_spent += res.tokens
        print(
            f"[orchestrator] {label}: exit={res.exit_status} timed_out={res.timed_out} "
            f"tokens={res.tokens} cum_tokens={self.tokens_spent}",
            flush=True,
        )
        if res.exit_status != 0 or res.timed_out:
            print(
                f"[orchestrator] stderr tail:\n{res.stderr_tail}",
                file=sys.stderr,
                flush=True,
            )

    def _review_third_party_dependencies(
        self,
        workspace: Path,
        framework: str,
        signals: tuple[DependencyReviewSignal, ...],
    ) -> list[str]:
        """Delegate ambiguous dependency provenance to a fresh, isolated agent."""
        cache_key = (
            _dependency_review_digest(workspace, framework, signals)
            + ":"
            + self.agent_cli
        )
        cached = self._dependency_review_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        errors: list[str]
        with tempfile.TemporaryDirectory(
            prefix="atrex-dependency-review-"
        ) as directory:
            review_workspace = Path(directory)
            candidate_root = review_workspace / "candidate"
            source_hashes: dict[str, str] = {}
            for source in _dependency_review_candidate_paths(workspace):
                relative = source.relative_to(workspace)
                destination = candidate_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                source_hashes[relative.as_posix()] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()

            request = {
                "schema_version": DEPENDENCY_REVIEW_SCHEMA_VERSION,
                "framework": framework,
                "optimization_mode": "production",
                "signals": [
                    {
                        "id": review_signal.id,
                        "kind": review_signal.kind,
                        "value": review_signal.value,
                    }
                    for review_signal in signals
                ],
                "candidate_files": sorted(source_hashes),
            }
            (review_workspace / "review_request.json").write_text(
                json.dumps(request, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = run_session(
                review_workspace,
                DEPENDENCY_REVIEW_PROMPT.read_text(encoding="utf-8"),
                timeout=DEPENDENCY_REVIEW_TIMEOUT_S,
                agent_cli=self.agent_cli,
                reasoning_effort="low",
                agent_plugins=False,
            )
            self._account(result, "independent dependency review")
            if result.exit_status != 0 or result.timed_out:
                errors = [
                    "independent dependency review agent failed "
                    f"(exit={result.exit_status}, timeout={result.timed_out})"
                ]
            else:
                changed = []
                for relative, expected_hash in source_hashes.items():
                    candidate_path = candidate_root / relative
                    if (
                        not candidate_path.is_file()
                        or hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                        != expected_hash
                    ):
                        changed.append(relative)
                if changed:
                    errors = [
                        "independent dependency review modified candidate evidence: "
                        + ", ".join(sorted(changed))
                    ]
                else:
                    review_path = review_workspace / "dependency_review.json"
                    try:
                        payload = json.loads(review_path.read_text(encoding="utf-8"))
                        errors, summary = _validate_dependency_review(payload, signals)
                    except (
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                        ValueError,
                    ) as exc:
                        errors = [
                            "independent dependency review produced no valid verdict: "
                            f"{type(exc).__name__}: {exc}"
                        ]
                    else:
                        status = "accepted" if not errors else "rejected"
                        print(
                            f"[production-policy] independent dependency review {status}: "
                            f"{summary}",
                            flush=True,
                        )

        self._dependency_review_cache[cache_key] = tuple(errors)
        return list(errors)

    def _production_kernel_violations(
        self,
        workspace: Path | None = None,
        *,
        require_gluon: bool = False,
    ) -> list[str]:
        return production_kernel_violations(
            workspace or self.workspace,
            self.framework,
            require_gluon=require_gluon,
            dependency_reviewer=self._review_third_party_dependencies,
        )

    def _link_runtime(self) -> None:
        self._assert_generalized_inputs_are_private()
        native_root = Path(self.atrex_bench_root) if self.atrex_bench_root else None
        link_runtime(self.workspace, native_root)
        install_workspace_policy(
            self.workspace,
            self.optimization_mode,
            self.framework,
            agent_runtime=self.agent_cli,
        )

    def _evaluator_directive(self) -> str:
        if self.atrex_bench_root:
            if self.private_reference_dir is not None:
                return (
                    "## Evaluation route: Atrex-Bench generalized private-case native\n\n"
                    "Treat workspace `agent_problem.json` as the authoritative public optimization "
                    "contract. Exact `shapes.json`, `metadata.json`, and `roofline.json` cases are "
                    "evaluator-only and intentionally absent from the workspace; never search for, "
                    "reconstruct, or read the private reference directory. The immutable "
                    "`test_kernel.py` adapter and sandbox inject those cases only into the remote "
                    "official evaluator. Optimize for the complete declared `shape_domain`, using "
                    "aggregate `distribution_profile` shares only for prioritization. Correctness "
                    "must pass every hidden case. After evaluation, use the real "
                    "`latency_us_by_shape` map keyed by opaque ids without attempting to infer their "
                    "private inputs. For profiling, choose a real opaque id from canonical "
                    "`memory/vN.json.performance.latency_us_by_shape` with PROFILE_SHAPE_ID; the "
                    "sandbox injects only that real case into the remote profile job. Do not edit "
                    "or replace the adapter or implement a custom correctness/timing harness."
                )
            return (
                "## Evaluation route: Atrex-Bench native\n\n"
                "This workspace's `test_kernel.py` is an orchestrator-installed immutable adapter. "
                "It invokes the canonical `atrex-bench/scripts/run_eval.py` against `kernel.py` and "
                "the workspace `reference.py`/`input.py`/`shapes.json`/`metadata.json`, then emits "
                "the optimizer's `RESULT_JSON` transport line from the official `eval_result.json`. "
                "Do not edit or replace this adapter and do not implement a custom correctness or "
                "timing harness. `--multi-seed N` maps to N additional Atrex-Bench correctness "
                "cases while performance remains one official run per shape."
            )
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            return (
                "## Evaluation route: SOL-ExecBench\n\n"
                "Keep using the immutable SOL `test_kernel.py`, which invokes `sol-execbench` over "
                "the complete `workload.jsonl`. Do not substitute the Atrex-Bench native evaluator."
            )
        return (
            "## Evaluation route: derived legacy boundary\n\n"
            "This derived boundary is not a complete Atrex-Bench operator directory. Preserve its "
            "committed full-shape `test_kernel.py` methodology and do not replace it after V0."
        )

    def _install_native_evaluator(self) -> None:
        """Seed the immutable adapter used only by native Atrex-Bench shape campaigns."""
        if not self.atrex_bench_root:
            return
        if not ATREX_BENCH_HARNESS.is_file():
            raise FileNotFoundError(f"missing {ATREX_BENCH_HARNESS}")
        shutil.copy2(ATREX_BENCH_HARNESS, self.workspace / "test_kernel.py")

    def _install_profile_driver(self) -> None:
        """Seed the immutable external profiling entry for every campaign layout.

        Both profiler wrappers run ``python <file>``, so profiling needs a runnable script.
        Keeping it out of ``kernel.py`` means a session that rewrites ``run()``/``Model``
        cannot silently destroy profiling: an in-kernel ``__main__`` block would vanish with
        the rewrite and the profiler would still exit 0 having captured nothing.
        """
        if not PROFILE_DRIVER.is_file():
            raise FileNotFoundError(f"missing {PROFILE_DRIVER}")
        shutil.copy2(PROFILE_DRIVER, self.workspace / "profile_driver.py")
        # Stage it when a repository already exists so the baseline commit tracks it without
        # depending on how the setup session stages files; restoring an immutable path needs
        # a blob in the root commit.
        if (self.workspace / ".git").exists():
            subprocess.run(
                ["git", "add", "profile_driver.py"],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _sandbox_directive(self) -> str:
        return sandbox_directive(
            self.sandbox_hardware, self.sandbox_profile, self.sandbox_url
        )

    def _mode_directive(self) -> str:
        return optimization_mode_directive(self.optimization_mode, self.framework)

    def setup_baseline(self) -> None:
        # SOL-ExecBench op: seed a correct, directly-submittable V0 mechanically
        # (no baseline session) — sol_seed.py copies the ground-truth files, writes
        # the DPS wrapper kernel.py + solution.json; this method benches V0 in the sandbox.
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            self._setup_baseline_sol(op_dir)
            return
        if not WORKSPACE_INIT.exists():
            raise FileNotFoundError(f"missing {WORKSPACE_INIT}")
        # workspace_init.sh builds the workspace as $(pwd)/kernel_opt_<name>,
        # so cwd must be the work_dir (or the process cwd when --workspace is absent).
        subprocess.run(
            ["bash", str(WORKSPACE_INIT), self.campaign_name, self.kernel_demo],
            cwd=str(self.workspace.parent),
            check=True,
        )
        # Production native tasks always expose a generalized public contract. Exact shapes and
        # release metadata remain in the source operator directory and are injected only at the
        # sandbox boundary. A missing public contract is authored before the baseline session.
        generalized = self.private_reference_dir is not None
        for name in agent_visible_operator_files(op_dir, generalized=generalized):
            source = op_dir / name
            if source.is_file():
                shutil.copy2(source, self.workspace / name)
        self._ensure_agent_problem()
        if generalized:
            # Seed the immutable public contract into the eventual V0 commit even when the
            # baseline agent stages only its own implementation files.
            subprocess.run(
                ["git", "add", AGENT_PROBLEM_FILENAME],
                cwd=str(self.workspace),
                check=True,
            )
        self._link_runtime()
        self._install_native_evaluator()
        self._install_profile_driver()
        prompt = _render(
            PROMPTS_DIR / "setup.md",
            WORKSPACE=str(self.workspace),
            PLATFORM=self.platform,
            FRAMEWORK=self.framework,
            KERNEL_DEMO="reference.py",
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            BASELINE_DRIVER=_baseline_driver_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            EVALUATOR=self._evaluator_directive(),
            MODE_POLICY=self._mode_directive(),
        )
        res = run_session(
            self.workspace,
            prompt,
            timeout=self.setup_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
            reasoning_effort="high",
            extra_environment=self.agent_environment(),
        )
        self._assert_generalized_inputs_are_private()
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0) — "
                "this is likely an API key / authentication issue. "
                f"{_agent_runtime.auth_hint(self.agent_cli)}."
            )
        baseline_memory = read_memory(self.workspace, 0)
        baseline_problem = "missing memory/v0.json" if baseline_memory is None else ""
        if baseline_memory is not None and not git_head(self.workspace):
            baseline_problem = "memory/v0.json exists but the workspace has no Git HEAD"
        if not baseline_problem:
            baseline_problem = self._generalized_memory_coverage_problem(
                baseline_memory
            )
        if not baseline_problem:
            baseline_problem = self._generalized_contract_commit_problem()
        if baseline_problem:
            print(
                f"[orchestrator] WARNING: incomplete setup ({baseline_problem}); "
                "starting one clean recovery session",
                file=sys.stderr,
                flush=True,
            )
            recovery_prompt = (
                self._mode_directive()
                + "\n\n# Recover incomplete V0 setup\n\n"
                + f"Workspace: `{self.workspace}`\n\n"
                + "A previous non-interactive setup session stopped before producing the required "
                f"baseline ({baseline_problem}). Continue from the files already present and finish V0 "
                "autonomously. "
                "Do not ask the user for confirmation or permission. Inspect the current workspace, "
                "implement `kernel.py`, preserve the evaluator route described below, run the complete "
                "workspace workload through the mandatory sandbox with `--no-memory`, parse its "
                "`[test_kernel] RESULT_JSON=...`, write local `memory/v0.json` and `baseline_report.md`, "
                "then Git commit `V0: baseline kernel`. Do not enter optimization iterations.\n\n"
                + self._evaluator_directive()
                + "\n\n"
                + self._sandbox_directive()
            )
            recovery = run_session(
                self.workspace,
                recovery_prompt,
                timeout=self.setup_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
                reasoning_effort="high",
                extra_environment=self.agent_environment(),
            )
            self._assert_generalized_inputs_are_private()
            self._account(recovery, "setup recovery")
            if recovery.exit_status != 0 and recovery.tokens == 0:
                raise RuntimeError(
                    f"setup recovery failed immediately (exit={recovery.exit_status}, tokens=0) — "
                    f"{_agent_runtime.auth_hint(self.agent_cli)}."
                )
            recovered_memory = read_memory(self.workspace, 0)
            recovery_problem = (
                "missing memory/v0.json" if recovered_memory is None else ""
            )
            if recovered_memory is not None and not git_head(self.workspace):
                recovery_problem = (
                    "memory/v0.json exists but the workspace still has no Git HEAD"
                )
            if not recovery_problem:
                recovery_problem = self._generalized_memory_coverage_problem(
                    recovered_memory
                )
            if not recovery_problem:
                recovery_problem = self._generalized_contract_commit_problem()
            if recovery_problem:
                detail = recovery.stderr_tail or recovery.stdout_tail
                raise RuntimeError(
                    f"setup recovery left an incomplete baseline ({recovery_problem})"
                    + (f": {detail}" if detail else "")
                )

    def _setup_baseline_sol(self, op_dir: Path) -> None:
        if not SOL_SEED.exists():
            raise FileNotFoundError(f"missing {SOL_SEED}")
        cmd = [
            sys.executable,
            str(SOL_SEED),
            "--op-dir",
            str(op_dir),
            "--name",
            self.campaign_name,
            "--workspace",
            str(self.workspace),
            "--framework",
            self.framework,
            "--platform",
            self.platform,
            # The local step only materializes sources and git state.  GPU
            # correctness/performance is run below in the remote sandbox.
            "--no-bench",
        ]
        subprocess.run(cmd, check=True)
        self._link_runtime()
        test = _sandbox_command(
            self.workspace,
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_timeout,
            ["python", "test_kernel.py", "--version", "v0", "--no-memory"],
            gateway_kind="run",
        )
        if test.stdout:
            print(
                test.stdout, end="" if test.stdout.endswith("\n") else "\n", flush=True
            )
        if test.stderr:
            print(
                test.stderr,
                end="" if test.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"sandbox V0 baseline produced no usable result: {exc}"
            ) from exc
        memory_path = _record_local_test_result(self.workspace, "v0", result)
        if test.returncode != 0 or not result.get("all_pass"):
            raise RuntimeError(
                "sandbox V0 baseline failed correctness/performance validation"
            )

        # sol_seed committed the source-only baseline. Fold the locally-owned
        # memory record into that commit without ever sending memory to the pod.
        mem = json.loads(memory_path.read_text(encoding="utf-8"))
        mem["git_commit_hash"] = git_head(self.workspace)
        mem.setdefault("optimization", {})["action_category"] = "baseline"
        memory_path.write_text(
            json.dumps(mem, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        subprocess.run(
            ["git", "add", "memory/v0.json", "CLAUDE.md", ".gitignore"],
            cwd=str(self.workspace),
            check=True,
        )
        subprocess.run(
            ["git", "commit", "--amend", "--no-edit"],
            cwd=str(self.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def ensure_framework_baseline(self) -> None:
        """Land the campaign's first real framework kernel as v1, exactly once.

        V0 is a PyTorch reference wrapper. This stage pays the framework bring-up cost once
        before optimization starts from a self-contained implementation.

        Idempotent and resume-safe: a pinned baseline is never rewritten, and a campaign that has
        already progressed past V0 without a pin is left exactly as it is.
        """
        action, reason = self._framework_baseline_decision()
        if action == "skip":
            if reason:
                print(
                    f"[orchestrator] framework baseline skipped: {reason}", flush=True
                )
            return
        print(f"[orchestrator] framework baseline: {action} ({reason})", flush=True)
        if action == "pin":
            baseline_commit = self._v0_baseline_commit()
            self._pin_framework_baseline(baseline_commit, version=0)
            return

        n = FRAMEWORK_BASELINE_VERSION
        baseline_commit = self._v0_baseline_commit()
        v0_blob = git_path_blob(self.workspace, baseline_commit, "kernel.py")
        pre_head = git_head(self.workspace)

        if action == "run":
            self._link_runtime()
            self._sync_framework_baseline_live(phase="framework_baseline")
            res = run_session(
                self.workspace,
                self._framework_baseline_prompt(n),
                timeout=self.framework_baseline_timeout,
                agent_cli=self.agent_cli,
                sandbox_hardware=self.sandbox_hardware,
                sandbox_profile=self.sandbox_profile,
                sandbox_url=self.sandbox_url,
                sandbox_timeout=self.sandbox_timeout,
                reasoning_effort="high",
                extra_environment=self.agent_environment(),
            )
            self._assert_generalized_inputs_are_private()
            self._account(res, f"framework baseline v{n}")
            if res.exit_status != 0 and res.tokens == 0:
                raise RuntimeError(
                    "framework baseline session produced no output "
                    f"(likely API key / auth issue — {_agent_runtime.auth_hint(self.agent_cli)})"
                )
            self._warn_restored_baseline_paths(baseline_commit)
            problem = self._framework_baseline_problem(v0_blob, baseline_commit)
            if problem:
                self._recover_framework_baseline(
                    problem, v0_blob, baseline_commit, pre_head
                )
                self._warn_restored_baseline_paths(baseline_commit)
                problem = self._framework_baseline_problem(v0_blob, baseline_commit)
        else:  # adopt: our own interrupted run already committed the kernel
            self._sync_framework_baseline_live(phase="framework_baseline")
            self._warn_restored_baseline_paths(baseline_commit)
            problem = self._framework_baseline_problem(v0_blob, baseline_commit)
        result: Optional[dict] = None
        if not problem:
            result, problem = self._validate_framework_baseline(n)
        if problem:
            self._record_framework_baseline_failure(problem)
            self._sync_framework_baseline_live(
                phase="failed",
                state="blocked",
                accepted=False,
                outcome={"summary": problem, "next_directions": []},
            )
            if pre_head:
                subprocess.run(
                    ["git", "reset", "--hard", pre_head],
                    cwd=str(self.workspace),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            raise RuntimeError(f"framework baseline v{n} rejected: {problem}")

        commit = self._commit_framework_baseline(n, result or {})
        self._sync_framework_baseline_live(
            phase="recorded",
            state="candidate_ready",
            accepted=True,
            canonical_memory=f"memory/v{n}.json",
            candidate_commit=commit,
            outcome={
                "summary": f"accepted self-contained {self.framework} framework baseline",
                "next_directions": [],
            },
        )
        latency = ((read_memory(self.workspace, n) or {}).get("performance") or {}).get(
            "latency_us"
        )
        print(
            f"[orchestrator] framework baseline v{n} accepted: {self.framework} "
            f"@ {commit[:8]} ({latency} us geomean)",
            flush=True,
        )

    def _sync_framework_baseline_live(
        self,
        *,
        phase: str,
        state: str = "in_progress",
        accepted: bool | None = None,
        canonical_memory: str = "",
        candidate_commit: str = "",
        outcome: dict | None = None,
    ) -> None:
        """Best-effort live progress before the Long Horizon supervisor starts."""
        try:
            from long_horizon.journal import sync_live_memory
            from long_horizon.store import CampaignStore

            store = CampaignStore(self.workspace)
            created_at = datetime.now(timezone.utc).isoformat()
            try:
                existing = json.loads(
                    store.live_memory_path.read_text(encoding="utf-8")
                )
                if isinstance(existing, dict) and existing.get("created_at"):
                    created_at = str(existing["created_at"])
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            value = {
                "schema_version": 1,
                "episode": 0,
                "memory_version": FRAMEWORK_BASELINE_VERSION,
                "base_commit": git_head(self.workspace),
                "episode_branch": "framework-baseline",
                "state": state,
                "experiments": [],
                "outcome": outcome,
                "candidate_commit": candidate_commit or None,
                "created_at": created_at,
            }
            sync_live_memory(
                store.live_memory_path,
                value,
                phase=phase,
                canonical_memory=canonical_memory,
                accepted=accepted,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                "[orchestrator] WARNING: could not update framework-baseline "
                f"memory/live.json: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )

    def _framework_baseline_decision(self) -> tuple[str, str]:
        """Resolve what the stage should do: skip | pin | run | adopt, with the reason."""
        if self.framework_baseline == "never":
            return "skip", ""
        if latest_version(self.workspace) < 0 or not git_head(self.workspace):
            raise RuntimeError(
                "framework baseline requires a committed V0 baseline first"
            )

        pinned_commit, pinned_version = resolve_framework_baseline_commit(
            self.workspace
        )
        if pinned_commit:
            return "skip", f"already pinned at {pinned_commit[:8]} (v{pinned_version})"

        violations = self._production_kernel_violations()
        progressed = not head_kernel_is_initial_baseline(self.workspace)
        if not violations and not progressed:
            return (
                "pin",
                "the V0 kernel is already a compliant framework implementation",
            )
        if any(v.startswith("unsupported production framework") for v in violations):
            return "skip", "; ".join(violations)
        if self.framework_baseline == "auto" and self.optimization_mode != "production":
            return (
                "skip",
                "leaderboard mode keeps the permissive V0 (use --framework-baseline always)",
            )
        if not progressed:
            return "run", f"V0 is not a self-contained {self.framework} kernel"
        if (
            latest_version(self.workspace) == FRAMEWORK_BASELINE_VERSION
            and not violations
        ):
            return "adopt", "an interrupted framework baseline is already committed"
        return "skip", (
            "HEAD has progressed beyond V0 without a framework-baseline pin; "
            "leaving this campaign on its existing baseline"
        )

    def _v0_baseline_commit(self) -> str:
        commit = v0_baseline_commit(self.workspace)
        if not commit:
            raise RuntimeError("framework baseline requires a committed V0 kernel.py")
        return commit

    def _framework_baseline_prompt(self, n: int) -> str:
        return _render(
            PROMPTS_DIR / "framework_baseline.md",
            WORKSPACE=str(self.workspace),
            N=n,
            PREV=n - 1,
            PLATFORM=self.platform,
            FRAMEWORK=self.framework,
            ARCH=self.arch or "the runtime GPU arch",
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            EVALUATOR=self._evaluator_directive(),
            MODE_POLICY=self._mode_directive(),
        )

    def _restore_immutable_baseline_paths(self, baseline_commit: str) -> list[str]:
        """Put back any ground-truth file the session edited, and report what was restored.

        A session that "fixes" the harness or memory/v0.json is a compliance problem, but a
        mechanically repairable one — discarding its kernel over it would throw away hours of
        work for nothing. Acceptance is decided by the kernel itself.
        """
        restored: list[str] = []
        for path in IMMUTABLE_BASELINE_PATHS:
            original = git_path_blob(self.workspace, baseline_commit, path)
            if not original or original == git_worktree_blob(self.workspace, path):
                continue
            checkout = subprocess.run(
                ["git", "checkout", baseline_commit, "--", path],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if checkout.returncode == 0:
                restored.append(path)
        return restored

    def _framework_baseline_problem(self, v0_blob: str, baseline_commit: str) -> str:
        """Static acceptance checks on the candidate about to be validated and committed.

        Everything is judged from the worktree: that is what the gateway uploads, and it lets a
        session that wrote the kernel but never committed it still be accepted.
        """
        candidate_blob = git_worktree_blob(self.workspace, "kernel.py")
        if not candidate_blob or candidate_blob == v0_blob:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        violations = self._production_kernel_violations()
        if violations:
            return (
                f"the candidate is not a self-contained {self.framework} implementation: "
                + "; ".join(violations)
            )
        if self.framework.lower() in {"triton", "gluon"} and kernel_is_gluon(
            self.workspace
        ):
            # A Gluon v1 would permanently disarm the orchestrator's mandatory Triton->Gluon latch.
            return "the framework baseline must be plain Triton; Gluon is a later orchestrator escalation"
        mutated = [
            path
            for path in IMMUTABLE_BASELINE_PATHS
            if git_path_blob(self.workspace, baseline_commit, path)
            and git_path_blob(self.workspace, baseline_commit, path)
            != git_worktree_blob(self.workspace, path)
        ]
        if mutated:
            return "the session modified immutable ground truth: " + ", ".join(mutated)
        return ""

    def _validate_framework_baseline(self, n: int) -> tuple[Optional[dict], str]:
        """Re-validate the candidate through the gateway: single seed, then five seeds."""
        # V1 is a correctness/framework bring-up gate, not a performance gate. Keep a
        # small timing sample so a slow but valid first implementation can enter the
        # optimization loop without exhausting the evaluator's benchmark budget.
        timing_args = ["--timed-runs", "5"]
        stages = (
            (
                "single-seed",
                [
                    "python",
                    "test_kernel.py",
                    "--version",
                    f"v{n}",
                    *timing_args,
                    "--no-memory",
                ],
            ),
            (
                "multi-seed",
                [
                    "python",
                    "test_kernel.py",
                    "--version",
                    f"v{n}",
                    "--multi-seed",
                    "5",
                    *timing_args,
                    "--no-memory",
                ],
            ),
        )
        result: Optional[dict] = None
        for stage_name, command in stages:
            try:
                test = _sandbox_command(
                    self.workspace,
                    self.sandbox_hardware,
                    self.sandbox_profile,
                    self.sandbox_url,
                    self.sandbox_timeout,
                    command,
                    gateway_kind="run",
                    private_reference_dir=self.private_reference_dir,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return None, f"{stage_name} validation failed to run: {exc}"
            if test.stdout:
                print(
                    test.stdout,
                    end="" if test.stdout.endswith("\n") else "\n",
                    flush=True,
                )
            if test.stderr:
                print(
                    test.stderr,
                    end="" if test.stderr.endswith("\n") else "\n",
                    file=sys.stderr,
                    flush=True,
                )
            if test.returncode != 0:
                return (
                    None,
                    f"{stage_name} validation command failed (exit={test.returncode})",
                )
            try:
                result = _test_result_from_stdout(test.stdout)
            except (RuntimeError, json.JSONDecodeError) as exc:
                return None, f"{stage_name} validation produced no usable result: {exc}"
            if not result.get("all_pass"):
                return None, f"{stage_name} correctness validation failed"

        assert result is not None
        latency = result.get("latency_us_geomean")
        if not isinstance(latency, (int, float)) or latency <= 0:
            return None, "validation reported no usable latency_us_geomean"
        # Require the framework baseline to preserve full-workload measurement coverage.
        baseline_shapes = set(
            ((read_memory(self.workspace, 0) or {}).get("performance") or {}).get(
                "latency_us_by_shape", {}
            )
        )
        measured_shapes = set(result.get("latency_us_by_shape") or {})
        if baseline_shapes and measured_shapes != baseline_shapes:
            return None, (
                "latency_us_by_shape does not cover the same workloads as v0 "
                f"(missing {sorted(baseline_shapes - measured_shapes)}, "
                f"unexpected {sorted(measured_shapes - baseline_shapes)})"
            )
        return result, ""

    def _warn_restored_baseline_paths(self, baseline_commit: str) -> None:
        restored = self._restore_immutable_baseline_paths(baseline_commit)
        if restored:
            print(
                "[orchestrator] framework baseline session edited immutable ground truth; "
                f"restored from V0: {', '.join(restored)}",
                file=sys.stderr,
                flush=True,
            )

    def _recover_framework_baseline(
        self, problem: str, v0_blob: str, baseline_commit: str, pre_head: str
    ) -> None:
        """Run one clean recovery session for a rejected candidate."""
        print(
            f"[orchestrator] WARNING: framework baseline rejected ({problem}); "
            "starting one clean recovery session",
            file=sys.stderr,
            flush=True,
        )
        if pre_head and git_head(self.workspace) != pre_head:
            # Undo the session's commits, keep its files: the recovery session needs to read them.
            subprocess.run(
                ["git", "reset", "--soft", pre_head],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        recovery_prompt = (
            self._mode_directive()
            + "\n\n# Recover a rejected framework baseline\n\n"
            + f"Workspace: `{self.workspace}`\n\n"
            + "A previous non-interactive session tried to replace the V0 PyTorch wrapper with a "
            + f"self-contained **{self.framework}** kernel and was rejected: "
            + f"**{problem}**\n\n"
            + "Continue from the files already present and finish the job autonomously. Do not ask "
            + "for confirmation. Keep the algorithm you already have where it is sound, fix the "
            + "stated problem, validate correctness through the sandbox with `--multi-seed 5`, "
            + f"write `memory/v{FRAMEWORK_BASELINE_VERSION}.json`, and commit `kernel.py`. Never "
            + "modify `test_kernel.py`, `reference.py`, `input.py`, `agent_problem.json`, "
            + "`shapes.json`, or `memory/v0.json`, "
            + f"or create `{FRAMEWORK_BASELINE_FILE}`. Do not enter optimization iterations.\n\n"
            + self._evaluator_directive()
            + "\n\n"
            + self._sandbox_directive()
        )
        recovery = run_session(
            self.workspace,
            recovery_prompt,
            timeout=self.framework_baseline_timeout,
            agent_cli=self.agent_cli,
            sandbox_hardware=self.sandbox_hardware,
            sandbox_profile=self.sandbox_profile,
            sandbox_url=self.sandbox_url,
            sandbox_timeout=self.sandbox_timeout,
            reasoning_effort="high",
            extra_environment=self.agent_environment(),
        )
        self._assert_generalized_inputs_are_private()
        self._account(
            recovery, f"framework baseline recovery v{FRAMEWORK_BASELINE_VERSION}"
        )

    def _record_framework_baseline_failure(self, problem: str) -> None:
        """Persist why the framework baseline was rejected, uncommitted so a reset cannot lose it."""
        n = FRAMEWORK_BASELINE_VERSION
        memory_path = self.workspace / "memory" / f"v{n}.json"
        try:
            memory = (
                json.loads(memory_path.read_text(encoding="utf-8"))
                if memory_path.exists()
                else {}
            )
        except (OSError, json.JSONDecodeError):
            memory = {}
        if not isinstance(memory, dict):
            memory = {}
        memory["version"] = f"v{n}"
        memory["masked"] = False
        memory["git_commit_hash"] = None
        memory["quality_gate"] = {"result": "FAIL", "failure_reason": problem}
        memory["correctness"] = {"status": "FAIL"}
        memory["optimization"] = {
            "action_category": FRAMEWORK_BASELINE_CATEGORY,
            "action_description": f"rejected {self.framework} baseline attempt",
        }
        pitfalls = memory.setdefault("pitfalls_and_fixes", [])
        if not isinstance(pitfalls, list):
            pitfalls = []
            memory["pitfalls_and_fixes"] = pitfalls
        pitfalls.append(
            {
                "error_type": "production_policy"
                if "self-contained" in problem
                else "correctness",
                "error_message": problem,
                "lesson": f"the next attempt must land a compliant, correctness-passing {self.framework} kernel",
            }
        )
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _commit_framework_baseline(self, n: int, result: dict) -> str:
        """Commit the accepted kernel (C1) and then pin it in a metadata-only commit (C2)."""
        staged = [
            path
            for path in (
                "kernel.py",
                "solution.json",
                "CLAUDE.md",
                "README.md",
                f"memory/v{n}.json",
            )
            if (self.workspace / path).exists()
        ]
        staged += [
            str(path.relative_to(self.workspace))
            for path in sorted(self.workspace.glob(f"plans/v{n}_*.md"))
        ]
        subprocess.run(
            ["git", "add", *staged],
            cwd=str(self.workspace),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.workspace),
                check=False,
            ).returncode
            != 0
        ):
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"v{n}: framework baseline ({self.framework}) replacing the V0 PyTorch wrapper",
                ],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        kernel_commit = subprocess.run(
            ["git", "rev-list", "-1", "HEAD", "--", "kernel.py"],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if git_kernel_blob(self.workspace) != git_worktree_blob(
            self.workspace, "kernel.py"
        ):
            raise RuntimeError(
                "framework baseline kernel.py differs between the worktree and the commit"
            )

        _record_local_test_result(self.workspace, f"v{n}", result)
        memory_path = self.workspace / "memory" / f"v{n}.json"
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = FRAMEWORK_BASELINE_CATEGORY
        optimization["action_description"] = (
            f"first self-contained {self.framework} implementation of the whole operator"
        )
        memory["git_commit_hash"] = kernel_commit
        memory[FRAMEWORK_BASELINE_CATEGORY] = {
            "framework": self.framework,
            "validated_stages": ["single-seed", "multi-seed-5"],
        }
        memory_path.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._pin_framework_baseline(kernel_commit, version=n)
        return kernel_commit

    def _pin_framework_baseline(self, commit: str, *, version: int) -> None:
        """Write and commit the framework-baseline marker.

        Deliberately a separate commit rather than an amend: amending would rewrite the very
        commit whose sha the marker records, leaving a dangling pointer. This commit does not
        touch kernel.py, so it never registers as an optimization win.
        """
        marker = {
            "schema_version": 1,
            "version": f"v{version}",
            "framework": self.framework,
            "platform": self.platform,
            "arch": self.arch,
            "commit": commit,
            "kernel_blob": git_path_blob(self.workspace, commit, "kernel.py"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.workspace / FRAMEWORK_BASELINE_FILE).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths = [FRAMEWORK_BASELINE_FILE]
        if (self.workspace / "memory" / f"v{version}.json").exists():
            paths.append(f"memory/v{version}.json")
        subprocess.run(
            ["git", "add", *paths],
            cwd=str(self.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.workspace),
                check=False,
            ).returncode
            != 0
        ):
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"v{version}: pin framework baseline {commit[:8]}",
                ],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        # The metadata commit must not read as a stalled optimization round on the next resume.
        write_stall(self.workspace, 0)

    def run(self) -> str:
        """Run the native long-horizon episode supervisor for this campaign."""
        from long_horizon.campaign import LongHorizonCampaign
        from long_horizon.session import LongSessionRunner
        from long_horizon.store import CampaignStore
        from long_horizon.verifier import GatewayABBAValidator

        CampaignStore.ensure_excluded(self.workspace)
        self.ensure_plan_reviewer_availability()

        verifier = GatewayABBAValidator(
            hardware=self.sandbox_hardware,
            profile=self.sandbox_profile,
            url=self.sandbox_url,
            timeout=self.sandbox_timeout,
            repeats=self.verify_repeats,
            per_run_timeout=self.verify_run_timeout,
            min_improvement_pct=self.min_improvement_pct,
            private_reference_dir=self.private_reference_dir,
        )
        engine = LongHorizonCampaign(
            base_campaign=self,
            max_version=self.max_iters,
            token_budget=self.token_budget,
            handoff_resumes=self.handoff_resumes,
            max_stall=self.max_stall,
            verifier=verifier,
            session_runner=LongSessionRunner(agent_cli=self.agent_cli),
        )
        return self._finish(engine.run())

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "memory_manager.py"),
                    "summary",
                    "--workspace",
                    str(self.workspace),
                ],
                check=False,
            )
        except OSError:
            pass
        # Production output is fail-closed: do not package a PyTorch baseline,
        # alternate DSL, or independently rejected dependency as a production candidate.
        if self.optimization_mode == "production":
            violations = self._production_kernel_violations()
            if violations:
                raise RuntimeError(
                    "no production-compliant final kernel: " + "; ".join(violations)
                )
        # SOL op: emit the self-contained, validated submission (SOL's output format).
        if (self.workspace / "definition.json").exists() and (
            self.workspace / "solution.json"
        ).exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "reference" / "sol_finalize.py"),
                        "--workspace",
                        str(self.workspace),
                    ],
                    check=False,
                )
            except OSError:
                pass
        return reason
