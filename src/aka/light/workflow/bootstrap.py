"""Single-operator optimization campaign: baseline setup, episode loop, promotion policy."""

from __future__ import annotations
from functools import cached_property

from contextvars import copy_context

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from aka.light.workspace.resource_catalog import resource

from aka.light.workflow.constants import (
    SUPPLEMENTAL_PENDING_PREFIX,
    AGENT_PROBLEM_GENERATION_PROMPT,
    ATREX_BENCH_HARNESS,
    ATREX_PRIVATE_REFERENCE_ENV,
    DEFAULT_CONVERT_AFTER,
    DEFAULT_HANDOFF_RESUMES,
    DEFAULT_VERIFY_REPEATS,
    DEFAULT_VERIFY_RUN_TIMEOUT,
    DEPENDENCY_REVIEW_PROMPT,
    RESEARCH_CLAIM_REVIEW_PROMPT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    FRAMEWORK_BASELINE_CATEGORY,
    FRAMEWORK_BASELINE_FILE,
    FRAMEWORK_BASELINE_TIMEOUT_S,
    FRAMEWORK_BASELINE_VERSION,
    IMMUTABLE_BASELINE_PATHS,
    DEFAULT_PROBLEM_GENERATION_TIMEOUT_S,
    PROFILE_DRIVER,
    PROMPTS_DIR,
    REPO_ROOT,
    SOL_SEED,
    TEST_RESULT_PREFIX,
    WORKSPACE_INIT,
)
from aka.light.execution.recovery import (
    ENVIRONMENT_TEMPFAIL,
    EnvironmentUnavailable,
    environment_state_file,
)
from aka.light.workflow.baseline_progress import (capture_unexpected_exit as capture_framework_baseline_exit)
from aka.light.workflow.baseline_progress import (load_progress as load_framework_baseline_progress)
from aka.light.workflow.baseline_progress import (mark_accepted as mark_framework_baseline_accepted)
from aka.light.workflow.baseline_progress import (progress_path as framework_baseline_progress_path)
from aka.light.workflow.baseline_progress import (restore_latest_candidate as restore_latest_framework_baseline_candidate)
from aka.light.evaluation.hardware import hardware_directive, hardware_vendor, kernel_is_gluon
from aka.light.evaluation.policy import BENCH_ITERS, CORRECTNESS_CASES, SHAPE_BATCH_SIZE
from aka.light.agent.dependency_policy import DEPENDENCY_GUARD_ENV
from aka.contracts.evaluation import DEFAULT_EVALUATION_STEP_TIMEOUT
from aka.light.workspace.versions import (
    objective_score_report,
    save_version_record,
    version_commit_paths,
    version_record_path,
)
from aka.light.workspace.baseline import (
    baseline_failure_path,
    read_baseline_record,
    write_baseline_failure,
    write_baseline_record,
)
from aka.legacy.task.problem.layout import (
    AGENT_PROBLEM_FILENAME,
    agent_visible_operator_files,
    has_agent_problem,
    is_sol_op,
    validate_agent_problem,
    validate_generated_agent_problem,
    validate_private_shapes,
)
from aka.light.acceptance.source_policy import (
    install_workspace_policy,
    optimization_mode_directive,
    production_kernel_violations,
    production_structure_violations,
    workspace_policy_block,
)
from aka.light.workflow.session_io import (
    SessionResult,
    _production_review_candidate_paths,
    _production_review_digest,
    _local_test_record,
    _record_local_test_result,
    _render,
    _sandbox_command,
    _test_result_from_stdout,
    _validate_production_review,
    run_session,
    sandbox_directive,
)
from aka.light.workflow.models import EpisodeHandoff
from aka.light.workspace.assets import (_agent_runtime_directive, link_runtime)
from aka.light.workspace.state import (
    git_head,
    git_path_blob,
    git_worktree_blob,
    head_kernel_is_initial_baseline,
    latest_version,
    read_memory,
    resolve_framework_baseline_commit,
    v0_baseline_commit,
    write_stall,
)


_WIKI_PROFILE_ROOT_ENV = "ATREX_WIKI_PROFILE_ROOT"
_WIKI_TASK_ID_ENV = "ATREX_WIKI_TASK_ID"

_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION = 5
_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_PATH = Path(
    ".atrex_long_horizon/framework_baseline/correctness_guidance.json"
)
_FRAMEWORK_BASELINE_SESSION_PATH = Path(
    ".atrex_long_horizon/framework_baseline/session.json"
)
_FRAMEWORK_BASELINE_EPISODE = 0
_FRAMEWORK_BASELINE_BRANCH = "framework-baseline"
_FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS = (
    "README.md",
    "reference.py",
    "input.py",
    "agent_problem.json",
    "shapes.json",
    "metadata.json",
    "definition.json",
    "workload.jsonl",
)
_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS = (
    "SEMANTIC_CHECKLIST:",
    "EDGE_CASES:",
    "TARGET_IMPLEMENTATION_RISKS:",
    "PRE_VALIDATION_CHECKS:",
    "RECOMMENDED_CORRECTNESS_FIRST_DESIGN:",
)


def _framework_baseline_failure_problem(result: SessionResult) -> str:
    """Prefer the Provider failure over the missing-handoff consequence."""
    if result.timed_out:
        detail = result.stderr_tail.strip() or result.stdout_tail.strip()
        suffix = f": {detail[-4000:]}" if detail else ""
        return f"Agent session timed out before publishing the Episode 0 handoff{suffix}"
    if result.exit_status != 0:
        detail = result.stderr_tail.strip() or result.stdout_tail.strip()
        suffix = f": {detail[-4000:]}" if detail else ""
        return (
            "Agent session failed before publishing the Episode 0 handoff "
            f"(exit={result.exit_status}){suffix}"
        )
    return (
        result.completion_diagnosis
        or "Framework Baseline Episode 0 ended without an accepted candidate_ready report"
    )


def framework_baseline_workspace_contract(workspace: Path) -> bytes:
    """Render the real SourceProvider boundary used by the pre-Episode V1 stage."""
    from aka.light.workspace.sources import instructions, policy_at

    source_contract = instructions(policy_at(workspace))
    return (
        "# Framework Baseline workspace contract\n\n"
        "This file is Supervisor-owned; do not edit it. The SourceProvider section below "
        "defines the complete candidate and its editable paths. Framework Baseline uses the same "
        "Runtime measurement operations and clean-Git candidate identity as an optimization "
        "Episode; its goal is correctness and compliance rather than speedup.\n"
        + source_contract
        + "\nOnly the SourceProvider-declared candidate paths are editable in this stage. "
        "README.md, CLAUDE.md, this file, evaluator/public-contract inputs, profile_driver.py "
        "and memory/ are Supervisor-owned. Put temporary implementation probes under scratch/. "
        "Use local Git for candidate checkpoints and leave it clean before Kernel-bound Runtime "
        "requests. Finish Episode 0 through the ordinary episode-report contract after a "
        "passing standard full Evaluate. The Supervisor runs the ordinary final-candidate "
        "gates without ABBA, records the accepted candidate in Research Store, and projects "
        "canonical state.\n"
    ).encode("utf-8")


def _framework_baseline_validation_failure(
    result: dict, *, returncode: int
) -> str:
    """Render complete redacted evaluator evidence for candidate repair."""
    from aka.light.evaluation.projection import (
        evaluation as project_evaluation,
        redact_diagnostics,
    )

    projected = project_evaluation(result)
    evidence = {
        key: projected[key]
        for key in (
            "correctness",
            "failures",
            "actionable_diagnostics",
            "diagnostics",
            "feedback_truncation",
        )
        if projected.get(key)
    }
    detail = json.dumps(
        redact_diagnostics(evidence), ensure_ascii=False, sort_keys=True
    )
    exit_note = f"; evaluator command exit={returncode}" if returncode else ""
    message = f"combined five-case correctness validation failed{exit_note}"
    if detail and detail != "{}":
        message += f". Evaluator diagnostics: {detail}"
    return message + ". Repair the reported shapes/checks, then rerun validation."


def _runtime_timeout_from_environment(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"{name} must be a finite positive number of seconds; "
            f"set a value greater than 0 or unset {name} to use the default"
        ) from None
    return value


def _git_command_error(action: str, result: subprocess.CompletedProcess[str]) -> RuntimeError:
    diagnostic = (result.stderr or result.stdout or "").strip()
    if not diagnostic:
        diagnostic = f"git exited with status {result.returncode}"
    return RuntimeError(f"{action}: {diagnostic}")


def _stage_framework_source_tree(workspace: Path, source_paths: list[str]) -> list[str]:
    """Stage existing or tracked Kernel paths and report Git pathspec failures."""
    stageable: list[str] = []
    for path in dict.fromkeys(source_paths):
        candidate = workspace / path
        if candidate.exists() or candidate.is_symlink():
            stageable.append(path)
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "--", path],
            cwd=str(workspace),
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise _git_command_error(
                f"failed to inspect framework baseline source path {path!r}", tracked
            )
        if tracked.stdout.strip():
            stageable.append(path)
    if not stageable:
        configured = ", ".join(repr(path) for path in dict.fromkeys(source_paths))
        raise RuntimeError(
            "cannot stage framework baseline source tree: none of the configured "
            f"paths exists or is tracked ({configured})"
        )
    staged = subprocess.run(
        ["git", "add", "-A", "--", *stageable],
        cwd=str(workspace),
        check=False,
        capture_output=True,
        text=True,
    )
    if staged.returncode != 0:
        raise _git_command_error("failed to stage framework baseline source tree", staged)
    return stageable


def framework_baseline_validation_command(n: int, *, atrex_bench: bool) -> list[str]:
    """Use the same Runtime-owned sampling policy as an ordinary Episode."""
    command = ["python", "test_kernel.py", "--version", f"v{n}"]
    if atrex_bench:
        command += ["--timed-runs", str(BENCH_ITERS)]
    else:
        command += ["--multi-seed", str(CORRECTNESS_CASES - 1)]
    command.append("--no-memory")
    return command


@dataclass
class Campaign:
    name: str
    kernel_demo: str
    platform: str
    framework: str
    specify_dsl: bool = True
    expose_evaluator_shapes: bool = True
    notes: str = "none"
    arch: str = ""  # real runtime GPU arch e.g. "sm_103" / "gfx942"; auto-detected
    work_dir: str = ""  # explicit working directory; "" = Path.cwd() (backward compat)
    workspace_suffix: str = ""  # internal auto-dispatch suffix, e.g. triton_h20
    max_iters: int = 20
    token_budget: int = 0  # 0 = no token cap (max-iters still bounds the run)
    target_util: float = 90.0
    problem_generation_timeout: int = DEFAULT_PROBLEM_GENERATION_TIMEOUT_S  # per authoring attempt
    max_stall: int = 0  # 0 = disabled; >0 = stop after N unpromoted episodes
    convert_after: int = (
        DEFAULT_CONVERT_AFTER  # triton-only: mandatory Gluon conversion threshold
    )
    sandbox_hardware: str = (
        ""  # agate scheduler token, e.g. REMOTE_GPU (may differ from platform)
    )
    sandbox_profile: str = ""  # pre/prod; empty preserves normal agate URL resolution
    sandbox_url: str = ""  # explicit endpoint, e.g. http://127.0.0.1:8000
    sandbox_timeout: int | None = None
    atrex_bench_root: str = ""  # native evaluator checkout owning run_eval.py
    agent_cli: str = "claude"  # episode backend: claude, qodercli, codex, or pi
    agent_sandbox: str = "none"  # opt-in coordinator isolation, independent of GPU transport
    bwrap_executable: str = "bwrap"
    agent_read_only_paths: tuple[str, ...] = ()
    agent_reference_paths: tuple[tuple[str, str], ...] = ()
    optimization_mode: str = (
        "leaderboard"  # permissive contest flow or strict production gate
    )
    expose_evaluator_shapes: bool = True
    framework_baseline: str = (
        "auto"  # bring up a target-framework V1 when the measured V0 is a wrapper
    )
    framework_baseline_timeout: int = FRAMEWORK_BASELINE_TIMEOUT_S
    handoff_resumes: int = DEFAULT_HANDOFF_RESUMES
    verify_repeats: int = DEFAULT_VERIFY_REPEATS
    # One Runtime-owned per-Shape budget shared by Eval and ABBA. The legacy
    # ``verify_run_timeout`` accessor below remains a structural compatibility
    # alias for upstream v0.4 call sites.
    evaluation_step_timeout: int = DEFAULT_EVALUATION_STEP_TIMEOUT
    shape_batch_size: int = SHAPE_BATCH_SIZE
    gpu_clock_mhz: int | None = None
    min_improvement_pct: float = 0.0
    v1_ask_codex: bool = False
    v1_ask_qoder: bool = False
    sandbox_ssh: str = ""  # standard OpenSSH target, e.g. user@gpu-host
    sandbox_ssh_init: str = ""  # remote environment activation command
    sandbox_ssh_gpu: int | None = None  # assigned physical NVIDIA GPU index
    sandbox_health_command: str = ""  # remote GPU readiness probe
    tokens_spent: int = field(default=0, init=False)
    _account_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _production_review_cache: dict[str, tuple[tuple[str, ...], dict[str, object]]] = (
        field(default_factory=dict, init=False, repr=False, compare=False)
    )
    _generated_agent_problem_digest: str = field(
        default="", init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.specify_dsl) is not bool:
            raise TypeError("specify_dsl must be a boolean")
        if type(self.expose_evaluator_shapes) is not bool:
            raise TypeError("expose_evaluator_shapes must be a boolean")
        if self.sandbox_timeout is None:
            from aka.contracts.evaluation import derived_evaluation_timeout
            self.sandbox_timeout = derived_evaluation_timeout(
                repeats=self.verify_repeats,
                step_timeout=self.evaluation_step_timeout,
            )
        if sum(
            bool(value)
            for value in (self.sandbox_ssh, self.sandbox_url, self.sandbox_profile)
        ) > 1:
            raise ValueError(
                "sandbox_ssh, sandbox_url, and sandbox_profile are mutually exclusive"
            )
        if self.problem_generation_timeout <= 0:
            raise ValueError("problem_generation_timeout must be positive")
        if type(self.shape_batch_size) is not int or self.shape_batch_size <= 0:
            raise ValueError("shape_batch_size must be a positive integer")
        if self.gpu_clock_mhz is not None and (
            type(self.gpu_clock_mhz) is not int or self.gpu_clock_mhz <= 0
        ):
            raise ValueError("gpu_clock_mhz must be a positive integer")
        if self.agent_reference_paths and self.agent_sandbox != "bwrap":
            raise ValueError(
                "--agent-reference-dir requires an Agent profile with sandbox=bwrap"
            )

    @property
    def campaign_name(self) -> str:
        suffix = f"_{self.workspace_suffix}" if self.workspace_suffix else ""
        return f"{self.name}{suffix}"

    @property
    def verify_run_timeout(self) -> int:
        return self.evaluation_step_timeout

    @property
    def dsl_context(self) -> str:
        if not self.specify_dsl:
            return ""
        if self.optimization_mode == "production":
            return (
                f"- Required final candidate DSL: `{self.framework}`. Development probes and "
                "temporary prototypes may use any language or DSL; they must not remain in the "
                "submitted Kernel source tree or runtime compute path unless compliant."
            )
        return f"- Recommended candidate DSL: `{self.framework}`"

    @property
    def dsl_query_context(self) -> str:
        return f", DSL `{self.framework}`" if self.specify_dsl else ""

    @property
    def private_reference_dir(self) -> Path | None:
        """Return evaluator-only native inputs behind a generalized public problem."""
        op_dir = Path(self.kernel_demo).resolve().parent
        use_generalized = (
            not self.expose_evaluator_shapes
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

        timeout = self.problem_generation_timeout
        print(
            "[orchestrator] generalized problem: production received detailed shapes only; "
            f"starting AKA problem-authoring session (timeout: {timeout}s per attempt, at most 2 attempts)",
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
                result = self._run_session(
                    staging,
                    prompt,
                    timeout=timeout,
                    agent_cli=self.agent_cli,
                    reasoning_effort="max",
                    extra_environment=self.agent_boundary_environment("problem-generation"),
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

    @property
    def shape_ids_are_opaque(self) -> bool:
        """Whether evaluator Shape parameters are hidden from the Agent."""
        return self.private_reference_dir is not None

    def _shape_disclosure_prompt(self) -> str:
        private_dir = self.private_reference_dir
        if private_dir is None:
            return ""
        return (
            "For this generalized native task, exact evaluator Shape parameters are private. "
            "Shape IDs in Runtime results are opaque; do not try to reconstruct hidden inputs."
        )


    def _framework_baseline_correctness_reviewers(self) -> tuple[str, ...]:
        return tuple(
            reviewer
            for reviewer, enabled in (
                ("codex", self.v1_ask_codex),
                ("qodercli", self.v1_ask_qoder),
            )
            if enabled
        )

    def agent_boundary_environment(self, role: str = "optimizer") -> dict[str, str]:
        return {
            "ATREX_AGENT_SANDBOX": self.agent_sandbox,
            "ATREX_BWRAP_EXECUTABLE": self.bwrap_executable,
            "ATREX_AGENT_READ_ONLY_PATHS": json.dumps(self.agent_read_only_paths),
            "ATREX_AGENT_REFERENCE_PATHS": json.dumps(dict(self.agent_reference_paths)),
            "ATREX_AGENT_WORKSPACE_ROLE": role,
        }

    def start_runtime(self) -> None:
        from aka.light.runtime.server import RuntimeConfig, SupervisorRuntime

        if getattr(self, "_supervisor_runtime", None) is not None:
            return
        try:
            request_timeout = _runtime_timeout_from_environment("ATREX_AKA_REQUEST_TIMEOUT_SECONDS")
            queue_timeout = _runtime_timeout_from_environment("ATREX_AKA_QUEUE_TIMEOUT_SECONDS", 60)
        except ValueError as error:
            raise SystemExit(f"orchestrator: {error}") from None
        self._supervisor_runtime = self.tool_runtime.open(RuntimeConfig(
            hardware=self.sandbox_hardware or self.platform, timeout=self.sandbox_timeout,
            evaluation_step_timeout=self.evaluation_step_timeout,
            shape_batch_size=self.shape_batch_size,
            gpu_clock_mhz=self.gpu_clock_mhz,
            url=self.sandbox_url, profile=self.sandbox_profile, ssh=self.sandbox_ssh,
            ssh_init=self.sandbox_ssh_init, ssh_gpu=self.sandbox_ssh_gpu,
            health_command=self.sandbox_health_command, private_reference_dir=self.private_reference_dir,
            atrex_bench_root=Path(self.atrex_bench_root).resolve() if self.atrex_bench_root else None,
            wiki_profile_root=self.workspace / ".gpu_wiki_profile", task_id=self.campaign_name,
            optimization_mode=self.optimization_mode,
            workspace_policy=workspace_policy_block(
                self.optimization_mode,
                self.framework,
                source_checks_enabled=self.source_checks_enabled,
                specify_dsl=self.specify_dsl,
                numerical_validation_enabled=self.module_options.numerical_validation,
            ),
            workspace=self.workspace,
            request_timeout=request_timeout,
            queue_timeout=queue_timeout,
        ), workspace_provider=self.candidate_workspace)
        self._supervisor_runtime.production_policy_validator = self.production_policy_feedback
        self._supervisor_runtime.numerical_validator = self.supplemental_numerical_feedback
        self._supervisor_runtime.research_claim_reviewer = self.review_research_claim

    def review_research_claim(
        self,
        evidence_root: Path,
        *,
        resume_session_id: str | None = None,
    ) -> dict[str, object]:
        """Run the strict three-tier reviewer with the Qwen-only provider policy."""
        from aka.light.agent.session_tail import read_regular_bytes
        from aka.light.execution.retry import check_review_service
        from aka.light.workflow.claim_review import validate_review

        evidence_root = Path(evidence_root).resolve(strict=True)
        model = "qwen3.8-max"
        boundary_environment = self.agent_boundary_environment(
            "research-claim-review"
        )
        # Independent claim review is a security boundary, not an optimizer
        # convenience setting.  Always use Bubblewrap even when the selected
        # optimizer profile deliberately runs without a sandbox.  If bwrap is
        # unavailable, launch fails closed and the caller archives an
        # unreviewed receipt instead of executing the reviewer natively.
        boundary_environment["ATREX_AGENT_SANDBOX"] = "bwrap"
        base_environment = {
            **boundary_environment,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        }
        from aka.light.agent.home import SESSION_HOME_ROOT_ENV
        from aka.light.runtime.paths import supervisor_campaign_root

        campaign_workspace = Path(getattr(self, "workspace", evidence_root.parent))
        base_environment[SESSION_HOME_ROOT_ENV] = str(
            supervisor_campaign_root(campaign_workspace)
            / "research-review-session-homes"
        )
        capture_base = (
            evidence_root.parent / "claim-review-sessions" / evidence_root.name
        )
        session_attempts: list[dict[str, object]] = []

        def session_audit() -> dict[str, object]:
            return {
                "schema": "atrex.research-claim-review-session.v1",
                "attempts": [dict(row) for row in session_attempts],
            }

        last_error: Exception | None = None
        for attempt in range(1, 3):
            response = evidence_root / "review.json"
            response.unlink(missing_ok=True)
            attempt_capture = capture_base / f"attempt-{attempt}"
            environment = {
                **base_environment,
                "ATREX_SESSION_CAPTURE_DIR": str(attempt_capture),
            }
            resumed = attempt == 1 and bool(resume_session_id)
            try:
                result = self._run_session(
                    evidence_root,
                    RESEARCH_CLAIM_REVIEW_PROMPT.read_text(encoding="utf-8"),
                    # Claim review is part of the durable research record, not
                    # an optimization Episode or an acceptance-side advisory.
                    # Let it finish without an active-time deadline.  The
                    # Provider/session may still fail explicitly, which is
                    # captured below for audit and retry.
                    timeout=None,
                    agent_cli="claude",
                    reasoning_effort="high",
                    agent_plugins=False,
                    extra_environment=environment,
                    session_id=resume_session_id if resumed else None,
                    resume_session=resumed,
                )
            except Exception as error:  # noqa: BLE001 - audit every launch-boundary failure
                # A launch-boundary failure has no Provider result or capture to
                # inspect. Preserve it as a Supervisor-private attempt so the
                # public receipt can stay bounded without erasing the cause.
                last_error = error
                session_attempts.append({
                    "attempt": attempt,
                    "session_id": resume_session_id if resumed else "",
                    "status": "capture_missing",
                    "exit_status": None,
                    "timed_out": False,
                    "conversation_path": None,
                    "resumed": resumed,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error)[:2000],
                    },
                })
                continue
            self._account(result, f"Qwen research-claim review attempt {attempt}")
            conversations = sorted(
                (
                    path
                    for path in attempt_capture.glob("run-*/conversation.jsonl")
                    if path.is_file()
                ),
                key=lambda path: path.stat().st_mtime_ns,
            )
            conversation_path = (
                conversations[-1].relative_to(evidence_root.parent).as_posix()
                if conversations else None
            )
            attempt_audit: dict[str, object] = {
                "attempt": attempt,
                "session_id": result.session_id,
                "status": "capture_missing" if conversation_path is None else "invalid",
                "exit_status": result.exit_status,
                "timed_out": result.timed_out,
                "conversation_path": conversation_path,
                "resumed": resumed,
            }
            session_attempts.append(attempt_audit)
            try:
                value = json.loads(read_regular_bytes(response, limit=128 * 1024 + 1))
                if response.stat().st_size > 128 * 1024:
                    raise ValueError("research claim review exceeds 128 KiB")
                review = validate_review(value)
                if result.timed_out:
                    print(
                        "[research-review] recovered a valid review from a timed-out reviewer",
                        flush=True,
                    )
                else:
                    check_review_service(result)
                if conversation_path is not None:
                    attempt_audit["status"] = "accepted"
                return {
                    **review,
                    "_review_session_audit": session_audit(),
                }
            except (OSError, ValueError, TypeError, KeyError) as error:
                last_error = error
                attempt_audit["error"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                }
                if not result.timed_out:
                    try:
                        check_review_service(result)
                    except ValueError as service_error:
                        last_error = service_error
                        attempt_audit["error"] = {
                            "type": type(service_error).__name__,
                            "message": str(service_error)[:2000],
                        }
        failure = RuntimeError(
            "Qwen research-claim reviewer produced no valid strict review"
        )
        failure.review_session_audit = session_audit()
        raise failure from last_error

    def production_policy_feedback(
        self,
        workspace: Path,
        source: bytes,
        *,
        parent=None,
        require_gluon: bool = False,
    ) -> list[str]:
        """Review the exact sealed report candidate before accepting its handoff."""
        if self.optimization_mode != "production" or not self.source_checks_enabled:
            return []
        from aka.light.workspace.sources import install

        with tempfile.TemporaryDirectory(prefix="atrex-production-candidate-") as directory:
            frozen = Path(directory)
            install(frozen, source)
            return self._production_kernel_violations(
                frozen,
                require_gluon=require_gluon,
            )

    def supplemental_numerical_feedback(
        self,
        workspace: Path,
        source: bytes | None = None,
        *,
        parent=None,
        flow: str = "episode",
        standard_correctness_passed=None,
    ) -> str:
        """Validate frozen candidate bytes using private plans and recorded GPU probes."""
        from aka.light.acceptance.numerical import validate_candidate

        if not self.module_options.numerical_validation:
            return ""
        self.start_runtime()
        return validate_candidate(
            self,
            workspace,
            source=source,
            parent=parent,
            flow=flow,
            standard_correctness_passed=standard_correctness_passed,
        )

    @property
    def plugin_registry(self):
        self.start_runtime()
        return self._supervisor_runtime.plugins

    def plugin_directive(self, phase: str) -> str:
        return self.plugin_registry.instructions(
            phase, PLATFORM=self.platform, ARCH=self.arch or "<exact runtime architecture>",
            FRAMEWORK=(self.framework if self.specify_dsl else "Agent-selected supported GPU DSL"),
            OPERATOR=self.name,
        )

    def wiki_prompt(self, slot: str) -> str | None:
        """The selected Wiki provider's replacement for a built-in Wiki prompt fragment."""
        return self.plugin_registry.wiki_prompt(slot) if self.tool_runtime.wiki else None

    def measure_for_acceptance(self, workspace: Path, argv: list[str], *, execution_timeout: int | None = None,
                               parent=None, reference_snapshot: Path | None = None,
                               terminal_validation: bool = False,
                               kernel_ids: dict[str, str] | None = None,
                               journal_workspace: Path | None = None) -> dict:
        self.start_runtime()
        return self._supervisor_runtime.execute_trusted(workspace, argv, execution_timeout=execution_timeout,
                                                       parent=parent, reference_snapshot=reference_snapshot,
                                                       terminal_validation=terminal_validation,
                                                       kernel_ids=kernel_ids,
                                                       journal_workspace=journal_workspace)

    def recorded_kernel(self, kernel_id: str) -> bytes:
        self.start_runtime()
        return self._supervisor_runtime.measurements.read_kernel(kernel_id)

    def republish_episode_report(self, episode: int, worktree: Path, *, base_commit: str,
                                 branch: str) -> bool:
        """Recover a finalized private report whose compatibility handoff was lost."""
        from aka.light.workflow.journal_service import SupervisorJournalService

        self.start_runtime()
        runtime = self._supervisor_runtime
        path = runtime.episode_journal_path(episode)
        if not path.is_file():
            return False
        private = runtime.read_episode_journal(episode)
        if not private.get("finalized_at"):
            return False
        if (private.get("episode") != episode or private.get("base_commit") != base_commit
                or private.get("episode_branch") != branch
                or private.get("state") not in {"candidate_ready", "pivot", "blocked"}):
            raise RuntimeError("Finalized private report does not match the recovering Episode")
        service = runtime.journal_factory.open(
            workspace=worktree, git_workspace=worktree,
            campaign_root=runtime.journals_root, evidence_root=path.parent,
            supervisor_git=True,
        )
        service._publish_report(private)
        return True

    def acceptance_measurement_context(
        self, agent_workspace: Path, incumbent_source: bytes
    ) -> dict:
        """Return the exact Agent-visible incumbent identity and frozen policy."""
        self.start_runtime()
        runtime = self._supervisor_runtime
        baseline_kernel_id = runtime.baseline_kernel_commit(agent_workspace)
        # Bind the source under the Agent-visible commit. The controller commit
        # may identify identical bytes in a different, private Git history and
        # must never leak into an Agent ABBA request.
        runtime.measurements.kernel(incumbent_source, baseline_kernel_id)
        return {
            "baseline_kernel_id": baseline_kernel_id,
            "measurement_repetitions": runtime.measurement_repetitions,
        }

    def acceptance_verifier(self, **options):
        self.start_runtime()
        return self._supervisor_runtime.acceptance.verifier(**options)

    def selected_episode_evaluation(self, episode: int, source: bytes) -> dict:
        from aka.light.workflow.journal_service import selected_evaluation
        self.start_runtime()
        path = self._supervisor_runtime.episode_journal_path(episode)
        return selected_evaluation(
            path, journals_root=self._supervisor_runtime.journals_root, source=source,
        )

    def register_runtime_episode(
        self,
        workspace: Path,
        *,
        episode: int,
        base_commit: str,
        branch: str,
        memory_version: int,
        baseline_kernel_commit_id: str | None = None,
        supervisor_git: bool = True,
        require_gluon: bool = False,
        origin: dict[str, int] | None = None,
    ) -> Path:
        """Start Runtime if needed and bind an Episode from trusted controller state."""
        self.start_runtime()
        return self._supervisor_runtime.register_episode(
            workspace,
            episode=episode,
            base_commit=base_commit,
            branch=branch,
            memory_version=memory_version,
            baseline_kernel_commit_id=baseline_kernel_commit_id,
            supervisor_git=supervisor_git,
            require_gluon=require_gluon,
            origin=(origin or getattr(self, "search_coordinate", None) or {
                "epoch": episode,
                "trajectory": 1,
                "episode": episode,
            }),
        )

    def read_runtime_episode_journal(self, episode: int) -> dict:
        """Read existing private state without registering or creating an Episode."""
        self.start_runtime()
        return self._supervisor_runtime.read_episode_journal(episode)

    def close_runtime(self) -> None:
        runtime = getattr(self, "_supervisor_runtime", None)
        if runtime is not None:
            runtime.close()
            self._supervisor_runtime = None

    def agent_environment(self) -> dict[str, str]:
        private_dir = self.private_reference_dir
        # Final-candidate policy is enforced by the Supervisor. Development
        # sessions may use temporary probes without an early local hook veto.
        environment = {DEPENDENCY_GUARD_ENV: "0"}
        environment.update(self.agent_boundary_environment())
        self.start_runtime()
        from aka.light.runtime.server import OWNER_ENV
        environment[OWNER_ENV] = self._supervisor_runtime.owner_id
        if private_dir is not None:
            environment[ATREX_PRIVATE_REFERENCE_ENV] = str(private_dir)
        if self.sandbox_ssh:
            environment["ATREX_SANDBOX_SSH"] = self.sandbox_ssh
            environment["ATREX_SANDBOX_SSH_INIT"] = self.sandbox_ssh_init
            if self.sandbox_ssh_gpu is not None:
                environment["ATREX_SANDBOX_SSH_GPU"] = str(self.sandbox_ssh_gpu)
            environment.pop("ATREX_SANDBOX_URL", None)
            environment.pop("ATREX_SANDBOX_PROFILE", None)
            if self.sandbox_health_command:
                environment["ATREX_SANDBOX_HEALTH_COMMAND"] = (
                    self.sandbox_health_command
                )
        state_file = environment_state_file()
        if state_file is not None:
            environment["ATREX_ENVIRONMENT_STATE_FILE"] = str(state_file)
        # Query events from disposable episode worktrees must land in the
        # incumbent workspace, where the completion hook can retain them.
        wiki_profile_root = self.workspace / ".gpu_wiki_profile"
        # Preserve lexical paths for bwrap's mount validation; native sessions
        # retain the existing symlink and '..' canonicalization.
        environment[_WIKI_PROFILE_ROOT_ENV] = str(
            wiki_profile_root.absolute()
            if self.agent_sandbox == "bwrap"
            else wiki_profile_root.resolve()
        )
        environment[_WIKI_TASK_ID_ENV] = self.campaign_name
        return environment


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
            for name in ("shapes.json", "roofline.json", "valid.py")
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
        with self._account_lock:
            self.tokens_spent += res.tokens
            cumulative = self.tokens_spent
        print(
            f"[orchestrator] {label}: exit={res.exit_status} timed_out={res.timed_out} "
            f"tokens={res.tokens} cum_tokens={cumulative}",
            flush=True,
        )
        if res.exit_status != 0 or res.timed_out:
            print(
                f"[orchestrator] stderr tail:\n{res.stderr_tail}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _persist_production_review_record(
        workspace: Path,
        candidate_digest: str,
        review_record: dict[str, object],
    ) -> list[str]:
        record_path = (
            workspace
            / ".atrex_long_horizon"
            / "production_reviews"
            / f"{candidate_digest}.json"
        )
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = record_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(review_record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(record_path)
        except OSError as exc:
            return [
                "could not persist production policy verdict: "
                f"{type(exc).__name__}: {exc}"
            ]
        return []

    def _review_production_candidate_once(
        self,
        workspace: Path,
        framework: str,
        require_gluon: bool,
        *,
        candidate_digest: str,
        record_workspace: Path | None = None,
    ) -> list[str]:
        """Delegate complete candidate policy review to a fresh, isolated agent."""
        from aka.light.execution.retry import check_review_service

        cache_key = candidate_digest + ":" + self.agent_cli
        cached = self._production_review_cache.get(cache_key)
        if cached is not None:
            cached_errors, review_record = cached
            persistence_errors = self._persist_production_review_record(
                record_workspace or workspace,
                candidate_digest,
                review_record,
            )
            return list(dict.fromkeys([*cached_errors, *persistence_errors]))

        errors: list[str]
        review_payload: object | None = None
        review_summary = ""
        with tempfile.TemporaryDirectory(
            prefix="atrex-production-review-"
        ) as directory:
            review_workspace = Path(directory)
            candidate_root = review_workspace / "candidate"
            source_hashes: dict[str, str] = {}
            for source in _production_review_candidate_paths(workspace):
                relative = source.relative_to(workspace)
                destination = candidate_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                source_hashes[relative.as_posix()] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()

            request = {
                "schema_version": DEPENDENCY_REVIEW_SCHEMA_VERSION,
                "framework": framework if self.specify_dsl else None,
                "dsl_policy": "pinned" if self.specify_dsl else "open",
                "required_phase": (
                    "gluon"
                    if require_gluon and self.specify_dsl
                    else "selected_framework"
                    if self.specify_dsl
                    else "open_dsl"
                ),
                "optimization_mode": "production",
                "candidate_digest": candidate_digest,
                "candidate_files": sorted(source_hashes),
            }
            (review_workspace / "review_request.json").write_text(
                json.dumps(request, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            try:
                result = self._run_session(
                    review_workspace,
                    DEPENDENCY_REVIEW_PROMPT.read_text(encoding="utf-8"),
                    timeout=None,
                    agent_cli=self.agent_cli,
                    reasoning_effort="high",
                    agent_plugins=False,
                    extra_environment=self.agent_boundary_environment("production-review"),
                )
                self._account(result, "independent production policy review")
                check_review_service(result)
            except ValueError as exc:
                return [f"independent production policy review failed: {exc}"]
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
                    "independent production policy review modified candidate evidence: "
                    + ", ".join(sorted(changed))
                ]
            else:
                review_path = review_workspace / "dependency_review.json"
                try:
                    review_payload = json.loads(
                        review_path.read_text(encoding="utf-8")
                    )
                    errors, review_summary = _validate_production_review(
                        review_payload,
                        candidate_files=frozenset(source_hashes),
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    from aka.light.execution.retry import ReviewerOutputUnavailable

                    raise ReviewerOutputUnavailable(
                        "independent production policy review produced no valid verdict: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                else:
                    status = "accepted" if not errors else "rejected"
                    print(
                        f"[production-policy] independent full-candidate review {status}: "
                        f"{review_summary}",
                        flush=True,
                    )

        try:
            reviewed_digest = _production_review_digest(
                workspace, framework, require_gluon, self.specify_dsl
            )
        except OSError as exc:
            errors = [
                "production candidate changed during policy review: "
                f"{type(exc).__name__}: {exc}"
            ]
        else:
            if reviewed_digest != candidate_digest:
                errors = ["production candidate changed during policy review"]
        review_record = {
            "schema_version": DEPENDENCY_REVIEW_SCHEMA_VERSION,
            "candidate_digest": candidate_digest,
            "framework": framework if self.specify_dsl else None,
            "dsl_policy": "pinned" if self.specify_dsl else "open",
            "required_phase": (
                "gluon"
                if require_gluon and self.specify_dsl
                else "selected_framework"
                if self.specify_dsl
                else "open_dsl"
            ),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "agent_runtime": self.agent_cli,
            "accepted": not errors,
            "errors": errors,
            "summary": review_summary,
            "review": review_payload,
        }
        persistence_errors = self._persist_production_review_record(
            record_workspace or workspace,
            candidate_digest,
            review_record,
        )
        self._production_review_cache[cache_key] = (tuple(errors), review_record)
        return list(dict.fromkeys([*errors, *persistence_errors]))

    def _review_production_candidate(
        self,
        workspace: Path,
        framework: str,
        require_gluon: bool,
    ) -> list[str]:
        """Bound timeout retries across sessions and supervisor restarts."""
        from aka.light.execution.retry import retry_review
        from aka.light.workspace.sources import seed_baseline_metadata

        # Open-DSL single-file workspaces are initialized before the Agent has
        # selected a DSL. Refresh Supervisor-owned solution.json from the
        # finished candidate so the fail-closed Production review sees the
        # implementation's real language and dependencies.
        seed_baseline_metadata(
            workspace,
            framework,
            self.campaign_name,
            native_eval=bool(self.atrex_bench_root),
        )

        digest = _production_review_digest(
            workspace, framework, require_gluon, self.specify_dsl
        )
        stage = f"dependency-review:{digest}:{self.agent_cli}:unbounded"
        try:
            return retry_review(
                self.workspace, stage,
                lambda: self._review_production_candidate_once(
                    workspace, framework, require_gluon, candidate_digest=digest,
                ),
                cancel=getattr(self, "search_cancel", None),
            )
        except ValueError as exc:
            return [f"independent production policy review failed: {exc}"]

    def _production_kernel_violations(
        self,
        workspace: Path | None = None,
        *,
        require_gluon: bool = False,
    ) -> list[str]:
        if not self.source_checks_enabled:
            return []
        return production_kernel_violations(
            workspace or self.workspace,
            self.framework,
            require_gluon=require_gluon,
            specify_dsl=self.specify_dsl,
            production_reviewer=self._review_production_candidate,
        )

    @cached_property
    def source_checks_enabled(self) -> bool:
        """Freeze main's multi-GPU source policy from evaluator-owned metadata."""
        from aka.light.execution.config import evaluator_num_gpus
        if self.sandbox_ssh:
            return True
        return evaluator_num_gpus(self.problem.evaluation.directory) < 2

    @property
    def research_records(self) -> bool:
        return self.tool_runtime.research_records

    @property
    def research_modules(self) -> tuple[str, ...]:
        """Return the one frozen research-module selection for every layer."""
        return self.tool_runtime.tool_modules

    def _link_runtime(self) -> None:
        from aka.light.workflow.store import CampaignStore

        self._assert_generalized_inputs_are_private()
        CampaignStore.ensure_excluded(self.workspace, wiki_trace_only=True)
        native_root = Path(self.atrex_bench_root) if self.atrex_bench_root else None
        self.candidate_workspace.install_runtime(
            self.workspace,
            native_root,
            plugin_registry=self.plugin_registry,
            research_records=self.research_records,
            research_modules=self.research_modules,
            is_ppu=hardware_vendor(self.platform, self.arch) == "ppu",
        )
        install_workspace_policy(
            self.workspace,
            self.optimization_mode,
            self.framework,
            agent_runtime=self.agent_cli,
            source_checks_enabled=self.source_checks_enabled,
            specify_dsl=self.specify_dsl,
            numerical_validation_enabled=self.module_options.numerical_validation,
            expose_evaluator_shapes=self.expose_evaluator_shapes,
        )

    def _evaluator_directive(self) -> str:
        public = (
            "Use agent_problem.json for the public domain. "
            + self._shape_disclosure_prompt()
            + " "
            if self.private_reference_dir else "The visible operator contract defines the supported inputs. "
        )
        if self.atrex_bench_root:
            profile_shapes = (
                "For generalized tasks, Shape IDs are opaque and exact parameters remain private."
                if self.private_reference_dir
                else "For ordinary public-shape tasks, read exact parameters from shapes.json."
            )
            return (
                "## Evaluation contract: Atrex-Bench\n\n" + public +
                "The Supervisor supplies the official evaluator. Minimize latency_us_geomean; "
                "performance_score is a reference-relative diagnostic, not the promotion criterion. "
                "Use --kind run --mode full --no-sync for the complete workload. "
                "Correctness-only, custom-input or subset results do not satisfy an Episode report. "
                "For diagnostic Profile, use --kind profile --profile-shape-id ID --no-sync "
                "with a Shape ID from a saved result. " + profile_shapes + " The same option selects the "
                "privately supplied case in Typed Profile or a compatible Dev fallback. "
                "Never substitute a custom timing/correctness harness for official evaluation."
            )
        if is_sol_op(Path(self.kernel_demo).resolve().parent):
            return (
                "## Evaluation contract: SOL-ExecBench\n\n"
                "The Supervisor runs the official evaluator over workload.jsonl. "
                "Minimize GeoMean workload latency. performance_score relative to "
                "the SOL reference is diagnostic only. Use --kind run --no-sync; never replace the evaluator. "
                "Atrex-Bench typed --mode controls do not apply to SOL-ExecBench."
            )
        raise ValueError("A canonical Atrex-Bench or SOL evaluator is required")

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
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_ssh,
        )


    def _mode_directive(self) -> str:
        return optimization_mode_directive(self.optimization_mode, self.framework,
                                           source_checks_enabled=self.source_checks_enabled,
                                           specify_dsl=self.specify_dsl,
                                           numerical_validation_enabled=self.module_options.numerical_validation)

    def setup_baseline(self) -> None:
        # SOL-ExecBench op: seed a correct, directly-submittable V0 mechanically
        # (no baseline session) — sol_seed.py copies the ground-truth files, writes
        # the DPS wrapper kernel.py + solution.json; this method benches V0 in the sandbox.
        op_dir = Path(self.kernel_demo).resolve().parent
        if is_sol_op(op_dir):
            self._setup_baseline_sol(op_dir)
            return
        if not self.atrex_bench_root:
            raise ValueError("Deterministic V0 requires a native Atrex-Bench or SOL operator; "
                             "provide the canonical evaluator checkout instead of a custom setup script")
        if not WORKSPACE_INIT.exists():
            raise FileNotFoundError(f"missing {WORKSPACE_INIT}")
        # workspace_init.sh builds the workspace as $(pwd)/kernel_opt_<name>,
        # so cwd must be the work_dir (or the process cwd when --workspace is absent).
        subprocess.run(
            ["bash", str(WORKSPACE_INIT), self.campaign_name, self.kernel_demo],
            cwd=str(self.workspace.parent),
            check=True,
        )
        # Production native tasks expose the detailed contract by default. The explicit hidden
        # shape setting arrives as a generalized Problem while retaining production candidate
        # and promotion policy.
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
        # A native Atrex-Bench V0 is already materialized as the verbatim reference
        # wrapper.  Its evaluator and memory schemas are also supervisor-owned, so an
        # Agent session adds no implementation value here.  Commit the sources, run the
        # required remote measurement, and record the result mechanically.
        self._setup_baseline_native(op_dir, generalized=generalized)

    def _native_v0_readme(self, *, generalized: bool) -> str:
        contract = (
            "`agent_problem.json` is the authoritative public contract. "
            + self._shape_disclosure_prompt()
            if generalized
            else "`shapes.json` and the other copied operator files define the public workload."
        )
        runtime_arch = self.arch or "unknown (use the runtime GPU API before choosing codegen)"
        notes = self.notes.strip() or "none"
        hardware_evidence = (
            (self.wiki_prompt("hardware_evidence") or
             "Before an optimization plan uses a hardware limit, source it through "
             "`--kind wiki-query` and cite the returned record.") + " "
            if self.tool_runtime.wiki else
            "Wiki queries are disabled. Use supplied, attributable hardware evidence; "
            "do not guess missing peak specifications. "
        )
        dsl_goal = (
            f"Final submitted Kernels must use `{self.framework}`; development probes may use "
            "other languages or DSLs."
            if self.specify_dsl
            else "The Agent may choose any evaluator-supported GPU implementation DSL."
        )
        framework_line = self.framework if self.specify_dsl else "Agent-selected supported GPU DSL"
        return (
            f"# kernel_opt_{self.campaign_name}\n\n"
            "Evidence-backed optimization of one native Atrex-Bench operator.\n\n"
            "## Goal\n\n"
            "Minimize full-workload `latency_us_geomean` while every evaluator case remains "
            "correct. Per-Shape latency and `performance_score` are diagnostic, not the promotion "
            f"objective. V0 is supplied by the selected SourceProvider. {dsl_goal}\n\n"
            "## Campaign\n\n"
            f"- Target platform: `{self.platform}`\n"
            f"- Runtime architecture: `{runtime_arch}`\n"
            f"- Optimization mode: `{self.optimization_mode}`\n"
            f"- Target framework: `{framework_line}`\n"
            f"- Target utilization stop condition: `{self.target_util:g}%`\n"
            f"- Additional notes: {notes}\n\n"
            "## Contract and evaluator\n\n"
            f"- {contract}\n"
            "- Use `python3 tools/sandbox.py --kind run --mode full --no-sync`; the Supervisor "
            "supplies the official evaluator and private inputs.\n"
            "- V0 uses exactly one full-workload base-seed evaluator run. It does not profile, "
            "run multi-seed correctness, or perform ABBA.\n"
            "- Ground-truth operator files and canonical memory are immutable to the Agent.\n\n"
            "## Hardware evidence policy\n\n"
            "V0 records identity only and does not speculate about peak specifications. "
            f"{hardware_evidence}The runtime architecture API is authoritative when a device "
            "name or vendor SMI is desensitized.\n\n"
            "## Stop conditions\n\n"
            "The supervisor stops at the configured iteration/token/stall limits, when the target "
            "utilization is reached, or on a terminal repeated blocker. Correctness is mandatory "
            "for every promoted version.\n"
        )

    @staticmethod
    def _print_v0_evaluator_output(test: subprocess.CompletedProcess[str]) -> None:
        """Log one bounded evaluator summary without duplicating its full Record."""
        from aka.light.evaluation.projection import (
            GATEWAY_ERROR_PREFIX,
            SOURCE_ERROR_PREFIX,
        )

        def bounded_text(value: object, limit: int = 1000) -> str:
            raw = " ".join(str(value or "").split()).encode("utf-8")
            if len(raw) <= limit:
                return raw.decode("utf-8", errors="replace")
            marker = b" ... [omitted] ... "
            remaining = max(0, limit - len(marker))
            left = remaining // 2
            right = remaining - left
            return (raw[:left] + marker + raw[-right:]).decode(
                "utf-8", errors="replace"
            )

        def structured_payload(prefixes: tuple[str, ...]) -> tuple[str, dict] | None:
            for line in reversed((test.stdout or "").splitlines()):
                for prefix in prefixes:
                    if not line.startswith(prefix):
                        continue
                    try:
                        payload = json.loads(line[len(prefix):])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        return prefix, payload
            return None

        structured = structured_payload(
            (TEST_RESULT_PREFIX, GATEWAY_ERROR_PREFIX, SOURCE_ERROR_PREFIX)
        )
        summary: dict[str, object] = {"exit_code": test.returncode}
        if structured is not None:
            prefix, payload = structured
            if prefix == TEST_RESULT_PREFIX:
                all_pass = payload.get("all_pass")
                summary["status"] = (
                    "passed" if test.returncode == 0 and all_pass is True else "failed"
                )
                correctness = payload.get("correctness")
                if isinstance(correctness, dict) and isinstance(
                    correctness.get("status"), str
                ):
                    summary["correctness"] = correctness["status"]
                elif isinstance(all_pass, bool):
                    summary["correctness"] = "PASS" if all_pass else "FAIL"
                for key in (
                    "latency_us_geomean",
                    "latency_us_arith_mean",
                    "performance_score",
                ):
                    value = payload.get(key)
                    if (
                        type(value) in {int, float}
                        and math.isfinite(value)
                    ):
                        summary[key] = value
                measured = payload.get("measured_shape_count")
                if type(measured) is not int:
                    latencies = payload.get("latency_us_by_shape")
                    measured = len(latencies) if isinstance(latencies, dict) else None
                if type(measured) is int:
                    summary["measured_shape_count"] = measured
                for key in ("gateway_record_id", "eval_id"):
                    if isinstance(payload.get(key), str):
                        summary[key] = bounded_text(payload[key], 256)
                failures = payload.get("failures")
                if isinstance(failures, list) and failures:
                    summary["failure_count"] = len(failures)
                    summary["error_summary"] = [
                        bounded_text(item) for item in failures[:2]
                    ]
                diagnostics = payload.get("actionable_diagnostics")
                if isinstance(diagnostics, list) and diagnostics:
                    summary["actionable_diagnostic_count"] = len(diagnostics)
                    messages = [
                        bounded_text(item.get("message"))
                        for item in diagnostics[:2]
                        if isinstance(item, dict) and item.get("message")
                    ]
                    if messages:
                        summary["actionable_diagnostics"] = messages
            else:
                error = payload.get("error")
                error = error if isinstance(error, dict) else payload
                summary.update({
                    "status": str(payload.get("status") or "failed"),
                    "operation": str(payload.get("operation") or "evaluate"),
                })
                for key in ("gateway_record_id", "error_class", "reason"):
                    value = payload.get(key) if key == "gateway_record_id" else error.get(key)
                    if isinstance(value, str):
                        summary[key] = bounded_text(value, 256)
                message = error.get("message")
                if message is not None:
                    summary["error_summary"] = bounded_text(message)
        else:
            summary["status"] = "failed" if test.returncode else "completed"
            detail = test.stderr or test.stdout
            if detail:
                summary["error_summary"] = bounded_text(detail)

        print(
            "[orchestrator] evaluator summary: "
            + json.dumps(summary, ensure_ascii=False, sort_keys=True),
            file=sys.stderr if test.returncode else sys.stdout,
            flush=True,
        )

    @property
    def generate_memory(self) -> bool:
        return bool(getattr(getattr(self, "module_options", None), "generate_memory", False))

    def _run_v0_evaluator(self) -> dict:
        """Run the sole V0 evaluator and persist Supervisor-private facts."""
        baseline_failure_path(self.workspace).unlink(missing_ok=True)
        test = _sandbox_command(
            self.workspace,
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_timeout,
            ["python", "test_kernel.py", "--version", "v0", "--no-memory"],
            ssh=self.sandbox_ssh,
            ssh_init=self.sandbox_ssh_init,
            health_command=self.sandbox_health_command,
            gateway_kind="run",
            private_reference_dir=self.private_reference_dir,
            gpu_clock_mhz=self.gpu_clock_mhz,
            # V0 is mandatory campaign evidence.  Keep polling/retrying through
            # infrastructure failures exactly like the later terminal candidate
            # gates; a completed incorrect V0 still returns normally and is
            # rejected by the correctness check below.
            terminal_validation=True,
        )
        self._print_v0_evaluator_output(test)
        if test.returncode == ENVIRONMENT_TEMPFAIL:
            raise EnvironmentUnavailable("V0 evaluation infrastructure is unavailable; retain its job evidence for recovery")
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"sandbox V0 baseline produced no usable result: {exc}"
            ) from exc
        record = _local_test_record(
            self.workspace,
            "v0",
            result,
            existing=read_baseline_record(self.workspace) or {},
        )
        # A sharded Evaluate can submit several backend ev_* jobs.  They are
        # implementation details of one logical Gateway Record: retain them for
        # Supervisor audit/recovery, but never project them into PROGRESS.md.
        backend_job_ids = list(
            dict.fromkeys(
                re.findall(r"\bev_[A-Za-z0-9_.-]+\b", test.stdout + "\n" + test.stderr)
            )
        )
        if backend_job_ids:
            record["backend_job_ids"] = backend_job_ids
        if test.returncode != 0 or not result.get("all_pass"):
            # Keep diagnostics without making latest_version() treat a failed or
            # interrupted baseline as canonical. A rerun will retry V0 directly.
            write_baseline_failure(self.workspace, record)
            raise RuntimeError(
                "sandbox V0 baseline failed correctness/performance validation"
            )
        write_baseline_record(self.workspace, record)
        return result

    def _run_v0_evaluator_with_correctness_prefetch(self) -> dict:
        """Measure V0 while prefetching the source-only V1 correctness review.

        The review packet deliberately excludes V0 measurement artifacts, so its
        digest is stable before and after the evaluator writes private V0 facts. Review failure remains non-fatal here: the ordinary V1
        entry point validates the cache and retries synchronously when necessary.
        """
        if not self._framework_baseline_correctness_reviewers():
            return self._run_v0_evaluator()
        print(
            "[orchestrator] V0 evaluator: prefetching V1 correctness review in parallel",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            review_future = executor.submit(
                    copy_context().run,
                self._ensure_framework_baseline_correctness_guidance
            )
            try:
                return self._run_v0_evaluator()
            finally:
                try:
                    review_future.result()
                except Exception as exc:
                    print(
                        "[orchestrator] WARNING: V1 correctness review prefetch failed; "
                        "the V1 stage will retry it: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )


    def _finalize_v0_measurement(
        self,
        result: dict,
        source_commit: str,
    ) -> None:
        """Seal V0 facts privately while keeping HEAD at the real source commit."""
        memory = read_baseline_record(self.workspace)
        if not isinstance(memory, dict):
            raise RuntimeError("V0 evaluator produced no private measurement record")
        memory = dict(memory)
        memory["git_commit_hash"] = source_commit
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = "baseline"
        optimization["action_description"] = (
            "selected initial implementation measured by the official full-workload evaluator"
        )
        optimization["expected_impact"] = "correctness and latency reference for later versions"
        optimization["risks_and_rollback"] = "none; immutable V0 reference baseline"
        memory = objective_score_report(self.workspace, 0, memory)
        write_baseline_record(self.workspace, memory)
        coverage_problem = self._generalized_memory_coverage_problem(memory)
        if coverage_problem:
            raise RuntimeError(f"invalid native V0 measurement: {coverage_problem}")
        recorded = read_memory(self.workspace, 0) or {}
        if recorded.get("git_commit_hash") != source_commit:
            raise RuntimeError("V0 measurement record does not point to the V0 source commit")
        from aka.light.workspace.sources import from_git, snapshot as source_snapshot
        if from_git(self.workspace, source_commit) != source_snapshot(self.workspace):
            raise RuntimeError("V0 candidate source differs from its recorded source commit")
        if git_head(self.workspace) != source_commit:
            raise RuntimeError("V0 measurement must not advance Git beyond its source commit")

    def _setup_baseline_native(self, op_dir: Path, *, generalized: bool) -> None:
        """Seed native Atrex-Bench V0 without launching a coding Agent."""
        kernel_path = self.workspace / "kernel.py"
        reference_path = self.workspace / "reference.py"
        if not kernel_path.is_file() or not reference_path.is_file():
            raise RuntimeError("native V0 requires workspace kernel.py and reference.py")
        selected = getattr(self, "_selected_initial_kernel", reference_path.read_bytes())
        if kernel_path.read_bytes() != selected:
            raise RuntimeError(
                "native V0 kernel.py does not match the selected initial source; refusing "
                "mechanical baseline generation"
            )
        self._ensure_framework_baseline_workspace_contract()
        (self.workspace / "README.md").write_text(
            self._native_v0_readme(generalized=generalized), encoding="utf-8"
        )
        source_paths = [
            ".gitignore",
            "CLAUDE.md",
            "README.md",
            "WORKSPACE.md",
            "kernel.py",
            "test_kernel.py",
            "profile_driver.py",
            *agent_visible_operator_files(op_dir, generalized=generalized),
        ]
        from aka.light.workspace.sources import snapshot as source_snapshot, read
        source_paths += list(read(self.workspace).materialized_files)
        source_paths = list(
            dict.fromkeys(
                path for path in source_paths if (self.workspace / path).is_file()
            )
        )
        subprocess.run(
            ["git", "add", *source_paths], cwd=str(self.workspace), check=True
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
                ["git", "commit", "-m", "V0: baseline kernel"],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        source_commit = v0_baseline_commit(self.workspace)
        if not source_commit:
            raise RuntimeError("native V0 has no committed kernel.py")
        from aka.light.workspace.sources import from_git
        if from_git(self.workspace, source_commit) != source_snapshot(self.workspace):
            raise RuntimeError("native V0 source history does not match the selected source")

        result = self._run_v0_evaluator_with_correctness_prefetch()
        self._finalize_v0_measurement(result, source_commit)
        self._assert_generalized_inputs_are_private()
        contract_problem = self._generalized_contract_commit_problem()
        if contract_problem:
            raise RuntimeError(f"invalid native V0 source commit: {contract_problem}")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if status:
            print(
                "[orchestrator] WARNING: native V0 is continuing with a dirty "
                f"workspace:\n{status}",
                file=sys.stderr,
                flush=True,
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
        self._ensure_framework_baseline_workspace_contract()
        source_commit = v0_baseline_commit(self.workspace)
        if not source_commit:
            raise RuntimeError("SOL V0 has no committed kernel.py")
        result = self._run_v0_evaluator_with_correctness_prefetch()
        self._finalize_v0_measurement(
            result,
            source_commit,
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
            self._seed_episode_session_from_framework_baseline()
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
        from aka.light.workspace.sources import revision_identity
        v0_identity = revision_identity(self.workspace, baseline_commit)
        self._ensure_framework_baseline_solution_manifest()

        self._link_runtime()
        if action == "run":
            self._restore_framework_baseline_candidate(v0_identity)
        self._prepare_framework_baseline_episode(baseline_commit)
        self._sync_framework_baseline_live(
            phase="framework_baseline_correctness_review"
        )
        self._ensure_framework_baseline_correctness_guidance()
        self._sync_framework_baseline_live(phase="framework_baseline")
        try:
            res = self._run_framework_baseline_agent(
                n=n,
                label="framework baseline",
            )
        finally:
            self._warn_restored_baseline_paths(baseline_commit)
        if res.handoff is None or res.handoff.status != "candidate_ready":
            problem = _framework_baseline_failure_problem(res)
            self._record_framework_baseline_failure(problem)
            self._sync_framework_baseline_live(
                phase="failed",
                state="blocked",
                accepted=False,
                outcome={"summary": problem, "next_directions": []},
            )
            raise RuntimeError(f"framework baseline Episode 0 failed: {problem}")

        # episode-report already ran the normal final-candidate correctness,
        # production-policy and optional numerical gates.  Episode 0 differs
        # from later Episodes only by deliberately skipping incumbent ABBA.
        result, reported_agent_commit = self._framework_baseline_report_result()
        commit = self._commit_framework_baseline(n, result)
        agent_commit = self._seed_episode_session_from_framework_baseline()
        if agent_commit != reported_agent_commit:
            raise RuntimeError(
                "Framework Baseline Episode 0 lost its sealed Agent commit while "
                "seeding Episode 1"
            )
        self._record_framework_baseline_research(
            agent_commit=agent_commit,
            result=result,
            session=res,
        )
        try:
            mark_framework_baseline_accepted(
                self.workspace,
                commit=commit,
                latency_us=result.get("latency_us_geomean"),
            )
        except (OSError, RuntimeError) as exc:
            print(
                "[orchestrator] WARNING: could not close the V1 crash record after "
                f"acceptance: {exc}",
                file=sys.stderr,
                flush=True,
            )
        self._sync_framework_baseline_live(
            phase="recorded",
            state="candidate_ready",
            accepted=True,
            canonical_memory=(f"memory/v{n}.json" if self.generate_memory else ""),
            candidate_commit=agent_commit,
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
            f"@ {(agent_commit[:8] if agent_commit else 'agent-git-unavailable')} "
            f"({latency} us geomean)",
            flush=True,
        )

    def _record_framework_baseline_research(
        self,
        *,
        agent_commit: str,
        result: dict,
        session: SessionResult,
    ) -> None:
        """Record accepted Episode 0 in normal Research and Episode projections."""
        from aka.light.workflow.store import CampaignStore

        runtime = getattr(self, "_supervisor_runtime", None)
        if runtime is None:
            raise RuntimeError("Framework Baseline Episode 0 Runtime is unavailable")
        coordinate = getattr(self, "search_coordinate", None)
        trajectory = (
            coordinate.get("trajectory", 1)
            if isinstance(coordinate, dict)
            else 1
        )
        origin = {
            "epoch": _FRAMEWORK_BASELINE_EPISODE,
            "trajectory": trajectory,
            "episode": _FRAMEWORK_BASELINE_EPISODE,
        }
        latency = result.get("latency_us_geomean")
        runtime.research.checkpoint_lineage(
            {
                "working_kernel_id": agent_commit,
                "lineage_best_kernel_id": agent_commit,
                "lineage_best_latency_us": latency,
                "last_correct_kernel_id": agent_commit,
            },
            origin=origin,
        )
        runtime.research.append(
            "promotion.accepted",
            {
                "kernel_id": agent_commit,
                "latency_us_geomean": latency,
                "source": "framework_baseline_episode_0",
            },
            origin=origin,
            dedupe_key=f"framework-baseline:{agent_commit}",
        )
        agent_workspace = getattr(
            self, "_framework_baseline_agent_workspace", self.workspace
        ).resolve()
        binding = runtime.journals.get(agent_workspace)
        if binding is None:
            raise RuntimeError("Framework Baseline Episode 0 Journal binding is unavailable")
        binding.service.publish_research_files()

        store = CampaignStore(self.workspace)
        attempt = {
            "episode": _FRAMEWORK_BASELINE_EPISODE,
            "memory_version": FRAMEWORK_BASELINE_VERSION,
            "mode": "framework_baseline",
            "status": "candidate_ready",
            "accepted": True,
            "candidate_kernel_commit_id": agent_commit,
            "gateway_record_id": result.get("gateway_record_id"),
            "latency_us_geomean": latency,
            "verification": "ordinary_final_candidate_gates_without_abba",
            "session_id": session.session_id or None,
            "tokens": max(0, int(session.tokens)),
        }
        store.archive_attempt(_FRAMEWORK_BASELINE_EPISODE, attempt)
        state = store.load_state()
        if not any(
            item.get("episode") == _FRAMEWORK_BASELINE_EPISODE
            and item.get("accepted") is True
            and item.get("candidate_kernel_commit_id") == agent_commit
            for item in state.attempts
        ):
            state.attempts.append(attempt)
        state.incumbent_kernel_commit_id = agent_commit
        if session.session_id:
            state.session_id = session.session_id
            state.session_backend = self.agent_cli
        store.save_state(state)

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
            from aka.light.workflow.journal_projection import sync_live_memory
            from aka.light.workflow.store import CampaignStore

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
        from aka.light.workspace.sources import layout_at
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

        from aka.light.workspace.sources import revision_contains_reference
        known_reference_wrapper = revision_contains_reference(self.workspace, self._v0_baseline_commit())
        if not layout_at(self.workspace, committed=True).replaceable_baseline and not known_reference_wrapper:
            return "pin", "source-tree task keeps the pinned package/adapter as its Framework baseline"

        progressed = not head_kernel_is_initial_baseline(self.workspace)
        from aka.light.workspace.sources import read, from_revision
        dirty_candidate = read(self.workspace).snapshot != from_revision(self.workspace, "HEAD").snapshot
        restart_journal = framework_baseline_progress_path(self.workspace).is_file()
        # A compliant dirty candidate must never cause V0 itself to be pinned.  It is an
        # interrupted V1 worktree and needs validation/recovery first.
        if restart_journal:
            return "run", "resuming the unexpected-exit V1 handoff"
        if not progressed and dirty_candidate:
            return "run", "resuming preserved framework-baseline work"

        if self.framework_baseline == "auto" and self.optimization_mode != "production" and self.framework.lower() != "cuda":
            return (
                "skip",
                "leaderboard mode keeps the permissive V0 (use --framework-baseline always)",
            )
        if not progressed:
            if not self.source_checks_enabled:
                return "pin", "multi-GPU Agate source checks are disabled; retain the validated V0"
            structural = production_structure_violations(
                self.workspace,
                self.framework,
                specify_dsl=self.specify_dsl,
            )
            if any(
                value.startswith("unsupported production framework")
                for value in structural
            ):
                return "skip", "; ".join(structural)
            known_reference_wrapper = known_reference_wrapper or is_sol_op(Path(self.kernel_demo).resolve().parent)
            if known_reference_wrapper:
                target = (
                    self.framework
                    if self.specify_dsl
                    else "Agent-selected supported GPU DSL"
                )
                return (
                    "run",
                    f"the supervisor-seeded V0 is the PyTorch reference wrapper, not "
                    f"a self-contained {target} kernel",
                )
            violations = self._production_kernel_violations()
            if not violations:
                return (
                    "pin",
                    "the V0 kernel is already a compliant framework implementation",
                )
            return "run", f"V0 is not a self-contained {self.framework} kernel"
        if latest_version(self.workspace) == FRAMEWORK_BASELINE_VERSION:
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

    def _capture_framework_baseline_exit(
        self,
        *,
        exit_status: int,
        timed_out: bool,
        tokens: int,
        session_id: str,
        stdout_tail: str,
        stderr_tail: str,
        error: str = "",
    ) -> None:
        capture_framework_baseline_exit(
            self.workspace,
            agent_cli=self.agent_cli,
            exit_status=exit_status,
            timed_out=timed_out,
            tokens=tokens,
            session_id=session_id,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=error,
        )

    def _framework_baseline_correctness_context(self) -> list[Path]:
        """Return the bounded public packet shown to the two V1 correctness reviewers."""
        context: list[Path] = []
        for relative in _FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS:
            path = self.workspace / relative
            if path.is_file():
                context.append(path)
        return context

    def _framework_baseline_correctness_context_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"v1-correctness-review-v{_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION}"
            .encode()
        )
        digest.update(b"\0enabled-reviewers\0")
        digest.update(
            ",".join(self._framework_baseline_correctness_reviewers()).encode()
        )
        for value in (self.framework, self.platform, self.arch):
            digest.update(b"\0")
            digest.update(value.encode("utf-8", errors="replace"))
        for path in self._framework_baseline_correctness_context():
            relative = path.relative_to(self.workspace).as_posix()
            contents = path.read_bytes()
            digest.update(b"\0")
            digest.update(relative.encode())
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
        return digest.hexdigest()

    def _load_framework_baseline_correctness_guidance(
        self,
    ) -> dict[str, object] | None:
        path = self.workspace / _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_PATH
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if (
            value.get("schema_version")
            != _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION
            or value.get("context_digest")
            != self._framework_baseline_correctness_context_digest()
        ):
            return None
        reviews = value.get("reviews")
        if not isinstance(reviews, dict):
            return None
        enabled_reviewers = self._framework_baseline_correctness_reviewers()
        if not enabled_reviewers:
            return None
        if not any(
            isinstance(reviews.get(reviewer), dict)
            and reviews[reviewer].get("status") == "ok"
            and isinstance(reviews[reviewer].get("guidance"), str)
            and reviews[reviewer]["guidance"].strip()
            for reviewer in enabled_reviewers
        ):
            return None
        return value

    def _run_framework_baseline_correctness_reviewer(
        self,
        agent_cli: str,
    ) -> tuple[dict[str, object], SessionResult | None]:
        """Run one isolated read-only reviewer and return its bounded guidance."""
        label = "Qoder" if agent_cli == "qodercli" else "Codex"
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"atrex-v1-correctness-{agent_cli}-"
            ) as directory:
                review_workspace = Path(directory)
                context_root = review_workspace / "context"
                source_hashes: dict[str, str] = {}
                for source in self._framework_baseline_correctness_context():
                    relative = source.relative_to(self.workspace)
                    destination = context_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    source_hashes[relative.as_posix()] = hashlib.sha256(
                        destination.read_bytes()
                    ).hexdigest()

                prompt = _render(
                    PROMPTS_DIR / "framework_baseline_correctness_review.md",
                    REVIEWER=label,
                    IMPLEMENTATION_TARGET=(
                        f"a `{self.framework}` Kernel"
                        if self.specify_dsl
                        else "a Kernel in an evaluator-supported GPU DSL"
                    ),
                    PLATFORM=self.platform,
                    ARCH=self.arch or "the runtime GPU architecture",
                )
                result = self._run_session(
                    review_workspace,
                    prompt,
                    timeout=None,
                    agent_cli=agent_cli,
                    reasoning_effort="max",
                    agent_plugins=False,
                    extra_environment=self.agent_boundary_environment("baseline-correctness-review"),
                )
                if result.exit_status != 0 or result.timed_out:
                    detail = result.stderr_tail or result.stdout_tail
                    return (
                        {
                            "status": "failed",
                            "reason": " ".join(detail.split())[:500]
                            or (
                                f"exit={result.exit_status}, "
                                f"timeout={result.timed_out}"
                            ),
                            "session_id": result.session_id,
                        },
                        result,
                    )

                changed = []
                for relative, expected_hash in source_hashes.items():
                    candidate = context_root / relative
                    if (
                        not candidate.is_file()
                        or hashlib.sha256(candidate.read_bytes()).hexdigest()
                        != expected_hash
                    ):
                        changed.append(relative)
                if changed:
                    return (
                        {
                            "status": "failed",
                            "reason": "reviewer modified bounded context: "
                            + ", ".join(sorted(changed)),
                            "session_id": result.session_id,
                        },
                        result,
                    )

                output = review_workspace / "correctness_review.md"
                try:
                    guidance = output.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError) as exc:
                    return (
                        {
                            "status": "failed",
                            "reason": f"missing correctness_review.md: {type(exc).__name__}",
                            "session_id": result.session_id,
                        },
                        result,
                    )
                missing = [
                    marker
                    for marker in _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS
                    if guidance.count(marker) != 1
                ]
                if missing:
                    return (
                        {
                            "status": "failed",
                            "reason": "malformed guidance; missing " + ", ".join(missing),
                            "session_id": result.session_id,
                        },
                        result,
                    )
                return (
                    {
                        "status": "ok",
                        "guidance": guidance[:16000],
                        "session_id": result.session_id,
                    },
                    result,
                )
        except Exception as exc:
            return (
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}"[:500],
                    "session_id": "",
                },
                None,
            )

    def _ensure_framework_baseline_correctness_guidance(self) -> None:
        """Ask the configured V1 reviewers for correctness-first guidance."""
        reviewers = self._framework_baseline_correctness_reviewers()
        labels = [
            "Qoder" if reviewer == "qodercli" else "Codex" for reviewer in reviewers
        ]
        if not reviewers:
            print(
                "[orchestrator] V1 pre-implementation correctness review: disabled by "
                "configuration",
                flush=True,
            )
            return
        if self._load_framework_baseline_correctness_guidance() is not None:
            print(
                "[orchestrator] V1 pre-implementation correctness review: using cached "
                + "/".join(labels)
                + " guidance",
                flush=True,
            )
            return

        print(
            "[orchestrator] V1 pre-implementation correctness review: launching "
            + " and ".join(labels)
            + (" in parallel" if len(reviewers) > 1 else ""),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=len(reviewers)) as executor:
            futures = {
                reviewer: executor.submit(
                    copy_context().run,
                    self._run_framework_baseline_correctness_reviewer, reviewer
                )
                for reviewer in reviewers
            }
            completed = {
                reviewer: future.result() for reviewer, future in futures.items()
            }

        reviews: dict[str, object] = {}
        statuses: list[str] = []
        for reviewer in reviewers:
            record, result = completed[reviewer]
            reviews[reviewer] = record
            if result is not None:
                self._account(result, f"V1 correctness review ({reviewer})")
            status = str(record.get("status", "failed"))
            statuses.append(f"{reviewer}={status}")

        value: dict[str, object] = {
            "schema_version": _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION,
            "context_digest": self._framework_baseline_correctness_context_digest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reviews": reviews,
        }
        path = self.workspace / _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        print(
            "[orchestrator] V1 pre-implementation correctness review: "
            + "; ".join(statuses),
            flush=True,
        )
        if self._load_framework_baseline_correctness_guidance() is None:
            print(
                "[orchestrator] WARNING: no configured reviewer produced valid V1 correctness "
                "guidance; continuing with the public contract",
                file=sys.stderr,
                flush=True,
            )

    def _framework_baseline_correctness_guidance_text(self) -> str:
        reviewers = self._framework_baseline_correctness_reviewers()
        if not reviewers:
            return ""
        value = self._load_framework_baseline_correctness_guidance()
        if value is None:
            return (
                "## External pre-implementation correctness guidance\n\n"
                "No valid guidance from the configured reviewers is available. Derive every "
                "correctness requirement directly from the public contract and immutable "
                "reference before editing.\n"
            )
        reviews = value["reviews"]
        assert isinstance(reviews, dict)
        reviewer_labels = [
            "Qoder" if reviewer == "qodercli" else "Codex" for reviewer in reviewers
        ]
        sections = [
            "## Mandatory external pre-implementation correctness guidance\n",
            " and ".join(reviewer_labels)
            + " reviewed the bounded public packet. Before editing candidate sources, reconcile the "
            "available guidance against the immutable reference. Shared requirements are "
            "mandatory; reviewer suggestions never override the public contract.\n",
        ]
        for reviewer in reviewers:
            label = "Qoder" if reviewer == "qodercli" else "Codex"
            record = reviews.get(reviewer)
            record = record if isinstance(record, dict) else {}
            sections.append(f"\n### {label}\n")
            if record.get("status") == "ok":
                guidance = str(record.get("guidance", ""))
                sections.append(guidance + "\n")
            else:
                sections.append(
                    "Reviewer unavailable: "
                    + str(record.get("reason", "no valid response"))
                    + "\n"
                )
        sections.append(
            "\nBefore implementation, write a concise internal checklist that resolves any "
            "disagreement and covers output initialization/padding, paged addressing, causal "
            "alignment, ragged batches, launch ABI, and numerical stability. Do not create a "
            "plan file.\n"
        )
        return "".join(sections)

    def _run_framework_baseline_agent(
        self,
        *,
        n: int,
        label: str,
        rejection: str = "",
    ) -> SessionResult:
        """Run or resume the Framework Baseline Agent in its preserved trajectory."""
        # Re-project the authoritative boundary before every initial or recovery
        # Session so a prior interrupted Agent cannot leave misleading instructions.
        self._ensure_framework_baseline_workspace_contract()
        prompt = self._framework_baseline_prompt(n)
        if rejection:
            prompt = (
                "# Repair the preserved V1 candidate\n\n"
                f"The supervisor rejected the current candidate:\n\n{rejection}\n\n"
                "Keep sound work, fix this exact rejection, and finish V1. The Supervisor will "
                "run or repeat the configured authoritative gates after this repair; the rejection "
                "text states which checks have already completed. Use Runtime measurement or "
                "diagnostic tools when they help reproduce and repair the failure, but do not "
                "override Runtime-owned correctness cases or create a separate evaluator.\n\n"
                + prompt
            )
        if framework_baseline_progress_path(self.workspace).is_file():
            resumable = bool(self._framework_baseline_session_id())
            prompt = (
                "# Resume the interrupted Framework Baseline\n\n"
                + (
                    "This is the original Provider Session. Continue from the existing conversation, "
                    "Git history, candidate and scratch files. "
                    if resumable
                    else "The original Provider Session is unavailable, but its Git history, "
                    "candidate and scratch files were preserved. Inspect them before continuing. "
                )
                + "Do not start over or repeat completed research.\n\n"
                + prompt
            )
        try:
            from aka.light.workflow.session import LongSessionRunner
            from aka.light.workflow.context import session_execution_policy
            from aka.light.workflow.protocol import atomic_write_text

            observation_root = (
                self.workspace
                / ".atrex_long_horizon"
                / "episodes"
                / f"e{_FRAMEWORK_BASELINE_EPISODE:04d}"
            ).resolve()
            observation_root.mkdir(parents=True, exist_ok=True)
            telemetry_environment = {
                "ATREX_SESSION_CAPTURE_DIR": str(observation_root / "sessions"),
                "ATREX_TELEMETRY_CAMPAIGN_ID": self.campaign_name,
                "ATREX_TELEMETRY_ITERATION_ID": "episode-0000",
                "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
            }
            telemetry_environment.update(self.agent_environment())
            # Framework Baseline and optimization Episodes are one trajectory.
            # Keep their Provider state under the same trajectory-stable Home
            # root so Episode 1 can resume the accepted Baseline Session even
            # though it executes in a different Git worktree.
            from aka.light.agent.home import SESSION_HOME_ROOT_ENV

            telemetry_environment[SESSION_HOME_ROOT_ENV] = str(
                self.workspace / ".atrex_long_horizon" / "agent-session-homes"
            )
            runner = LongSessionRunner(
                agent=self.agent,
                agent_cli=self.agent_cli,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0,
            )
            agent_workspace = getattr(
                self, "_framework_baseline_agent_workspace", self.workspace
            )
            # Runtime publishes the compatibility Journal and handoff into its
            # trusted ``git_workspace`` (the Controller workspace passed to
            # ``register_runtime_episode``), not into the separately projected
            # Agent workspace.  Watch the same Controller projection used by
            # normal Episodes; otherwise an accepted Episode 0 report is
            # durably sealed while the Session Runner incorrectly diagnoses a
            # missing handoff.
            handoff_path = self.workspace / ".atrex_long_horizon" / "handoff.json"
            continuing_session_id = self._framework_baseline_session_id()
            atomic_write_text(observation_root / "BRIEF.md", prompt)
            result = runner.run(
                agent_workspace,
                prompt,
                output_directory=observation_root,
                handoff_path=handoff_path,
                handoff_resumes=self.handoff_resumes,
                completion_check=self._framework_baseline_completion_check,
                telemetry_environment=telemetry_environment,
                stage_name="Episode 0 (Framework baseline)",
                initial_purpose="framework_baseline",
                session_id=continuing_session_id,
                resume_session=bool(continuing_session_id),
                completion_instruction=(
                    "Finish the preserved V1 implementation in the candidate paths declared "
                    "by WORKSPACE.md and the active SourceProvider. Use the normal Runtime "
                    "measurement loop until the candidate passes standard full Evaluate, close "
                    "any active local Directions, leave the complete candidate at a clean Git "
                    "HEAD, and submit the ordinary episode-report for Episode 0. Correctness is "
                    "required; speedup and ABBA are not."
                ),
                **session_execution_policy(self),
            )
        except EnvironmentUnavailable:
            capture_framework_baseline_exit(
                self.workspace,
                agent_cli=self.agent_cli,
                exit_status=75,
                timed_out=False,
                tokens=0,
                session_id="",
                stdout_tail="",
                stderr_tail="remote GPU environment unavailable",
                error="EnvironmentUnavailable",
            )
            raise
        except KeyboardInterrupt:
            # Respect an intentional user stop: preserve a mechanical snapshot, but do not
            # immediately launch another Agent after Ctrl-C.
            capture_framework_baseline_exit(
                self.workspace,
                agent_cli=self.agent_cli,
                exit_status=130,
                timed_out=False,
                tokens=0,
                session_id="",
                stdout_tail="",
                stderr_tail="",
                error="KeyboardInterrupt",
            )
            raise
        except Exception as exc:
            self._capture_framework_baseline_exit(
                exit_status=1,
                timed_out=False,
                tokens=0,
                session_id="",
                stdout_tail="",
                stderr_tail="",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._assert_generalized_inputs_are_private()
        self._account(result, f"{label} v{n} ({self.agent_cli})")
        self._remember_framework_baseline_session(result)
        if result.exit_status != 0 or result.timed_out:
            self._capture_framework_baseline_exit(
                exit_status=result.exit_status,
                timed_out=result.timed_out,
                tokens=result.tokens,
                session_id=result.session_id,
                stdout_tail=result.stdout_tail,
                stderr_tail=result.stderr_tail,
            )
        return result

    def _framework_baseline_completion_check(
        self, handoff: EpisodeHandoff
    ) -> str:
        """Validate Episode 0 through the same published Journal contract as Episodes."""
        from aka.light.workflow.journal_projection import validate_terminal
        from aka.light.workflow.journal_service import sealed_candidate_source
        from aka.light.workspace.sources import from_git

        if handoff.status != "candidate_ready":
            return (
                "Framework Baseline Episode 0 must finish with a correct candidate_ready "
                "report; a pivot cannot establish V1. Repair the candidate or submit a real "
                "blocked report only when a concrete technical blocker prevents completion."
            )
        agent_workspace = getattr(
            self, "_framework_baseline_agent_workspace", self.workspace
        )
        try:
            private = self.read_runtime_episode_journal(
                _FRAMEWORK_BASELINE_EPISODE
            )
        except (OSError, RuntimeError, ValueError) as error:
            return f"cannot read the private Episode 0 Journal: {error}"
        diagnosis = validate_terminal(
            # ``_publish_legacy_report`` projects the public Journal beside the
            # Controller handoff.  The Agent workspace contains editable source
            # and Git history, but is not the compatibility publication target.
            self.workspace / ".atrex_long_horizon" / "journal.json",
            expected_episode=_FRAMEWORK_BASELINE_EPISODE,
            base_commit=str(private.get("base_commit", "")),
            branch=_FRAMEWORK_BASELINE_BRANCH,
            state=handoff.status,
            candidate_commit=handoff.candidate_commit,
            runtime_journal_loader=lambda: private,
        )
        if diagnosis:
            return (
                "Supervisor could not confirm the published Episode 0 handoff: "
                f"{diagnosis}. Resubmit the identical accepted episode-report once to repair "
                "publication; if it persists, report the concrete Runtime blocker."
            )
        try:
            source = sealed_candidate_source(private, self.workspace)
            if source is None:
                return "the accepted Episode 0 report has no sealed candidate source"
            if from_git(agent_workspace, handoff.candidate_commit) != source:
                return (
                    "the accepted Episode 0 Agent commit does not match the Supervisor's "
                    "sealed candidate source"
                )
            self.selected_episode_evaluation(
                _FRAMEWORK_BASELINE_EPISODE, source
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            return (
                "Supervisor could not verify Episode 0's accepted full-Evaluate evidence: "
                f"{error}. Resubmit the exact report only if publication was interrupted; "
                "otherwise repair the cited candidate or evidence."
            )
        return ""

    def _framework_baseline_report_result(self) -> tuple[dict, str]:
        """Return the exact full Evaluate and Agent commit sealed by Episode 0."""
        from aka.light.workflow.journal_service import sealed_candidate_source
        from aka.light.workspace.sources import from_git

        private = self.read_runtime_episode_journal(_FRAMEWORK_BASELINE_EPISODE)
        if private.get("state") != "candidate_ready" or not private.get("finalized_at"):
            raise RuntimeError("Framework Baseline Episode 0 has no accepted candidate report")
        agent_commit = private.get("candidate_kernel_id")
        if not isinstance(agent_commit, str) or not agent_commit:
            raise RuntimeError(
                "Framework Baseline Episode 0 has no sealed Agent Kernel commit identity"
            )
        source = sealed_candidate_source(private, self.workspace)
        if source is None:
            raise RuntimeError("Framework Baseline Episode 0 has no sealed candidate source")
        agent_workspace = getattr(
            self, "_framework_baseline_agent_workspace", self.workspace
        )
        if from_git(agent_workspace, agent_commit) != source:
            raise RuntimeError(
                "Framework Baseline Episode 0 Agent commit differs from its sealed source"
            )
        return (
            self.selected_episode_evaluation(
                _FRAMEWORK_BASELINE_EPISODE, source
            ),
            agent_commit,
        )

    def _prepare_framework_baseline_episode(self, baseline_commit: str) -> Path:
        """Bind Framework Baseline as Episode 0 in the normal Agent workspace.

        Episode 0 deliberately does not increment ``SupervisorState.episodes``:
        it is the framework bring-up stage rather than an optimization-budget
        iteration.  Its Agent Git repository, branches, scratch state, Runtime
        records and provider Session are nevertheless the same trajectory state
        that Episode 1 reuses.
        """
        coordinate = getattr(self, "search_coordinate", None)
        trajectory = (
            coordinate.get("trajectory", 1)
            if isinstance(coordinate, dict)
            else 1
        )
        workspace = self.register_runtime_episode(
            self.workspace,
            episode=_FRAMEWORK_BASELINE_EPISODE,
            base_commit=baseline_commit,
            branch=_FRAMEWORK_BASELINE_BRANCH,
            memory_version=FRAMEWORK_BASELINE_VERSION,
            supervisor_git=True,
            origin={
                "epoch": _FRAMEWORK_BASELINE_EPISODE,
                "trajectory": trajectory,
                "episode": _FRAMEWORK_BASELINE_EPISODE,
            },
        )
        self._framework_baseline_agent_workspace = workspace
        return workspace

    def _framework_baseline_session_id(self) -> str:
        """Return the resumable native Baseline Session, when available."""
        if self.agent_cli != "claude":
            return ""
        path = self.workspace / _FRAMEWORK_BASELINE_SESSION_PATH
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ""
        if not isinstance(value, dict) or value.get("session_backend") != self.agent_cli:
            return ""
        session_id = value.get("session_id")
        return session_id if isinstance(session_id, str) and session_id else ""

    def _remember_framework_baseline_session(self, result: SessionResult) -> None:
        """Persist the latest resumable Baseline Session outside candidate Git."""
        if self.agent_cli != "claude" or not result.session_id:
            return
        from aka.light.workflow.protocol import atomic_write_json

        path = self.workspace / _FRAMEWORK_BASELINE_SESSION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            {
                "session_id": result.session_id,
                "session_backend": self.agent_cli,
            },
        )

    def _seed_episode_session_from_framework_baseline(self) -> str:
        """Seed Episode 1 and return only the Agent-visible baseline commit."""
        from aka.light.workflow.store import CampaignStore

        store = CampaignStore(self.workspace)
        state = store.load_state()
        runtime = getattr(self, "_supervisor_runtime", None)
        candidate = None
        if runtime is not None:
            try:
                candidate = runtime.read_episode_journal(
                    _FRAMEWORK_BASELINE_EPISODE
                ).get("candidate_kernel_id")
            except (OSError, RuntimeError, ValueError):
                candidate = None
            if isinstance(candidate, str) and candidate:
                state.incumbent_kernel_commit_id = candidate
                store.save_state(state)
        if self.agent_cli != "claude":
            return candidate if isinstance(candidate, str) else ""
        marker_path = self.workspace / FRAMEWORK_BASELINE_FILE
        session_path = self.workspace / _FRAMEWORK_BASELINE_SESSION_PATH
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return candidate if isinstance(candidate, str) else ""
        if marker.get("version") != f"v{FRAMEWORK_BASELINE_VERSION}":
            return candidate if isinstance(candidate, str) else ""
        session_id = session.get("session_id")
        session_backend = session.get("session_backend")
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_backend != self.agent_cli
        ):
            return candidate if isinstance(candidate, str) else ""

        # Never replace an Episode Session. This path only seeds a trajectory
        # before Episode 1; later resumes keep the latest Episode-owned ID.
        if state.episodes or state.session_id:
            return candidate if isinstance(candidate, str) else ""
        state.session_id = session_id
        state.session_backend = session_backend
        store.save_state(state)
        return candidate if isinstance(candidate, str) else ""

    def _restore_framework_baseline_candidate(self, v0_identity: str) -> None:
        """Restore an ignored snapshot only when the worktree fell back to the V0 wrapper."""
        from aka.light.workspace.sources import workspace_identity
        if workspace_identity(self.workspace) != v0_identity:
            return
        try:
            progress = load_framework_baseline_progress(self.workspace)
        except RuntimeError as exc:
            print(f"[orchestrator] WARNING: {exc}", file=sys.stderr, flush=True)
            return
        snapshot = progress.get("latest_snapshot")
        if not isinstance(snapshot, dict):
            return
        restored = restore_latest_framework_baseline_candidate(self.workspace)
        if restored and workspace_identity(self.workspace) != v0_identity:
            print(
                "[orchestrator] restored interrupted V1 candidate from local checkpoint: "
                + ", ".join(restored),
                flush=True,
            )

    def _ensure_framework_baseline_solution_manifest(self) -> None:
        from aka.light.workspace.sources import seed_baseline_metadata
        seed_baseline_metadata(self.workspace, self.framework, self.campaign_name,
                               native_eval=bool(self.atrex_bench_root))

    def _ensure_framework_baseline_workspace_contract(self) -> None:
        from aka.light.workspace.files import publish

        workspace = self.workspace.resolve()
        publish(
            workspace,
            "WORKSPACE.md",
            framework_baseline_workspace_contract(workspace),
        )

    def _framework_baseline_prompt(self, n: int) -> str:
        agent_workspace = getattr(
            self, "_framework_baseline_agent_workspace", self.workspace
        )
        baseline_objective = (
            f"Create the first correct, compliant `{self.framework}` Kernel from the V0 "
            "PyTorch reference."
            if self.specify_dsl
            else "Choose a supported DSL and create the first correct, compliant Kernel from "
            "the V0 PyTorch reference."
        )
        return _render(
            PROMPTS_DIR / "framework_baseline.md",
            BASELINE_OBJECTIVE=baseline_objective,
            PLUGINS=self.plugin_directive("framework_baseline"),
            WORKSPACE=str(agent_workspace),
            N=n,
            PREV=n - 1,
            PLATFORM=self.platform,
            FRAMEWORK=self.framework,
            ARCH=self.arch or "the runtime GPU arch",
            DSL_CONTEXT=self.dsl_context,
            NOTES=self.notes,
            CORRECTNESS_GUIDANCE=self._framework_baseline_correctness_guidance_text(),
            SHAPE_DISCLOSURE=self._shape_disclosure_prompt(),
            REFERENCE_KNOWLEDGE=self.reference_knowledge_prompt(),
        )

    def reference_knowledge_prompt(self) -> str:
        if not self.agent_reference_paths:
            return ""
        entries = "\n".join(
            f"- `references/{name}/`" for name, _ in self.agent_reference_paths
        )
        return (
            "## Reference knowledge\n\n"
            f"{entries}\n\n"
            "These Supervisor-mounted directories are read-only engineering references. "
            "Use them selectively for architecture facts, implementation patterns, and probe design. "
            "They are not the task contract, measurement evidence, or candidate source. Verify "
            "applicability on the target hardware through Runtime. Do not execute or import them, "
            "make the final Kernel depend on them, or copy them blindly."
        )

    def _restore_immutable_baseline_paths(self, baseline_commit: str) -> list[str]:
        """Put back any ground-truth file the session edited, and report what was restored.

        A session that "fixes" the harness or evaluator-owned baseline is a compliance problem, but a
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

    def _framework_baseline_problem(
        self,
        v0_identity: str,
        baseline_commit: str,
        *,
        include_policy_review: bool = True,
    ) -> str:
        """Static acceptance checks on the candidate about to be validated and committed.

        Everything is judged from the worktree: that is what the gateway uploads, and it lets a
        session that wrote the kernel but never committed it still be accepted.
        """
        from aka.light.workspace.sources import workspace_identity
        candidate_blob = workspace_identity(self.workspace)
        if not candidate_blob or candidate_blob == v0_identity:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        try:
            workspace_contract = (self.workspace / "WORKSPACE.md").read_bytes()
        except OSError:
            workspace_contract = b""
        try:
            expected_workspace_contract = framework_baseline_workspace_contract(
                self.workspace
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            return f"the candidate SourceProvider contract is invalid: {exc}"
        if workspace_contract != expected_workspace_contract:
            return (
                "the session modified or removed Supervisor-owned WORKSPACE.md; restore the "
                "workspace contract and edit only the SourceProvider-declared candidate paths"
            )
        violations = [] if not self.source_checks_enabled else (
            self._production_kernel_violations()
            if include_policy_review
            else production_structure_violations(
                self.workspace,
                self.framework,
                specify_dsl=self.specify_dsl,
            )
        )
        if violations:
            target = (
                self.framework
                if self.specify_dsl
                else "an Agent-selected supported GPU DSL"
            )
            return (
                f"the candidate is not a self-contained {target} implementation: "
                + "; ".join(violations)
            )
        if (self.source_checks_enabled
                and self.framework.lower() in {"triton", "gluon"}
                and kernel_is_gluon(self.workspace)):
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

    def _framework_baseline_external_gates(
        self, n: int
    ) -> tuple[Optional[dict], str]:
        """Run every independent Framework Baseline gate concurrently."""
        run_policy_review = (
            self.source_checks_enabled and self.optimization_mode == "production"
        )
        if run_policy_review:
            message = (
                "running policy review, combined correctness/performance validation and "
                "supplemental numerical validation in parallel"
            )
        elif self.optimization_mode != "production":
            message = (
                "production policy review skipped in leaderboard mode; running "
                "correctness/performance and supplemental numerical validation in parallel"
            )
        else:
            message = (
                "source review disabled; running correctness/performance and supplemental "
                "numerical validation in parallel"
            )
        print(f"[orchestrator] framework baseline: {message}", flush=True)
        with ThreadPoolExecutor(max_workers=3 if run_policy_review else 2) as executor:
            policy_future = (
                executor.submit(
                    copy_context().run, self._production_kernel_violations
                )
                if run_policy_review
                else None
            )
            validation_future = executor.submit(copy_context().run, self._validate_framework_baseline, n)

            def standard_correctness_passed() -> bool:
                try:
                    _result, problem = validation_future.result()
                except Exception:
                    return False
                return not problem

            numerical_future = executor.submit(
                copy_context().run,
                self.supplemental_numerical_feedback,
                self.workspace,
                flow="framework_baseline",
                standard_correctness_passed=standard_correctness_passed,
            )
            violations: list[str] = []
            if policy_future is not None:
                try:
                    violations = policy_future.result()
                except Exception as exc:
                    violations = [
                        "independent production policy review failed: "
                        f"{type(exc).__name__}: {exc}"
                    ]
            try:
                result, validation_problem = validation_future.result()
            except Exception as exc:
                result, validation_problem = (
                    None,
                    "combined validation failed: "
                    f"{type(exc).__name__}: {exc}",
                )
            try:
                numerical_problem = numerical_future.result()
            except Exception as exc:
                numerical_problem = (
                    "supplemental numerical validation failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        policy_problem = ""
        if violations:
            target = self.framework if getattr(self, "specify_dsl", True) else "an Agent-selected supported GPU DSL"
            policy_problem = (
                f"the candidate is not a self-contained {target} implementation: "
                + "; ".join(violations)
            )
        failures = [
            ("production_review", policy_problem),
            ("correctness_performance", validation_problem),
            ("numerical", numerical_problem),
        ]
        failures = [(name, problem) for name, problem in failures if problem]
        if not failures:
            return result, ""
        primary_problem = failures[0][1]
        statuses = {
            "production_review": (
                "failed" if policy_problem else "passed"
            ) if run_policy_review else "skipped",
            "correctness_performance": "failed" if validation_problem else "passed",
            "numerical": "failed" if numerical_problem else "passed",
        }
        details = "\n".join(
            f"- {name}: {statuses[name]}" + (f" — {problem}" if problem else "")
            for name, problem in (
                ("production_review", policy_problem),
                ("correctness_performance", validation_problem),
                ("numerical", numerical_problem),
            )
        )
        return result, (
            primary_problem
            + "\n\nAll parallel Framework Baseline acceptance checks completed:\n"
            + details
        )

    def _validate_framework_baseline(self, n: int) -> tuple[Optional[dict], str]:
        """Validate V1 once under the same five-case policy as Episodes."""
        command = framework_baseline_validation_command(
            n, atrex_bench=bool(self.atrex_bench_root)
        )
        try:
            test = _sandbox_command(
                self.workspace,
                self.sandbox_hardware,
                self.sandbox_profile,
                self.sandbox_url,
                self.sandbox_timeout,
                command,
                ssh=self.sandbox_ssh,
                ssh_init=self.sandbox_ssh_init,
                health_command=self.sandbox_health_command,
                gateway_kind="run",
                private_reference_dir=self.private_reference_dir,
                evaluation_step_timeout=getattr(self, "evaluation_step_timeout", None),
                gpu_clock_mhz=self.gpu_clock_mhz,
                terminal_validation=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"combined validation failed to run: {exc}"
        self._print_v0_evaluator_output(test)
        from aka.light.evaluation.projection import (
            redact_diagnostics,
            source_error_from_stdout,
        )
        source_error = source_error_from_stdout(test.stdout)
        if source_error is not None:
            error = source_error["error"]
            return None, " ".join(
                (error["message"], "; ".join(error["violations"]), error["next_action"])
            )
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            if test.returncode != 0:
                detail = redact_diagnostics(test.stderr or test.stdout)
                suffix = f": {detail}" if detail else ""
                return None, f"combined validation command failed (exit={test.returncode}){suffix}"
            return None, f"combined validation produced no usable result: {exc}"
        if test.returncode != 0:
            return result, _framework_baseline_validation_failure(
                result, returncode=test.returncode
            )
        if result.get("all_pass") is not True:
            return result, _framework_baseline_validation_failure(
                result, returncode=test.returncode
            )

        latency = result.get("latency_us_geomean")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or latency <= 0
            or not math.isfinite(float(latency))
        ):
            return None, "validation reported no usable latency_us_geomean"
        # Light accepts measured latency, not optional reference-relative scores.
        # Missing metadata must not reject a correct framework implementation.
        by_shape = result.get("latency_us_by_shape")
        if not isinstance(by_shape, dict) or not by_shape or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or not math.isfinite(float(value))
            for value in by_shape.values()
        ):
            return None, "validation reported no usable latency_us_by_shape"
        # Require the framework baseline to preserve full-workload measurement coverage.
        baseline_shapes = set(
            ((read_memory(self.workspace, 0) or {}).get("performance") or {}).get(
                "latency_us_by_shape", {}
            )
        )
        measured_shapes = set(by_shape)
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
        self, problem: str, v0_identity: str, baseline_commit: str, pre_head: str
    ) -> SessionResult:
        """Resume the Framework Baseline Agent so it can repair a rejected candidate."""
        print(
            f"[orchestrator] WARNING: framework baseline rejected ({problem}); "
            "resuming the original Agent session for repair",
            file=sys.stderr,
            flush=True,
        )
        if pre_head and git_head(self.workspace) != pre_head:
            # Undo the session's commits, keep its files: the recovery session needs to read them.
            subprocess.run(
                ["git", "reset", "--mixed", pre_head],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            return self._run_framework_baseline_agent(
                n=FRAMEWORK_BASELINE_VERSION,
                label="framework baseline recovery",
                rejection=problem,
            )
        finally:
            self._warn_restored_baseline_paths(baseline_commit)

    def _repair_framework_baseline_until_accepted(
        self,
        *,
        problem: str,
        v0_identity: str,
        baseline_commit: str,
        pre_head: str,
    ) -> tuple[Optional[dict], str]:
        """Resume one Baseline trajectory until its authoritative gates pass.

        Candidate rejection is actionable feedback, not a retry-budget event.  The
        same Provider Session therefore keeps repairing and resubmitting without a
        hard-coded attempt limit.  Explicitly incomplete supplemental validation is
        still an operator-facing blocker; user cancellation and unrecoverable
        Agent/session failures terminate through their normal paths.
        """
        result: Optional[dict] = None
        while True:
            if problem:
                if problem.startswith(SUPPLEMENTAL_PENDING_PREFIX):
                    return result, problem
                recovery = self._recover_framework_baseline(
                    problem, v0_identity, baseline_commit, pre_head
                )
                if recovery.exit_status != 0 and recovery.tokens == 0:
                    raise RuntimeError(
                        "the resumed framework baseline Agent exited without usable output; "
                        f"inspect {framework_baseline_progress_path(self.workspace)}"
                    )
                problem = self._framework_baseline_problem(
                    v0_identity, baseline_commit, include_policy_review=False
                )
                if problem:
                    continue

            result, problem = self._framework_baseline_external_gates(
                FRAMEWORK_BASELINE_VERSION
            )
            if not problem or problem.startswith(SUPPLEMENTAL_PENDING_PREFIX):
                return result, problem

    def _record_framework_baseline_failure(self, problem: str) -> None:
        """Persist why the framework baseline was rejected, uncommitted so a reset cannot lose it."""
        n = FRAMEWORK_BASELINE_VERSION
        memory_path = version_record_path(self.workspace, n, prefer_summary=self.generate_memory)
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
        memory["correctness"] = {"status": "UNKNOWN" if problem.startswith(SUPPLEMENTAL_PENDING_PREFIX) else "FAIL"}
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
        save_version_record(self.workspace, n, memory, generate_memory=self.generate_memory)

    def _commit_framework_baseline(self, n: int, result: dict) -> str:
        """Commit the accepted kernel (C1) and then pin it in a metadata-only commit (C2)."""
        from aka.light.workspace.sources import read, commit_paths
        artifact = read(self.workspace)
        source = artifact.snapshot
        source_files = artifact.files
        staged = commit_paths(self.workspace, normalize=True)
        staged += [path for path in (
            "CLAUDE.md", "README.md",
            *version_commit_paths(self.workspace, n, generate_memory=self.generate_memory),
        ) if (self.workspace / path).exists()]
        staged = list(dict.fromkeys(staged))
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
            ["git", "rev-list", "-1", "HEAD", "--", *source_files],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        from aka.light.workspace.sources import from_git
        if from_git(self.workspace, kernel_commit) != read(self.workspace).snapshot:
            raise RuntimeError(
                "framework baseline candidate differs between the worktree and the commit"
            )

        _record_local_test_result(
            self.workspace, f"v{n}", result, generate_memory=self.generate_memory
        )
        memory_path = version_record_path(self.workspace, n, prefer_summary=self.generate_memory)
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = FRAMEWORK_BASELINE_CATEGORY
        optimization["action_description"] = (
            f"first self-contained {self.framework} implementation of the whole operator"
        )
        # Canonical memory is projected to the Agent. Never place the private
        # Controller/Supervisor commit in it; use the accepted Agent HEAD only.
        agent_kernel_commit = None
        runtime = getattr(self, "_supervisor_runtime", None)
        if runtime is not None:
            try:
                agent_kernel_commit = runtime.read_episode_journal(
                    _FRAMEWORK_BASELINE_EPISODE
                ).get("candidate_kernel_id")
            except (OSError, RuntimeError, ValueError):
                agent_kernel_commit = None
        memory["git_commit_hash"] = (
            agent_kernel_commit
            if isinstance(agent_kernel_commit, str) and agent_kernel_commit
            else None
        )
        memory[FRAMEWORK_BASELINE_CATEGORY] = {
            "framework": self.framework,
            "validated_stages": ["combined-base-performance+five-cases" if self.atrex_bench_root
                                 else "combined-base-performance+multi-seed-5"],
        }
        save_version_record(self.workspace, n, memory, generate_memory=self.generate_memory)
        self._pin_framework_baseline(kernel_commit, version=n)
        return kernel_commit

    def _pin_framework_baseline(self, commit: str, *, version: int) -> None:
        """Write and commit the framework-baseline marker.

        Deliberately a separate commit rather than an amend: amending would rewrite the very
        commit whose sha the marker records, leaving a dangling pointer. This commit does not
        touch kernel.py, so it never registers as an optimization win.
        """
        from aka.light.workspace.sources import baseline_marker
        marker = {
            "schema_version": 1,
            "version": f"v{version}",
            "framework": self.framework,
            "platform": self.platform,
            "arch": self.arch,
            "commit": commit,
            **baseline_marker(self.workspace, commit),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.workspace / FRAMEWORK_BASELINE_FILE).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths = [FRAMEWORK_BASELINE_FILE]
        paths.extend(path for path in version_commit_paths(
            self.workspace, version, generate_memory=self.generate_memory
        ) if (self.workspace / path).exists())
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
        from aka.light.workflow.episode import LongHorizonCampaign
        from aka.light.workflow.session import LongSessionRunner
        from aka.light.acceptance.recorded_verifier import RecordedABBAValidator

        # LongHorizonCampaign.run creates/measures V0 before constructing its
        # CampaignStore, which installs the Git excludes. A fresh managed
        # workspace does not exist yet at this application boundary.

        verifier = self.acceptance_verifier(
            execute=self.measure_for_acceptance,
            hardware=self.sandbox_hardware,
            profile=self.sandbox_profile,
            url=self.sandbox_url,
            ssh=self.sandbox_ssh,
            ssh_init=self.sandbox_ssh_init,
            health_command=self.sandbox_health_command,
            timeout=self.sandbox_timeout,
            repeats=self.verify_repeats,
            per_run_timeout=self.verify_run_timeout,
            min_improvement_pct=self.min_improvement_pct,
            private_reference_dir=self.private_reference_dir,
            shape_batch_size=self.shape_batch_size,
        )
        engine = LongHorizonCampaign(
            base_campaign=self,
            max_version=self.max_iters,
            token_budget=self.token_budget,
            handoff_resumes=self.handoff_resumes,
            max_stall=self.max_stall,
            verifier=verifier,
            session_runner=LongSessionRunner(agent=self.agent, agent_cli=self.agent_cli,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0),
        )
        return self._finish(engine.run())

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
        if self.generate_memory:
            try:
                subprocess.run(
                    [sys.executable, str(resource("tools/memory_manager.py")),
                     "summary", "--workspace", str(self.workspace)],
                    check=False,
                )
            except OSError:
                pass
        # Production output is fail-closed: do not package a PyTorch baseline,
        # alternate DSL, or independently rejected dependency as a production candidate.
        if self.optimization_mode == "production":
            violations = self._production_kernel_violations(
                require_gluon=kernel_is_gluon(self.workspace)
            )
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
                        str(resource("reference/sol_finalize.py")),
                        "--workspace",
                        str(self.workspace),
                    ],
                    check=False,
                )
            except OSError:
                pass
        return reason
