"""Single-operator optimization campaign: V0 setup and persistent Session segments."""

from __future__ import annotations
from functools import cached_property

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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from aka.light.workspace.resource_catalog import resource

from aka.light.workflow.constants import (
    AGENT_PROBLEM_GENERATION_PROMPT,
    ATREX_BENCH_HARNESS,
    ATREX_PRIVATE_REFERENCE_ENV,
    DEFAULT_VERIFY_REPEATS,
    DEPENDENCY_REVIEW_PROMPT,
    RESEARCH_CLAIM_REVIEW_PROMPT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    DEFAULT_PROBLEM_GENERATION_TIMEOUT_S,
    PROFILE_DRIVER,
    SOL_SEED,
    TEST_RESULT_PREFIX,
    WORKSPACE_INIT,
)
from aka.light.execution.recovery import (
    ENVIRONMENT_TEMPFAIL,
    EnvironmentUnavailable,
    environment_state_file,
)
from aka.light.evaluation.hardware import hardware_vendor, kernel_is_gluon
from aka.light.evaluation.policy import SHAPE_BATCH_SIZE
from aka.light.agent.dependency_policy import DEPENDENCY_GUARD_ENV
from aka.contracts.evaluation import DEFAULT_EVALUATION_STEP_TIMEOUT
from aka.light.workspace.baseline import (
    baseline_measurement_spec,
    read_baseline_record,
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
    workspace_policy_block,
)
from aka.light.acceptance.policy_modes import (
    canonical_submission_policy,
    compute_policy,
    dsl_policy,
    permits_hidden_shapes,
    requires_explicit_dsl,
)
from aka.legacy.acceptance.review import _validate_production_review
from aka.light.workflow.session_io import (
    SessionResult,
    _production_review_candidate_paths,
    _production_review_digest,
    _local_test_record,
    _render,
    _sandbox_command,
    _test_result_from_stdout,
    sandbox_directive,
)
from aka.light.workspace.state import (
    git_head,
    git_path_blob,
    read_memory,
    v0_baseline_commit,
)


_WIKI_PROFILE_ROOT_ENV = "ATREX_WIKI_PROFILE_ROOT"
_WIKI_TASK_ID_ENV = "ATREX_WIKI_TASK_ID"
_CLAIM_REVIEW_MAX_ATTEMPTS = 2



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
    sandbox_hardware: str = (
        ""  # agate scheduler token, e.g. REMOTE_GPU (may differ from platform)
    )
    sandbox_profile: str = ""  # pre/prod; empty preserves normal agate URL resolution
    sandbox_url: str = ""  # explicit endpoint, e.g. http://127.0.0.1:8000
    sandbox_timeout: int | None = None
    atrex_bench_root: str = ""  # native evaluator checkout owning run_eval.py
    agent_cli: str = "claude"  # episode backend selected from the Agent registry
    agent_sandbox: str = "none"  # opt-in coordinator isolation, independent of GPU transport
    bwrap_executable: str = "bwrap"
    agent_read_only_paths: tuple[str, ...] = ()
    agent_reference_paths: tuple[tuple[str, str], ...] = ()
    optimization_mode: str = "fast"
    expose_evaluator_shapes: bool = True
    verify_repeats: int = DEFAULT_VERIFY_REPEATS
    # One Runtime-owned per-Shape budget shared by Eval and ABBA. The legacy
    # ``verify_run_timeout`` accessor below remains a structural compatibility
    # alias for upstream v0.4 call sites.
    evaluation_step_timeout: int = DEFAULT_EVALUATION_STEP_TIMEOUT
    shape_batch_size: int = SHAPE_BATCH_SIZE
    gpu_clock_mhz: int | None = None
    min_improvement_pct: float = 0.0
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
    _measurement_store_root: Path | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.specify_dsl) is not bool:
            raise TypeError("specify_dsl must be a boolean")
        if self.optimization_mode in {"fast", "performance"}:
            self.specify_dsl = False
        elif self.optimization_mode == "ultra":
            if not self.specify_dsl or not self.framework.strip():
                raise ValueError(
                    "ultra mode requires one explicit framework and does not support open DSL"
                )
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
        if self.optimization_mode == "leaderboard":
            return (
                f"- Recommended candidate DSL: `{self.framework}`"
                if self.specify_dsl
                else ""
            )
        policy = canonical_submission_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        )
        if policy == "fast":
            return "- Final-source DSL: Agent-selected from evaluator-supported implementations."
        if policy == "performance":
            return (
                "- Final-source DSL: Agent-selected using the supplied expert guidance and "
                "measurement evidence; recommendations are not hard submission constraints."
            )
        if requires_explicit_dsl(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        ):
            return (
                f"- Required final candidate DSL: `{self.framework}`. Development probes and "
                "temporary prototypes may use any language or DSL; they must not remain in the "
                "submitted Kernel source tree or runtime compute path unless compliant."
            )
        return ""

    @property
    def dsl_query_context(self) -> str:
        return f", DSL `{self.framework}`" if self.specify_dsl else ""

    @property
    def private_reference_dir(self) -> Path | None:
        """Return evaluator-only native inputs behind a generalized public problem."""
        if self.expose_evaluator_shapes:
            return None
        op_dir = Path(self.kernel_demo).resolve().parent
        if not permits_hidden_shapes(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        ):
            raise ValueError(
                "--no-expose-evaluator-shapes is supported only for performance/ultra "
                "Native Atrex-Bench tasks"
            )
        if is_sol_op(op_dir):
            raise ValueError(
                "--no-expose-evaluator-shapes is incompatible with SOL tasks: "
                "SOL has no generalized agent_problem.json projection"
            )
        if not (op_dir / "shapes.json").is_file():
            raise ValueError(
                "--no-expose-evaluator-shapes requires Native Atrex-Bench shapes.json"
            )
        if not self.atrex_bench_root:
            raise ValueError(
                "--no-expose-evaluator-shapes requires the canonical Native Atrex-Bench "
                "runtime so Light can create agent_problem.json"
            )
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
        if not permits_hidden_shapes(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        ):
            raise RuntimeError(
                "a generalized fast-mode campaign requires a user-provided "
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

    def agent_boundary_environment(self, role: str = "optimizer") -> dict[str, str]:
        return {
            "ATREX_AGENT_SANDBOX": self.agent_sandbox,
            "ATREX_BWRAP_EXECUTABLE": self.bwrap_executable,
            "ATREX_AGENT_READ_ONLY_PATHS": json.dumps(self.agent_read_only_paths),
            "ATREX_AGENT_REFERENCE_PATHS": json.dumps(dict(self.agent_reference_paths)),
            "ATREX_AGENT_WORKSPACE_ROLE": role,
        }

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

    def start_runtime(self) -> None:
        from aka.light.runtime.server import RuntimeConfig

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
            measurement_root=self._measurement_store_root,
            research_synchronization=(
                str(self.search_coordinate.get("synchronization"))
                if isinstance(getattr(self, "search_coordinate", None), dict)
                and self.search_coordinate.get("cross_trajectory_sharing") is True
                else "independent"
            ),
            request_timeout=request_timeout,
            queue_timeout=queue_timeout,
        ), workspace_provider=self.candidate_workspace)
        self._supervisor_runtime.production_policy_validator = self.production_policy_feedback
        self._supervisor_runtime.production_review_required = (
            self.source_checks_enabled
        )
        self._supervisor_runtime.submission_specify_dsl = self.specify_dsl
        self._supervisor_runtime.numerical_validator = self.supplemental_numerical_feedback
        self._supervisor_runtime.research_claim_reviewer = self.review_research_claim

    def review_research_claim(
        self,
        evidence_root: Path,
        *,
        resume_session_id: str | None = None,
        archive_root: Path | None = None,
    ) -> dict[str, object]:
        """Run the strict three-tier reviewer with the Qwen-only provider policy."""
        from aka.light.agent.session_tail import read_regular_bytes
        from aka.light.execution.retry import check_review_service
        from aka.light.workflow.claim_review import validate_review

        evidence_root = Path(evidence_root).resolve(strict=True)
        claim_type = None
        try:
            evidence_manifest = json.loads(
                read_regular_bytes(
                    evidence_root / "evidence-manifest.json",
                    limit=1024 * 1024,
                )
            )
        except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
            evidence_manifest = None
        if isinstance(evidence_manifest, dict):
            candidate_type = evidence_manifest.get("claim_type")
            if candidate_type in {"direction", "experiment"}:
                claim_type = candidate_type
        # Valid Runtime evidence always identifies the claim type. Keep the
        # legacy bucket only for old/manual review bundles that predate that
        # manifest field; it remains visible when such receipts are present.
        review_kind = (
            f"{claim_type}_claim_review"
            if claim_type is not None
            else "research_claim_review"
        )
        model = "qwen3.8-max"
        boundary_environment = self.agent_boundary_environment(
            "research-claim-review"
        )
        # Independent claim review is a security boundary, not an optimizer
        # convenience setting.  Always use Bubblewrap even when the selected
        # optimizer profile deliberately runs without a sandbox.  If bwrap is
        # unavailable, launch fails closed and the caller archives a
        # review_unavailable receipt instead of executing the reviewer natively.
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
        runtime = getattr(self, "_supervisor_runtime", None)
        runtime_root = getattr(runtime, "root", None)
        capture_base = (
            Path(runtime_root) / "review-capture" / "claims" / evidence_root.name
            if runtime_root is not None
            else Path(tempfile.gettempdir())
            / "atrex-light-review-capture"
            / "claims"
            / f"{evidence_root.name}-{uuid.uuid4().hex}"
        )
        archive_root = Path(archive_root).resolve() if archive_root is not None else None
        session_attempts: list[dict[str, object]] = []

        def session_audit() -> dict[str, object]:
            return {
                "schema": "atrex.research-claim-review-session.v2",
                "attempts": [dict(row) for row in session_attempts],
            }

        last_error: Exception | None = None
        # One initial attempt plus one automatic retry.  Only after both fail
        # does Journal projection expose ``review_unavailable``.
        for attempt in range(1, _CLAIM_REVIEW_MAX_ATTEMPTS + 1):
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
                    "token_usage_path": None,
                    "resumed": resumed,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error)[:2000],
                    },
                })
                continue
            from aka.light.workflow.review_usage import account_review
            account_review(
                self,
                result,
                review_kind=review_kind,
                label=(
                    f"Qwen {claim_type}-claim review attempt {attempt}"
                    if claim_type is not None
                    else f"Qwen research-claim review attempt {attempt}"
                ),
                backend="claude",
                model=model,
                persist_separate_receipt=False,
            )
            from aka.light.agent.session_capture import latest_complete_session_archive

            archive = latest_complete_session_archive(attempt_capture)
            if archive is not None and archive_root is not None:
                from aka.light.agent.session_tail import read_regular_bytes
                from aka.light.experiment.store import private_write_bytes

                attempt_root = archive_root / "attempts" / f"a{attempt}"
                conversation_data = read_regular_bytes(
                    archive.conversation.resolve(strict=True),
                    limit=64 * 1024 * 1024 + 1,
                )
                token_data = read_regular_bytes(
                    archive.token_usage.resolve(strict=True),
                    limit=64 * 1024 * 1024 + 1,
                )
                if len(conversation_data) > 64 * 1024 * 1024 or len(token_data) > 64 * 1024 * 1024:
                    raise ValueError("research claim Review archive exceeds 64 MiB")
                private_write_bytes(
                    attempt_root / "conversation.jsonl",
                    conversation_data,
                )
                private_write_bytes(
                    attempt_root / "token-usage.json",
                    token_data,
                )
                conversation_path = f"attempts/a{attempt}/conversation.jsonl"
                token_usage_path = f"attempts/a{attempt}/token-usage.json"
            else:
                conversation_path = (
                    archive.conversation.relative_to(evidence_root.parent).as_posix()
                    if archive is not None else None
                )
                token_usage_path = (
                    archive.token_usage.relative_to(evidence_root.parent).as_posix()
                    if archive is not None else None
                )
            attempt_audit: dict[str, object] = {
                "attempt": attempt,
                "session_id": result.session_id,
                "status": "capture_missing" if conversation_path is None else "invalid",
                "exit_status": result.exit_status,
                "timed_out": result.timed_out,
                "conversation_path": conversation_path,
                "token_usage_path": token_usage_path,
                "resumed": resumed,
            }
            session_attempts.append(attempt_audit)
            if archive is None:
                from aka.light.execution.retry import ReviewerOutputUnavailable

                last_error = ReviewerOutputUnavailable(
                    "research claim reviewer did not persist a complete "
                    "conversation.jsonl and token-usage.json archive"
                )
                attempt_audit["error"] = {
                    "type": type(last_error).__name__,
                    "message": str(last_error),
                }
                continue
            try:
                value = json.loads(read_regular_bytes(response, limit=128 * 1024 + 1))
                if response.stat().st_size > 128 * 1024:
                    raise ValueError("research claim review exceeds 128 KiB")
                review = validate_review(value)
                if archive is not None and archive_root is not None:
                    from aka.light.agent.session_tail import read_regular_bytes
                    from aka.light.experiment.store import private_write_bytes

                    conversation_data = read_regular_bytes(
                        archive.conversation.resolve(strict=True),
                        limit=64 * 1024 * 1024 + 1,
                    )
                    token_data = read_regular_bytes(
                        archive.token_usage.resolve(strict=True),
                        limit=64 * 1024 * 1024 + 1,
                    )
                    if len(conversation_data) > 64 * 1024 * 1024 or len(token_data) > 64 * 1024 * 1024:
                        raise ValueError("research claim Review archive exceeds 64 MiB")
                    private_write_bytes(
                        archive_root / "conversation.jsonl",
                        conversation_data,
                    )
                    private_write_bytes(
                        archive_root / "token-usage.json",
                        token_data,
                    )
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
        submission_id: str | None = None,
    ) -> dict[str, object] | list[str]:
        """Review exact frozen candidate bytes and preserve observed implementation facts."""
        if not self.source_checks_enabled:
            return []
        from aka.light.workspace.sources import install

        review_records: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="atrex-production-candidate-") as directory:
            frozen = Path(directory)
            install(frozen, source)
            review_kwargs = {
                "require_gluon": require_gluon,
                "review_record_sink": review_records.append,
            }
            if submission_id is not None:
                review_kwargs["submission_id"] = submission_id
            violations = self._production_kernel_violations(
                frozen, **review_kwargs
            )
        implementation_review: dict[str, object] = {
            "availability": "unavailable",
        }
        if review_records:
            record = review_records[-1]
            payload = record.get("review")
            payload = payload if isinstance(payload, dict) else {}
            framework = payload.get("framework")
            framework = framework if isinstance(framework, dict) else {}
            technologies = framework.get("technologies")
            unexpected = framework.get("unexpected_core_technologies")
            implementation_review = {
                "availability": "available",
                "technologies": (
                    list(technologies) if isinstance(technologies, list) else []
                ),
                "dsl_policy": record.get("dsl_policy"),
                "configured_dsl": record.get("framework"),
                "required_phase": record.get("required_phase"),
                "evaluator_supported": framework.get("evaluator_supported"),
                "configured_dsl_used": framework.get("configured_framework_used"),
                "required_phase_used": framework.get("required_phase_used"),
                "unexpected_core_technologies": (
                    list(unexpected) if isinstance(unexpected, list) else []
                ),
            }
        return {
            "violations": violations,
            "implementation_review": implementation_review,
        }

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

    def acceptance_measurement_context(
        self, agent_workspace: Path, incumbent_source: bytes
    ) -> dict:
        """Return the exact Agent-visible incumbent identity and frozen policy."""
        self.start_runtime()
        runtime = self._supervisor_runtime
        baseline_kernel_id = runtime.baseline_kernel_commit(agent_workspace)
        # Bind source under the Agent-visible commit. Runtime-private source
        # archives never replace or leak a second Git identity into requests.
        runtime.measurements.kernel(incumbent_source, baseline_kernel_id)
        return {
            "baseline_kernel_id": baseline_kernel_id,
            "measurement_repetitions": runtime.measurement_repetitions,
        }

    def register_runtime_episode(
        self,
        workspace: Path,
        *,
        episode: int,
        base_commit: str,
        branch: str,
        memory_version: int,
        baseline_kernel_commit_id: str | None = None,
        managed_agent_workspace: bool = True,
        require_gluon: bool = False,
        origin: dict[str, int] | None = None,
        session_capture_root: Path | None = None,
    ) -> Path:
        """Start Runtime if needed and bind an audit segment to Agent Git."""
        self.start_runtime()
        return self._supervisor_runtime.register_episode(
            workspace,
            episode=episode,
            base_commit=base_commit,
            branch=branch,
            memory_version=memory_version,
            baseline_kernel_commit_id=baseline_kernel_commit_id,
            managed_agent_workspace=managed_agent_workspace,
            require_gluon=require_gluon,
            session_capture_root=session_capture_root,
            origin=(origin or getattr(self, "search_coordinate", None) or {
                "epoch": episode,
                "trajectory": 1,
                "episode": episode,
            }),
        )


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
        """Require the private evaluation result to cover every shape by opaque id."""
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
                "evaluation result lacks complete real-shape performance coverage "
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
        submission_id: str,
        review_record: dict[str, object],
        *,
        conversation: Path | None = None,
        token_usage: Path | None = None,
    ) -> list[str]:
        from aka.light.runtime.paths import supervisor_submission_root
        from aka.light.workflow.review_archive import archive_review

        record_path = (
            supervisor_submission_root(workspace)
            / submission_id
            / "review"
        )
        try:
            archive_review(
                record_path,
                result=review_record,
                conversation=conversation,
                token_usage=token_usage,
            )
        except (OSError, ValueError) as exc:
            return [
                "could not persist final-source policy verdict: "
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
        submission_id: str | None = None,
        record_workspace: Path | None = None,
        review_record_sink=None,
    ) -> list[str]:
        """Delegate complete candidate policy review to a fresh, isolated agent."""
        from aka.light.execution.retry import check_review_service
        from aka.light.runtime.paths import supervisor_submission_root

        cache_key = candidate_digest + ":" + self.agent_cli
        cached = self._production_review_cache.get(cache_key)
        if (
            cached is not None
            and submission_id is not None
            and not isinstance(cached[1].get("submission_id"), str)
        ):
            # An advisory caller may have populated the in-memory verdict
            # without a durable Review archive. A formal submission needs its
            # own co-located conversation and token receipt, so rerun it.
            cached = None
        if cached is not None:
            cached_errors, review_record = cached
            if review_record_sink is not None:
                review_record_sink(dict(review_record))
            if submission_id is None:
                return list(cached_errors)
            previous_submission_id = review_record.get("submission_id")
            previous_root = (
                supervisor_submission_root(self.workspace)
                / str(previous_submission_id)
                / "review"
            )
            reused_record = {
                **review_record,
                "submission_id": submission_id,
                "review_execution": "reused",
                "reused_from_submission_id": previous_submission_id,
            }
            persistence_errors = self._persist_production_review_record(
                self.workspace,
                submission_id,
                reused_record,
                conversation=previous_root / "conversation.jsonl",
                token_usage=previous_root / "token-usage.json",
            )
            return list(dict.fromkeys([*cached_errors, *persistence_errors]))

        errors: list[str]
        review_payload: object | None = None
        derived_review: dict[str, object] | None = None
        review_summary = ""
        effective_dsl_policy = dsl_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        )
        effective_compute_policy = compute_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        )
        pinned_dsl = effective_dsl_policy == "pinned"
        from aka.light.agent.home import SESSION_HOME_ROOT_ENV
        from aka.light.runtime.paths import supervisor_campaign_root

        supervisor_root = supervisor_campaign_root(self.workspace)
        runtime = getattr(self, "_supervisor_runtime", None)
        runtime_root = getattr(runtime, "root", None)
        capture_root = (
            (
                Path(runtime_root)
                if runtime_root is not None
                else supervisor_root
            )
            / "review-capture"
            / "final-source"
            / (submission_id or uuid.uuid4().hex)
        )
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
                "framework": framework if pinned_dsl else None,
                "recommended_framework": (
                    framework
                    if effective_dsl_policy == "guided" and framework
                    else None
                ),
                "dsl_policy": effective_dsl_policy,
                "compute_policy": effective_compute_policy,
                "required_phase": (
                    "gluon"
                    if require_gluon and pinned_dsl
                    else "selected_framework"
                    if pinned_dsl
                    else "open_dsl"
                ),
                "optimization_mode": canonical_submission_policy(
                    self.optimization_mode,
                    specify_dsl=self.specify_dsl,
                ),
                "candidate_digest": candidate_digest,
                "candidate_files": sorted(source_hashes),
            }
            (review_workspace / "review_request.json").write_text(
                json.dumps(request, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            try:
                review_environment = {
                    **self.agent_boundary_environment("production-review"),
                    "ATREX_SESSION_CAPTURE_DIR": str(capture_root),
                    SESSION_HOME_ROOT_ENV: str(
                        supervisor_root
                        / "review-session-homes"
                        / "final_source_policy_review"
                    ),
                }
                result = self._run_session(
                    review_workspace,
                    DEPENDENCY_REVIEW_PROMPT.read_text(encoding="utf-8"),
                    timeout=None,
                    agent_cli=self.agent_cli,
                    reasoning_effort="high",
                    agent_plugins=False,
                    extra_environment=review_environment,
                )
                from aka.light.workflow.review_usage import account_review
                account_review(
                    self,
                    result,
                    review_kind="final_source_policy_review",
                    label="independent final-source policy review",
                    backend=self.agent_cli,
                    persist_separate_receipt=False,
                )
                from aka.light.agent.session_capture import latest_complete_session_archive
                from aka.light.execution.retry import ReviewerOutputUnavailable

                session_archive = latest_complete_session_archive(capture_root)
                if session_archive is None:
                    raise ReviewerOutputUnavailable(
                        "independent final-source policy reviewer did not persist "
                        "a complete conversation.jsonl and token-usage.json archive"
                    )
                check_review_service(result)
            except ValueError as exc:
                return [f"independent final-source policy review failed: {exc}"]
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
                    "independent final-source policy review modified candidate evidence: "
                    + ", ".join(sorted(changed))
                ]
            else:
                review_path = review_workspace / "dependency_review.json"
                try:
                    review_payload = json.loads(
                        review_path.read_text(encoding="utf-8")
                    )
                    errors, review_summary, derived_review = _validate_production_review(
                        review_payload,
                        candidate_files=frozenset(source_hashes),
                        dsl_policy=effective_dsl_policy,
                        compute_policy=effective_compute_policy,
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    from aka.light.execution.retry import ReviewerOutputUnavailable

                    raise ReviewerOutputUnavailable(
                        "independent final-source policy review produced no valid verdict: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                else:
                    status = "accepted" if not errors else "rejected"
                    print(
                        f"[final-source-policy] independent full-candidate review {status}: "
                        f"{review_summary}",
                        flush=True,
                    )

        try:
            reviewed_digest = _production_review_digest(
                workspace,
                framework,
                require_gluon,
                pinned_dsl,
                canonical_submission_policy(
                    self.optimization_mode,
                    specify_dsl=self.specify_dsl,
                ),
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
            "framework": framework if pinned_dsl else None,
            "dsl_policy": effective_dsl_policy,
            "compute_policy": effective_compute_policy,
            "required_phase": (
                "gluon"
                if require_gluon and pinned_dsl
                else "selected_framework"
                if pinned_dsl
                else "open_dsl"
            ),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "agent_runtime": self.agent_cli,
            "accepted": not errors,
            "errors": errors,
            "summary": review_summary,
            "review": review_payload,
            "derived_review": derived_review,
            "review_execution": "executed",
            "session_id": result.session_id,
        }
        if submission_id is not None:
            review_record["submission_id"] = submission_id
            persistence_errors = self._persist_production_review_record(
                self.workspace,
                submission_id,
                review_record,
                conversation=session_archive.conversation,
                token_usage=session_archive.token_usage,
            )
        else:
            persistence_errors = []
        self._production_review_cache[cache_key] = (tuple(errors), review_record)
        if review_record_sink is not None:
            review_record_sink(dict(review_record))
        return list(dict.fromkeys([*errors, *persistence_errors]))

    def _review_production_candidate(
        self,
        workspace: Path,
        framework: str,
        require_gluon: bool,
        *,
        review_record_sink=None,
        submission_id: str | None = None,
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
            workspace,
            framework,
            require_gluon,
            dsl_policy(
                self.optimization_mode,
                specify_dsl=self.specify_dsl,
            ) == "pinned",
            canonical_submission_policy(
                self.optimization_mode,
                specify_dsl=self.specify_dsl,
            ),
        )
        stage = f"dependency-review:{digest}:{self.agent_cli}:unbounded"
        try:
            return retry_review(
                self.workspace, stage,
                lambda: self._review_production_candidate_once(
                    workspace,
                    framework,
                    require_gluon,
                    candidate_digest=digest,
                    submission_id=submission_id,
                    review_record_sink=review_record_sink,
                ),
                cancel=getattr(self, "search_cancel", None),
            )
        except ValueError as exc:
            return [f"independent final-source policy review failed: {exc}"]

    def _production_kernel_violations(
        self,
        workspace: Path | None = None,
        *,
        require_gluon: bool = False,
        review_record_sink=None,
        submission_id: str | None = None,
    ) -> list[str]:
        if not self.source_checks_enabled:
            return []
        production_reviewer = self._review_production_candidate
        if review_record_sink is not None:
            production_reviewer = lambda candidate, framework, gluon: (
                self._review_production_candidate(
                    candidate,
                    framework,
                    gluon,
                    review_record_sink=review_record_sink,
                    submission_id=submission_id,
                )
            )
        elif submission_id is not None:
            production_reviewer = lambda candidate, framework, gluon: (
                self._review_production_candidate(
                    candidate,
                    framework,
                    gluon,
                    submission_id=submission_id,
                )
            )
        return production_kernel_violations(
            workspace or self.workspace,
            self.framework,
            require_gluon=require_gluon,
            specify_dsl=(
                dsl_policy(
                    self.optimization_mode,
                    specify_dsl=self.specify_dsl,
                ) == "pinned"
            ),
            production_reviewer=production_reviewer,
        )

    @cached_property
    def source_checks_enabled(self) -> bool:
        """Freeze main's multi-GPU source policy from evaluator-owned metadata."""
        if self.optimization_mode in {"fast", "performance", "ultra"}:
            # The three explicit policies define final-source review semantics
            # independently of GPU count.  Communication/runtime plumbing can
            # be allowed by the reviewer without disabling provenance review.
            return True
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
            "- Ground-truth operator files and Supervisor evaluation evidence are immutable to the Agent.\n\n"
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

    def _run_v0_evaluator(self, source_commit: str) -> dict:
        """Run and immediately seal V0 as a standard Campaign measurement."""
        from aka.light.experiment.store import DuplicateTask

        store, request, kernels = baseline_measurement_spec(
            self.workspace, source_commit
        )
        try:
            with store.reserve(request, kernels) as task:
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
                    evaluation_step_timeout=self.evaluation_step_timeout,
                    gpu_clock_mhz=self.gpu_clock_mhz,
                    job_root=task.directory / "repetition-1",
                    # V0 is mandatory campaign evidence. Keep polling/retrying through
                    # infrastructure failures exactly like later terminal validation.
                    terminal_validation=True,
                )
                response = {
                    "exit_code": test.returncode,
                    "stdout": test.stdout,
                    "stderr": test.stderr,
                }
                # Seal every terminal V0 result, including a correctness rejection.
                # Infrastructure interruptions that throw before a terminal result
                # leave this exact reservation and its physical Job checkpoints
                # available for recovery on the next launch.
                task.finish(response, cacheable=True)
        except DuplicateTask as duplicate:
            saved = store.read(duplicate.record_id)
            receipt = saved.get("response", {}).get("agent_result", {})
            saved_result = receipt.get("result", {}) if isinstance(receipt, dict) else {}
            result = dict(saved_result) if isinstance(saved_result, dict) else {}
            result["all_pass"] = result.get("correct") is True
            if result["all_pass"] is not True:
                raise RuntimeError("saved V0 measurement did not pass correctness")
            return result
        self._print_v0_evaluator_output(test)
        if test.returncode == ENVIRONMENT_TEMPFAIL:
            raise EnvironmentUnavailable("V0 evaluation infrastructure is unavailable; retain its job evidence for recovery")
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"sandbox V0 Evaluate produced no usable result: {exc}"
            ) from exc
        if test.returncode != 0 or not result.get("all_pass"):
            raise RuntimeError(
                "sandbox V0 Evaluate failed correctness/performance validation"
            )
        return result


    def _finalize_v0_measurement(
        self,
        result: dict,
        source_commit: str,
    ) -> None:
        """Validate the V0 Measurement while keeping HEAD at its source commit."""
        memory = read_baseline_record(self.workspace)
        if not isinstance(memory, dict):
            raise RuntimeError("V0 evaluator produced no MeasurementStore record")
        if memory.get("git_commit_hash") != source_commit:
            raise RuntimeError("V0 measurement does not point to its source commit")
        coverage_problem = self._generalized_memory_coverage_problem(memory)
        if coverage_problem:
            raise RuntimeError(f"invalid native V0 measurement: {coverage_problem}")
        recorded = read_memory(self.workspace, 0) or {}
        if recorded.get("git_commit_hash") != source_commit:
            raise RuntimeError("V0 measurement record does not point to the V0 source commit")
        from aka.light.workspace.sources import from_git, snapshot as source_snapshot
        archived_source = from_git(self.workspace, source_commit)
        if archived_source != source_snapshot(self.workspace):
            raise RuntimeError("V0 candidate source differs from its recorded source commit")
        if git_head(self.workspace) != source_commit:
            raise RuntimeError("V0 measurement must not advance Git beyond its source commit")

        # Current managed Campaigns keep exactly one canonical V0 source
        # archive in MeasurementStore.  campaign.json records only its public
        # Agent Git identity and content digest; source preparation snapshots
        # are temporary and never become persistent Campaign state.
        from aka.light.runtime.paths import (
            CAMPAIGN_MANIFEST,
            record_campaign_v0,
            supervisor_campaign_base_root,
            supervisor_measurement_root,
        )
        campaign_root = supervisor_campaign_base_root(self.workspace)
        if (campaign_root / CAMPAIGN_MANIFEST).is_file():
            from aka.light.experiment.store import MeasurementStore

            identity = MeasurementStore(
                supervisor_measurement_root(self.workspace)
            ).kernel(archived_source, source_commit)
            record_campaign_v0(
                self.workspace,
                kernel_commit_id=identity["kernel_id"],
                kernel_artifact_digest=identity["kernel_artifact_digest"],
            )

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
        (self.workspace / "README.md").write_text(
            self._native_v0_readme(generalized=generalized), encoding="utf-8"
        )
        source_paths = [
            ".gitignore",
            "CLAUDE.md",
            "README.md",
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

        result = self._run_v0_evaluator(source_commit)
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
        source_commit = v0_baseline_commit(self.workspace)
        if not source_commit:
            raise RuntimeError("SOL V0 has no committed kernel.py")
        result = self._run_v0_evaluator(source_commit)
        self._finalize_v0_measurement(
            result,
            source_commit,
        )

    def run(self) -> str:
        """Run the native long-horizon episode supervisor for this campaign."""
        from aka.light.workflow.episode import LongHorizonCampaign
        from aka.light.workflow.session import LongSessionRunner

        # LongHorizonCampaign.run creates/measures V0 before constructing its
        # CampaignStore, which installs the Git excludes. A fresh managed
        # workspace does not exist yet at this application boundary.

        engine = LongHorizonCampaign(
            base_campaign=self,
            max_episodes=self.max_iters,
            token_budget=self.token_budget,
            session_runner=LongSessionRunner(agent=self.agent, agent_cli=self.agent_cli,
                usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0),
        )
        return self._finish(engine.run())

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
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
