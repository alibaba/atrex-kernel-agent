"""Single-operator preparation, source review and persistent Session segments."""

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
from aka.resources.bundles import resource_path as resource
from aka.workflow.light.long_horizon.review_archive import (
    review_root, review_revision_root, reviewer_environment, reviewer_workspace,
)
from aka.workflow.light.support.paths import supervisor_measurement_root

from aka.workflow.light.orchestration.constants import (
    AGENT_PROBLEM_GENERATION_PROMPT,
    ATREX_PRIVATE_REFERENCE_ENV,
    DEFAULT_VERIFY_REPEATS,
    DEPENDENCY_REVIEW_PROMPT,
    RESEARCH_CLAIM_REVIEW_PROMPT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    DEFAULT_PROBLEM_GENERATION_TIMEOUT_S,
    SOL_SEED,
    WORKSPACE_INIT,
)
from aka.execution.recovery import (
    ENVIRONMENT_TEMPFAIL,
    EnvironmentUnavailable,
    environment_state_file,
)
from aka.workflow.light.orchestration.hardware import hardware_vendor, kernel_is_gluon
from aka.workflow.light.measurement_policy import SHAPE_BATCH_SIZE
from aka.agent.execution.dependency_policy import DEPENDENCY_GUARD_ENV
from aka.contracts.evaluation import DEFAULT_EVALUATION_STEP_TIMEOUT
from aka.task.problem.layout import (
    AGENT_PROBLEM_FILENAME,
    agent_visible_operator_files,
    has_agent_problem,
    is_sol_op,
    validate_agent_problem,
    validate_generated_agent_problem,
    validate_private_shapes,
)
from aka.workflow.light.orchestration.optimization_policy import (
    install_workspace_policy,
    optimization_mode_directive,
    production_kernel_violations,
    production_structure_violations,
    workspace_policy_block,
)
from aka.workflow.light.long_horizon.policy_modes import (
    canonical_submission_policy,
    dsl_policy,
    permits_hidden_shapes,
    requires_explicit_dsl,
)
from aka.workflow.light.long_horizon.review import validate_light_production_review
from aka.workflow.light.orchestration.session_io import (
    SessionResult,
    _production_review_candidate_paths,
    _production_review_digest,
    _render,
    sandbox_directive,
)
from aka.workflow.light.orchestration.workspace_state import (
    git_head,
    git_path_blob,
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
    provided_bootstrap: dict | None = None
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

    @property
    def campaign_name(self) -> str:
        suffix = f"_{self.workspace_suffix}" if self.workspace_suffix else ""
        return f"{self.name}{suffix}"

    @property
    def verify_run_timeout(self) -> int:
        return self.evaluation_step_timeout

    @property
    def dsl_context(self) -> str:
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

    @property
    def evaluator_reference_dir(self) -> Path | None:
        """Return the trusted Native evaluator contract for Runtime only."""
        if not self.atrex_bench_root:
            return None
        op_dir = Path(self.kernel_demo).resolve().parent
        required = ("reference.py", "input.py", "shapes.json")
        return op_dir if all((op_dir / name).is_file() for name in required) else None

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
        return (
            "When Wiki is enabled, use `python3 tools/sandbox.py --kind wiki-query` "
            "to find relevant mechanisms and implementations. Describe the missing fact, "
            "then inspect the returned paths, line numbers and snippets for applicability."
        )

    def start_runtime(self) -> None:
        from aka.workflow.light.orchestration.supervisor_runtime import RuntimeConfig

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
            evaluator_reference_dir=self.evaluator_reference_dir,
            atrex_bench_root=Path(self.atrex_bench_root).resolve() if self.atrex_bench_root else None,
            wiki_reference_paths=self.agent_reference_paths,
            wiki_profile_root=self.workspace / ".gpu_wiki_profile", task_id=self.campaign_name,
            optimization_mode=self.optimization_mode,
            workspace_policy=workspace_policy_block(
                self.optimization_mode,
                self.framework,
                source_checks_enabled=self.source_checks_enabled,
                specify_dsl=self.specify_dsl,
            ) + self._bootstrap_directive(),
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
        self._supervisor_runtime.claim_review_enabled = self.module_options.claim_review
        self._supervisor_runtime.research_claim_reviewer = self.review_research_claim

    def review_research_claim(
        self,
        evidence_root: Path,
        *,
        resume_session_id: str | None = None,
        archive_root: Path | None = None,
    ) -> dict[str, object]:
        source = Path(evidence_root).resolve(strict=True)
        from aka.core.filesystem import read_regular_bytes
        target = json.loads(read_regular_bytes(source / "submissions" / "target.json"))
        identity = target.get("direction_id") or target.get("experiment_id")
        if not identity:
            raise ValueError("Reviewed claim requires a stable Direction/Experiment ID")
        root = review_root(supervisor_measurement_root(self.workspace), identity)
        with reviewer_workspace(root) as workspace:
            shutil.copytree(source, workspace, dirs_exist_ok=True)
            return self._review_research_claim_in_workspace(
                workspace, resume_session_id=resume_session_id, archive_root=archive_root,
            )

    def _review_research_claim_in_workspace(
        self, evidence_root: Path, *, resume_session_id: str | None = None,
        archive_root: Path | None = None,
    ) -> dict[str, object]:
        """Run the strict three-tier reviewer with the Qwen-only provider policy."""
        from aka.workflow.light.orchestration.session_tail import read_regular_bytes
        from aka.workflow.light.orchestration.infrastructure_retry import check_review_service, ReviewerRetryableFailure
        from aka.workflow.light.long_horizon.claim_review import validate_review

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
        # The public Agent owner applies the selected Campaign sandbox to
        # this independent evidence workspace, just as for other Reviews.
        base_environment = {
            **boundary_environment,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        }
        base_environment.update(reviewer_environment(evidence_root.parent))
        runtime = getattr(self, "_supervisor_runtime", None)
        capture_base = (Path(archive_root) if archive_root is not None
                        else evidence_root.parent / "attempts" / uuid.uuid4().hex) / "captures"
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
            if runtime is not None and hasattr(runtime, "audit_service_exchange"):
                try:
                    target = json.loads(read_regular_bytes(
                        evidence_root / "submissions" / "target.json",
                        limit=2 * 1024 * 1024,
                    ))
                except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
                    target = None
                runtime.audit_service_exchange(
                    service="independent_claim_reviewer",
                    attempt=attempt,
                    request={
                        "target": target,
                        "evidence_manifest": evidence_manifest,
                    },
                )
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
            from aka.workflow.light.long_horizon.review_usage import account_review
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
            from aka.agent.session_capture import latest_complete_session_archive

            archive = latest_complete_session_archive(attempt_capture)
            if archive is not None and archive_root is not None:
                from aka.workflow.light.orchestration.session_tail import read_regular_bytes
                from aka.experiment.measurement_store import private_write_bytes

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
            # Preserve the artifact even if the process timed out or failed.
            # Execution completion still gates acceptance; this is audit only.
            if response.is_file() and not response.is_symlink():
                try:
                    raw_review = read_regular_bytes(response, limit=128 * 1024 + 1)
                    if len(raw_review) <= 128 * 1024:
                        from aka.experiment.measurement_store import private_write_bytes
                        private_write_bytes(attempt_capture / "review.json", raw_review)
                        validate_review(json.loads(raw_review))
                        attempt_audit["schema_valid_artifact"] = True
                    else:
                        attempt_audit["schema_valid_artifact"] = False
                except (OSError, ValueError, TypeError, KeyError):
                    attempt_audit["schema_valid_artifact"] = False
            if archive is None:
                from aka.workflow.light.orchestration.infrastructure_retry import ReviewerOutputUnavailable

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
                check_review_service(result)
                value = json.loads(read_regular_bytes(response, limit=128 * 1024 + 1))
                if response.stat().st_size > 128 * 1024:
                    raise ValueError("research claim review exceeds 128 KiB")
                review = validate_review(value)
                if runtime is not None and hasattr(runtime, "audit_service_exchange"):
                    runtime.audit_service_exchange(
                        service="independent_claim_reviewer",
                        attempt=attempt,
                        response=review,
                    )
                if archive is not None and archive_root is not None:
                    from aka.workflow.light.orchestration.session_tail import read_regular_bytes
                    from aka.experiment.measurement_store import private_write_bytes

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
                if conversation_path is not None:
                    attempt_audit["status"] = "accepted"
                return {
                    **review,
                    "_review_session_audit": session_audit(),
                }
            except (OSError, ValueError, TypeError, KeyError, ReviewerRetryableFailure) as error:
                last_error = error
                attempt_audit["error"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
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
        submission_revision: int = 1,
        resume_session_id: str | None = None,
    ) -> dict[str, object] | list[str]:
        """Review exact frozen candidate bytes and preserve observed implementation facts."""
        if not self.source_checks_enabled:
            return []
        from aka.task.source.formats import install

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
                review_kwargs["submission_revision"] = submission_revision
                review_kwargs["resume_session_id"] = resume_session_id
            violations = self._production_kernel_violations(
                frozen, **review_kwargs
            )
        policy = canonical_submission_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        )
        implementation_review: dict[str, object] = {
            "availability": (
                "not_applicable" if policy == "fast" else "unavailable"
            ),
        }
        policy_review: dict[str, object] = {
            "verdict": "not_applicable" if policy == "fast" else "unavailable",
            "checks": [],
            "dependencies": [],
            "summary": (
                "Fast does not run the independent final-source Reviewer."
                if policy == "fast"
                else "The independent final-source Review is unavailable."
            ),
        }
        if review_records:
            record = review_records[-1]
            policy_result = record.get("policy_result")
            policy_result = (
                policy_result if isinstance(policy_result, dict) else {}
            )
            if policy_result:
                policy_review = dict(policy_result)
            implementation_review = {
                "availability": "available",
                "dsl_policy": record.get("dsl_policy"),
                "configured_dsl": record.get("framework"),
                "required_phase": record.get("required_phase"),
            }
        return {
            "violations": violations,
            "source_contract": (
                "pass" if not violations or review_records else "fail"
            ),
            "review": policy_review,
            "implementation_review": implementation_review,
        }

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
        from aka.workflow.light.orchestration.supervisor_runtime import OWNER_ENV
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
        """Keep the public evaluator contract outside Kernel-only Agent Git."""
        if self.private_reference_dir is None:
            return ""
        head = git_head(self.workspace)
        if head and git_path_blob(self.workspace, head, AGENT_PROBLEM_FILENAME):
            return "agent_problem.json must not be tracked by the V0 Kernel commit"
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
        submission_revision: int,
        review_record: dict[str, object],
        *,
        conversation: Path | None = None,
        token_usage: Path | None = None,
    ) -> list[str]:
        from aka.workflow.light.long_horizon.review_archive import archive_review

        record_path = review_revision_root(
            supervisor_measurement_root(workspace),
            submission_id,
            submission_revision,
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
        submission_revision: int = 1,
        resume_session_id: str | None = None,
        record_workspace: Path | None = None,
        review_record_sink=None,
    ) -> list[str]:
        """Delegate complete candidate policy review to a fresh, isolated agent."""
        from aka.workflow.light.orchestration.infrastructure_retry import check_review_service

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
            previous_revision = review_record.get("submission_revision", 1)
            previous_root = review_revision_root(
                supervisor_measurement_root(self.workspace),
                str(previous_submission_id),
                int(previous_revision),
            )
            reused_record = {
                **review_record,
                "submission_id": submission_id,
                "submission_revision": submission_revision,
                "review_execution": "reused",
                "reused_from_submission_id": previous_submission_id,
                "reused_from_submission_revision": previous_revision,
            }
            if previous_submission_id != submission_id:
                # Verdict reuse does not transfer another Review family's
                # native Home or resumable session ownership.
                reused_record.pop("session_id", None)
            persistence_errors = self._persist_production_review_record(
                self.workspace,
                submission_id,
                submission_revision,
                reused_record,
                conversation=previous_root / "conversation.jsonl",
                token_usage=previous_root / "token-usage.json",
            )
            return list(dict.fromkeys([*cached_errors, *persistence_errors]))

        errors: list[str]
        review_payload: object | None = None
        policy_result: dict[str, object] | None = None
        review_summary = ""
        effective_dsl_policy = dsl_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        )
        pinned_dsl = effective_dsl_policy == "pinned"
        runtime = getattr(self, "_supervisor_runtime", None)
        identity = submission_id or "review-" + uuid.uuid4().hex
        family_root = review_root(supervisor_measurement_root(self.workspace), identity)
        revision_root = review_revision_root(supervisor_measurement_root(self.workspace), identity, submission_revision)
        # Each retry has its own capture; failed attempts remain auditable.
        capture_root = revision_root / "captures" / uuid.uuid4().hex
        with reviewer_workspace(family_root) as review_workspace:
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

            for name, data in getattr(self, "_provided_bootstrap_files", {}).items():
                original = review_workspace / "bootstrap" / name
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(data)
            request = {
                "schema_version": DEPENDENCY_REVIEW_SCHEMA_VERSION,
                "framework": framework if pinned_dsl else None,
                "recommended_framework": (
                    framework
                    if effective_dsl_policy == "guided" and framework
                    else None
                ),
                "dsl_policy": effective_dsl_policy,
                "compute_policy": "provided_bootstrap_or_self_authored" if self.provided_bootstrap else "self_authored_only",
                "provided_bootstrap": self.provided_bootstrap,
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
                "submission_id": submission_id,
                "submission_revision": submission_revision,
            }
            review_exchange_id = uuid.uuid4().hex
            if runtime is not None and hasattr(runtime, "audit_service_exchange"):
                runtime.audit_service_exchange(
                    service="independent_final_source_reviewer",
                    exchange_id=review_exchange_id,
                    request={
                        "review_request": request,
                        "candidate_manifest": {
                            "files": [
                                {"path": path, "sha256": source_hashes[path]}
                                for path in sorted(source_hashes)
                            ],
                        },
                    },
                )
            (review_workspace / "review_request.json").write_text(
                json.dumps(request, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            try:
                review_environment = {
                    **self.agent_boundary_environment("production-review"),
                    "ATREX_SESSION_CAPTURE_DIR": str(capture_root),
                    **reviewer_environment(family_root),
                }
                result = self._run_session(
                    review_workspace,
                    DEPENDENCY_REVIEW_PROMPT.read_text(encoding="utf-8"),
                    timeout=None,
                    agent_cli=self.agent_cli,
                    reasoning_effort="high",
                    agent_plugins=False,
                    extra_environment=review_environment,
                    session_id=resume_session_id,
                    resume_session=bool(resume_session_id),
                )
                from aka.workflow.light.long_horizon.review_usage import account_review
                account_review(
                    self,
                    result,
                    review_kind="final_source_policy_review",
                    label="independent final-source policy review",
                    backend=self.agent_cli,
                    persist_separate_receipt=False,
                )
                from aka.agent.session_capture import latest_complete_session_archive
                from aka.workflow.light.orchestration.infrastructure_retry import ReviewerOutputUnavailable

                session_archive = latest_complete_session_archive(capture_root)
                # Archive execution (including failure) before validating the
                # verdict. A rejected/unavailable reviewer is still auditable.
                invocation_record = {
                    "review_execution": "executed", "accepted": False,
                    "availability": "unavailable", "agent_runtime": self.agent_cli,
                    "session_id": result.session_id,
                    "resumed_from_session_id": resume_session_id,
                    "submission_id": submission_id, "submission_revision": submission_revision,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "exit_status": result.exit_status, "timed_out": result.timed_out,
                    "stdout_tail": result.stdout_tail, "stderr_tail": result.stderr_tail,
                }
                from aka.workflow.light.long_horizon.review_archive import archive_review
                archive_review(
                    revision_root, result=invocation_record,
                    conversation=session_archive.conversation if session_archive else None,
                    token_usage=session_archive.token_usage if session_archive else None,
                )
                raw_verdict = review_workspace / "dependency_review.json"
                if raw_verdict.is_file() and not raw_verdict.is_symlink():
                    from aka.workflow.light.long_horizon.review_archive import _copy_regular
                    _copy_regular(raw_verdict, capture_root / "dependency_review.json")
                if session_archive is None:
                    raise ReviewerOutputUnavailable(
                        "independent final-source policy reviewer did not persist "
                        "a complete conversation.jsonl and token-usage.json archive"
                    )
                if (resume_session_id and result.exit_status != 0 and self.agent_cli == "codex"
                        and "no rollout found for thread id" in result.stderr_tail):
                    raise ReviewerOutputUnavailable(
                        "Codex reviewer resume storage is unavailable: " + result.stderr_tail
                    )
                check_review_service(result)
            except ValueError as exc:
                if runtime is not None and hasattr(runtime, "audit_service_exchange"):
                    runtime.audit_service_exchange(
                        service="independent_final_source_reviewer",
                        exchange_id=review_exchange_id,
                        response={
                            "status": "unavailable",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        },
                    )
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
            review_path = review_workspace / "dependency_review.json"
            try:
                from aka.core.filesystem import read_regular_bytes
                raw_review = read_regular_bytes(review_path, limit=128 * 1024 + 1)
                if len(raw_review) > 128 * 1024:
                    raise ValueError("final-source review exceeds 128 KiB")
                review_payload = json.loads(raw_review)
                if runtime is not None and hasattr(runtime, "audit_service_exchange"):
                    runtime.audit_service_exchange(
                        service="independent_final_source_reviewer",
                        exchange_id=review_exchange_id,
                        response=review_payload,
                    )
                if changed:
                    errors = [
                        "independent final-source policy review modified candidate evidence: "
                        + ", ".join(sorted(changed))
                    ]
                else:
                    errors, review_summary, policy_result = validate_light_production_review(
                        review_payload,
                        candidate_files=frozenset(source_hashes),
                        dsl_policy=effective_dsl_policy,
                        allow_provided_bootstrap=bool(self.provided_bootstrap),
                    )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                from aka.workflow.light.orchestration.infrastructure_retry import ReviewerOutputUnavailable

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
            "compute_policy": "provided_bootstrap_or_self_authored" if self.provided_bootstrap else "self_authored_only",
            "provided_bootstrap": self.provided_bootstrap,
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
            "policy_result": policy_result,
            "review_execution": "executed",
            "session_id": result.session_id,
        }
        if resume_session_id is not None:
            review_record["resumed_from_session_id"] = resume_session_id
        if submission_id is not None:
            review_record["submission_id"] = submission_id
            review_record["submission_revision"] = submission_revision
            persistence_errors = self._persist_production_review_record(
                self.workspace,
                submission_id,
                submission_revision,
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
        submission_revision: int = 1,
        resume_session_id: str | None = None,
    ) -> list[str]:
        """Bound timeout retries across sessions and supervisor restarts."""
        from aka.workflow.light.orchestration.infrastructure_retry import retry_review

        # Review the exact submitted source. In particular, do not synthesize
        # solution.json from a configured DSL: it is not part of that commit
        # and could invent dependencies or conceal a submitted manifest error.
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
        next_resume_id = resume_session_id

        def run_review():
            nonlocal next_resume_id
            resume_id, next_resume_id = next_resume_id, None
            return self._review_production_candidate_once(
                workspace, framework, require_gluon, candidate_digest=digest,
                submission_id=submission_id, submission_revision=submission_revision,
                resume_session_id=resume_id, review_record_sink=review_record_sink,
            )

        try:
            return retry_review(
                self.workspace, stage,
                run_review,
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
        submission_revision: int = 1,
        resume_session_id: str | None = None,
    ) -> list[str]:
        if not self.source_checks_enabled:
            return []
        if canonical_submission_policy(
            self.optimization_mode,
            specify_dsl=self.specify_dsl,
        ) == "fast":
            return production_structure_violations(
                workspace or self.workspace,
                self.framework,
                require_gluon=False,
                specify_dsl=False,
            )
        production_reviewer = self._review_production_candidate
        if review_record_sink is not None:
            production_reviewer = lambda candidate, framework, gluon: (
                self._review_production_candidate(
                    candidate,
                    framework,
                    gluon,
                    review_record_sink=review_record_sink,
                    submission_id=submission_id,
                    submission_revision=submission_revision,
                    resume_session_id=resume_session_id,
                )
            )
        elif submission_id is not None:
            production_reviewer = lambda candidate, framework, gluon: (
                self._review_production_candidate(
                    candidate,
                    framework,
                    gluon,
                    submission_id=submission_id,
                    submission_revision=submission_revision,
                    resume_session_id=resume_session_id,
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
        from aka.execution.config import evaluator_num_gpus
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
        from aka.workflow.light.long_horizon.main_adapter import link_episode_runtime
        self._assert_generalized_inputs_are_private()
        link_episode_runtime(self, self.workspace)

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
                "Use --kind evaluate --mode full --no-sync for the complete workload. "
                "Correctness-only, custom-input or subset results do not qualify as full Evaluate evidence for submit-kernel. "
                + profile_shapes + " Use tools/sandbox.py --help for available diagnostic tools. "
                "Never substitute a custom timing/correctness harness for official evaluation."
            )
        if is_sol_op(Path(self.kernel_demo).resolve().parent):
            return (
                "## Evaluation contract: SOL-ExecBench\n\n"
                "The Supervisor runs the official evaluator over workload.jsonl. "
                "Minimize GeoMean workload latency. Use --kind evaluate --no-sync; never replace the evaluator. "
                "Atrex-Bench typed --mode controls do not apply to SOL-ExecBench."
            )
        raise ValueError("A canonical Atrex-Bench or SOL evaluator is required")

    def _sandbox_directive(self) -> str:
        directive = sandbox_directive(
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_ssh,
        )
        tools = getattr(getattr(getattr(self, "request", None), "runtime", None),
                        "agate_tools", ("env", "evaluate", "command"))
        from aka.contracts.runtime_tools import runtime_tools
        available = runtime_tools(tools, research_modules=self.research_modules,
                                  submissions=True, wiki=self.tool_runtime.wiki)
        return directive + "\nEnabled Runtime tools: " + ", ".join(available) + ". Use python3 tools/sandbox.py --help for discovery.\n"


    def _bootstrap_directive(self) -> str:
        from aka.workflow.light.provided_bootstrap import directive
        return directive(self.provided_bootstrap)

    def _mode_directive(self) -> str:
        return optimization_mode_directive(self.optimization_mode, self.framework,
                                           source_checks_enabled=self.source_checks_enabled,
                                           specify_dsl=self.specify_dsl) + self._bootstrap_directive()

    def run(self) -> str:
        """Run the native long-horizon episode supervisor for this campaign."""
        from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
        from aka.workflow.light.long_horizon.session import LongSessionRunner

        # LongHorizonCampaign.run initializes source without measuring it before constructing its
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
