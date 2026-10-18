"""Single-operator optimization campaign: baseline setup, episode loop, promotion policy."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Optional


from aka.contracts.problem import LoadedProblem
from aka.contracts.evaluation import EvaluationUnavailable
from aka.legacy.agent.runtime import auth_hint

from aka.legacy.workflow.constants import SUPPLEMENTAL_PENDING_PREFIX, SUPPLEMENTAL_REPAIR_PREFIX, ATREX_PRIVATE_REFERENCE_ENV, DEFAULT_CONVERT_AFTER, DEFAULT_FAST_EPISODES, DEFAULT_FAST_TRIALS, DEFAULT_HANDOFF_RESUMES, DEFAULT_VERIFY_REPEATS, DEFAULT_VERIFY_RUN_TIMEOUT, DEPENDENCY_REVIEW_SCHEMA_VERSION, DEPENDENCY_REVIEW_TIMEOUT_S, FRAMEWORK_BASELINE_CATEGORY, FRAMEWORK_BASELINE_TIMEOUT_S, FRAMEWORK_BASELINE_VERSION
from .resources import DEPENDENCY_REVIEW_PROMPT, PROMPTS_DIR
from aka.legacy.execution.defaults import DEFAULT_SANDBOX_TIMEOUT
from aka.legacy.task.candidate_workspace.filenames import FRAMEWORK_BASELINE_FILE
from aka.legacy.execution.recovery import EnvironmentUnavailable, environment_state_file
from .framework_baseline_progress import (
    capture_unexpected_exit as capture_framework_baseline_exit,
)
from .framework_baseline_progress import (
    capture_terminal_rejection as capture_framework_baseline_rejection,
)
from .framework_baseline_progress import (
    load_progress as load_framework_baseline_progress,
)
from .framework_baseline_progress import (
    mark_accepted as mark_framework_baseline_accepted,
)
from .framework_baseline_progress import (
    progress_path as framework_baseline_progress_path,
)
from .framework_baseline_progress import (
    restore_latest_candidate as restore_latest_framework_baseline_candidate,
)
from .framework_baseline_progress import (
    save_supervisor_recovery as save_framework_baseline_recovery,
)
from aka.legacy.workflow.hardware import hardware_directive
from aka.legacy.task.hardware import hardware_vendor
from aka.legacy.task.source.inspection import kernel_is_gluon
from aka.legacy.task.problem import AGENT_PROBLEM_FILENAME
from aka.legacy.workflow.policy import install_workspace_policy, optimization_mode_directive
from aka.legacy.acceptance.source_policy import production_kernel_violations, production_structure_violations
from .plan_reviewers import (
    REVIEWER_ENVIRONMENT,
    discover_plan_reviewers,
    plan_reviewer_environment,
)
from aka.legacy.execution.configuration import distributed_evaluation_world_size, evaluator_num_gpus
from aka.legacy.resources.plugins import PluginRegistry
from aka.legacy.agent.session import SessionResult
from aka.legacy.acceptance.review import _production_review_candidate_paths, _production_review_digest, _validate_production_review
from aka.legacy.evaluation.local_results import _record_local_test_result
from aka.legacy.resources.prompts import render_prompt
from .runtime_directives import _agent_runtime_directive, _baseline_driver_directive
from aka.legacy.task.candidate_workspace.state import git_head, git_kernel_blob, git_path_blob, git_worktree_blob, head_kernel_is_initial_baseline, latest_version, read_memory, resolve_framework_baseline_commit, v0_baseline_commit, write_stall

_LONG_REVIEWER_SESSION_ENV = {
    "codex": "ATREX_CODEX_REVIEW_SESSION_FILE",
    "qoder": "ATREX_QODER_REVIEW_SESSION_FILE",
}


_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_TIMEOUT_S = 600
_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION = 3
_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_PATH = Path(
    ".atrex_long_horizon/framework_baseline/correctness_guidance.json"
)
_FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS = (
    "README.md",
    "reference.py",
    "input.py",
    "agent_problem.json",
    "shapes.json",
    "definition.json",
    "workload.jsonl",
)
_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS = (
    "SEMANTIC_CHECKLIST:",
    "EDGE_CASES:",
    "CUDA_IMPLEMENTATION_RISKS:",
    "PRE_SMOKE_CHECKS:",
    "TARGETED_REFERENCES:",
    "RECOMMENDED_CORRECTNESS_FIRST_DESIGN:",
)
_FRAMEWORK_BASELINE_REFERENCE_CATALOG_LIMIT = 80
_FRAMEWORK_BASELINE_REFERENCE_PER_PROJECT_LIMIT = 6
_FRAMEWORK_BASELINE_REFERENCE_MIN_SOURCE_BYTES = 1000
_FRAMEWORK_BASELINE_SELECTED_REFERENCE_LIMIT = 2
_FRAMEWORK_BASELINE_REFERENCE_EXTENSIONS = {
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".md",
    ".py",
}


from aka.contracts.workflow import CampaignRequest
from .dependencies import CampaignDependencies


@dataclass
class Campaign:
    request: CampaignRequest
    dependencies: CampaignDependencies
    problem: LoadedProblem
    tokens_spent: int = field(default=0, init=False)
    _production_review_cache: dict[str, tuple[tuple[str, ...], dict[str, object]]] = (
        field(default_factory=dict, init=False, repr=False, compare=False)
    )
    _generated_agent_problem_digest: str = field(
        default="", init=False, repr=False, compare=False
    )
    _plan_reviewer_environment: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            self.problem.initial_source is None
            or self.problem.evaluation is None
        ):
            raise TypeError(
                "Campaign requires explicit Problem initial source and evaluation roles"
            )
        self.name = self.problem.name
        self.agent = self.dependencies.agent
        self.agent_runtime = self.agent.runtime(self.request.settings.agent_cli)
        self.problem_provider = self.dependencies.problem
        self.source_provider = self.dependencies.source
        self.candidate_workspace = self.dependencies.candidate
        self.workflow = self.dependencies.workflow
        for definition in fields(self.request.settings):
            setattr(
                self, definition.name, getattr(self.request.settings, definition.name)
            )
        self.atrex_bench_root = str(self.problem.evaluation.runtime_root or "")
        self.services = self.dependencies.services(self)
        self.evaluation = self.dependencies.evaluation(self)
        self.execution = self.dependencies.execution(self)
        self.plugin_registry.check_lock(self.workspace)
        if self.long_reviewer_session and self.long_reviewer_session not in (
            _LONG_REVIEWER_SESSION_ENV
        ):
            raise NotImplementedError(
                f"long reviewer sessions are not implemented for "
                f"{self.long_reviewer_session}"
            )

    @property
    def campaign_name(self) -> str:
        return self.request.name

    @property
    def private_reference_dir(self) -> Path | None:
        return self._problem_inputs().private_reference_dir

    def _loaded_problem(self):
        return self.problem

    def _problem_inputs(self):
        return self.problem_provider.content(
            self._loaded_problem(),
            self.workspace,
            generated_digest=self._generated_agent_problem_digest,
        )

    def _ensure_agent_problem(self) -> None:
        """Schedule the original authoring session when public input preparation requires it."""
        inputs = self._problem_inputs()
        public = inputs.prepare_public()
        self.candidate_workspace.install_files(self.workspace, public.files)
        if public.ready:
            self._generated_agent_problem_digest = inputs.generated_digest
            return
        destination = self.workspace / AGENT_PROBLEM_FILENAME
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
            self.candidate_workspace.install_files(staging, inputs.authoring_assets())
            for attempt in range(2):
                repair_context = (
                    "The current agent_problem.json failed orchestrator validation. Replace it "
                    f"with a corrected file. Validation error: {validation_error}"
                    if validation_error
                    else "Create agent_problem.json now."
                )
                prompt = render_prompt(
                    self.plugin_registry.resource_result().resource("task-problem-generation-prompt").source,
                    REPAIR_CONTEXT=repair_context,
                )
                result = self._run_session(
                    staging,
                    prompt,
                    timeout=min(self.setup_timeout, 1_800),
                    agent_cli=self.agent_cli,
                    reasoning_effort="max",
                )
                self._account(result, f"agent problem generation attempt {attempt + 1}")
                generated = staging / AGENT_PROBLEM_FILENAME
                try:
                    inputs.validate_generated(generated)
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
                self.candidate_workspace.install_files(
                    self.workspace, (inputs.generated_file(generated),)
                )
                inputs.record_generated()
                self._generated_agent_problem_digest = inputs.generated_digest
                print(
                    f"[orchestrator] generalized problem: generated {destination}",
                    flush=True,
                )
                return

    @cached_property
    def plugin_registry(self) -> PluginRegistry:
        return PluginRegistry()

    def plugin_directive(self, phase: str) -> str:
        registry = self.plugin_registry
        registry.check_lock(self.workspace)
        return registry.instructions(
            phase,
            PLATFORM=self.platform,
            ARCH=self.arch or "<exact runtime architecture>",
            FRAMEWORK=self.framework,
            OPERATOR=self.name,
        )

    def _episode_plan_reviewers(self, episode_mode: str) -> tuple[str, ...]:
        if episode_mode == "goal":
            episode_mode = "full"
        if episode_mode not in ("fast", "full"):
            raise ValueError(f"unsupported episode mode: {episode_mode}")
        return tuple(
            reviewer
            for reviewer in ("codex", "qoder")
            if reviewer in getattr(self, f"{episode_mode}_episode_reviewers")
        )

    def _framework_baseline_correctness_reviewers(self) -> tuple[str, ...]:
        return tuple(
            reviewer
            for reviewer, enabled in (
                ("codex", "codex" in self.framework_baseline_reviewers),
                ("qodercli", "qoder" in self.framework_baseline_reviewers),
            )
            if enabled
        )

    def agent_environment(self, *, episode_mode: str = "") -> dict[str, str]:
        private_dir = self.private_reference_dir
        environment = dict(self._plan_reviewer_environment)
        if episode_mode:
            environment["ATREX_EPISODE_MODE"] = episode_mode
            enabled_reviewers = set(self._episode_plan_reviewers(episode_mode))
            reviewer_mode = "full" if episode_mode == "goal" else episode_mode
            for reviewer, (enabled_name, reason_name) in REVIEWER_ENVIRONMENT.items():
                if reviewer in enabled_reviewers:
                    continue
                environment[enabled_name] = "0"
                environment[reason_name] = (
                    f"disabled by --no-{reviewer_mode}-episode-ask-{reviewer}"
                )
        if self.long_reviewer_session:
            env_name = _LONG_REVIEWER_SESSION_ENV[self.long_reviewer_session]
            state_file = (
                self.workspace
                / f".atrex_long_horizon/{self.long_reviewer_session}_reviewer_session.json"
            )
            environment[env_name] = str(state_file.resolve())
        if private_dir is not None:
            environment[ATREX_PRIVATE_REFERENCE_ENV] = str(private_dir)
        environment.update(self.execution.environment())
        if self.execution.uses_ssh:
            environment.pop("ATREX_SANDBOX_URL", None)
            environment.pop("ATREX_SANDBOX_PROFILE", None)
        state_file = environment_state_file()
        if state_file is not None:
            environment["ATREX_ENVIRONMENT_STATE_FILE"] = str(state_file)
        environment.update(
            self.plugin_registry.environment(self.workspace, self.campaign_name)
        )
        environment.update(self.services.task_environment())
        return environment

    def ensure_plan_reviewer_availability(self, *, episode_mode: str) -> None:
        """Probe the reviewers enabled for this episode mode at most once each."""
        reviewers = self._episode_plan_reviewers(episode_mode)
        if not reviewers:
            return
        if all(
            REVIEWER_ENVIRONMENT[reviewer][0] in self._plan_reviewer_environment
            for reviewer in reviewers
        ):
            return
        value, reused = discover_plan_reviewers(
            self.workspace,
            agent_cli=self.agent_cli,
            reviewers=reviewers,
        )
        self._plan_reviewer_environment = plan_reviewer_environment(value)
        statuses = []
        for name in reviewers:
            record = value["reviewers"][name]
            status = "available" if record["available"] else "disabled"
            statuses.append(f"{name}={status} ({record['reason']})")
        source = "cached" if reused else "startup probe"
        print(
            f"[orchestrator] {episode_mode} episode plan reviewers ({source}): "
            + "; ".join(statuses),
            flush=True,
        )

    def _generalized_memory_coverage_problem(self, memory: dict | None) -> str:
        """Require successful canonical memory to cover every private shape by opaque id."""
        private_dir = self.private_reference_dir
        if private_dir is None or memory is None:
            return ""
        performance = memory.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        measured = performance.get("latency_us_by_shape")
        measured = measured if isinstance(measured, dict) else {}
        expected_ids = set(self._problem_inputs().evaluation.case_ids)
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
        self._problem_inputs().assert_private()

    @property
    def workspace(self) -> Path:
        return self.request.workspace

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
    ) -> list[str]:
        """Delegate complete candidate policy review to a fresh, isolated agent."""
        from aka.legacy.agent.availability import check_review_service

        candidate_digest = _production_review_digest(
            workspace, framework, require_gluon
        )
        cache_key = candidate_digest + ":" + self.agent_cli
        cached = self._production_review_cache.get(cache_key)
        if cached is not None:
            cached_errors, review_record = cached
            persistence_errors = self._persist_production_review_record(
                workspace,
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
                "framework": framework,
                "required_phase": "gluon" if require_gluon else "selected_framework",
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
                    timeout=self.production_review_timeout,
                    agent_cli=self.agent_cli,
                    reasoning_effort="high",
                    agent_plugins=False,
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
                    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
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
                    errors = [
                        "independent production policy review produced no valid verdict: "
                        f"{type(exc).__name__}: {exc}"
                    ]
                else:
                    status = "accepted" if not errors else "rejected"
                    print(
                        f"[production-policy] independent full-candidate review {status}: "
                        f"{review_summary}",
                        flush=True,
                    )

        try:
            reviewed_digest = _production_review_digest(
                workspace, framework, require_gluon
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
            "framework": framework,
            "required_phase": "gluon" if require_gluon else "selected_framework",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "agent_runtime": self.agent_cli,
            "accepted": not errors,
            "errors": errors,
            "summary": review_summary,
            "review": review_payload,
        }
        persistence_errors = self._persist_production_review_record(
            workspace,
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
        from aka.legacy.workflow.review_retry import retry_review

        digest = _production_review_digest(workspace, framework, require_gluon)
        stage = f"dependency-review:{digest}:{self.agent_cli}:{self.production_review_timeout}"
        try:
            return retry_review(
                workspace,
                stage,
                lambda: self._review_production_candidate_once(
                    workspace, framework, require_gluon
                ),
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
            production_reviewer=self._review_production_candidate,
        )

    @cached_property
    def source_checks_enabled(self) -> bool:
        """Freeze policy from trusted inputs, never candidate worktree metadata."""
        if self.execution.uses_ssh:
            return True
        reference = self.private_reference_dir
        if reference is not None:
            return evaluator_num_gpus(reference) < 2
        baseline = v0_baseline_commit(self.workspace) if self.workspace.is_dir() else ""
        if baseline:
            # Resume from the immutable V0, including the absence of metadata.
            # Reading HEAD or the live file would let a candidate disable its gate.
            if not git_path_blob(self.workspace, baseline, "metadata.json"):
                return True
            result = subprocess.run(
                ["git", "show", f"{baseline}:metadata.json"],
                cwd=str(self.workspace), capture_output=True, encoding="utf-8",
                check=True,
            )
            try:
                metadata = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON in V0 metadata.json") from exc
            if not isinstance(metadata, dict):
                raise ValueError("V0 metadata.json must contain a JSON object")
            return distributed_evaluation_world_size(metadata) < 2
        # Before V0 exists, the Task-owned evaluation snapshot is authoritative.
        return evaluator_num_gpus(self.problem.evaluation.directory) < 2

    def _link_runtime(self) -> None:
        from aka.legacy.workflow.store import CampaignStore
        from .environment import install_campaign_resources

        self._assert_generalized_inputs_are_private()
        CampaignStore.ensure_excluded(self.workspace, wiki_trace_only=True)
        install_campaign_resources(self, self.workspace)
        install_workspace_policy(
            self.workspace,
            self.optimization_mode,
            self.framework,
            agent_runtime=self.agent_cli,
            source_checks_enabled=self.source_checks_enabled,
        )

    def _sandbox_directive(self) -> str:
        return self.execution.directive()

    def _fast_sandbox_directive(self) -> str:
        return self.execution.directive(fast=True)

    def _mode_directive(self) -> str:
        return optimization_mode_directive(
            self.optimization_mode, self.framework,
            source_checks_enabled=self.source_checks_enabled,
        )

    def setup_baseline(self) -> None:
        # SOL-ExecBench op: seed a correct, directly-submittable V0 mechanically
        # (no baseline session) — Task installs the ground-truth files, writes
        # the DPS wrapper kernel.py + solution.json; this method benches V0 in the sandbox.
        problem = self._loaded_problem()
        op_dir = problem.evaluation.directory
        if problem.initial_source.kind == "sol":
            self.evaluation.setup_baseline_sol(op_dir)
            return
        from aka.contracts.workspace import SourceRequest
        from .resources import initial_files

        content = self.source_provider.prepare(SourceRequest(initial=problem.initial_source))
        code = self.candidate_workspace.initialize_kernel(
            self.workspace, content, runtime_files=initial_files(self.plugin_registry))
        if code:
            raise RuntimeError(f'Task workspace initialization failed with code {code}')
        # Production native tasks always expose a generalized public contract. Exact shapes and
        # release metadata remain in the source operator directory and are injected only at the
        # sandbox boundary. A missing public contract is authored before the baseline session.
        inputs = self._problem_inputs()
        generalized = inputs.private_reference_dir is not None
        self.candidate_workspace.install_files(self.workspace, inputs.operator_files())
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
        self.evaluation.install_native_evaluator()
        self.evaluation.install_profile_driver()
        # A native Atrex-Bench V0 is already materialized as the verbatim reference
        # wrapper.  Its evaluator and memory schemas are also supervisor-owned, so an
        # Agent session adds no implementation value here.  Commit the sources, run the
        # one required remote measurement, and record the result mechanically.  The
        # derived legacy boundary below retains the Agent fallback because its harness
        # and input layout are not canonical enough to synthesize safely.
        if self.atrex_bench_root:
            self.evaluation.setup_baseline_native(op_dir, generalized=generalized)
            return
        prompt = render_prompt(
            PROMPTS_DIR / "setup.md",
            PLUGINS=self.plugin_directive("setup"),
            WORKSPACE=str(self.workspace),
            PLATFORM=self.platform,
            FRAMEWORK=self.framework,
            KERNEL_DEMO="reference.py",
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            BASELINE_DRIVER=_baseline_driver_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._sandbox_directive(),
            EVALUATOR=self.evaluation.evaluator_directive(),
            MODE_POLICY=self._mode_directive(),
        )
        res = self._run_session(
            self.workspace,
            prompt,
            timeout=self.setup_timeout,
            agent_cli=self.agent_cli,
            reasoning_effort="high",
            environment=self.agent_environment(),
            remote=True,
        )
        self._assert_generalized_inputs_are_private()
        self._account(res, "setup")
        if res.exit_status != 0 and res.tokens == 0:
            raise RuntimeError(
                f"setup session failed immediately (exit={res.exit_status}, tokens=0) — "
                "this is likely an API key / authentication issue. "
                f"{auth_hint(self.agent_cli)}."
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
                "workspace workload exactly once with the base seed through the mandatory sandbox "
                "with `--no-memory`; do not run `--multi-seed` for V0. Parse its "
                "`[test_kernel] RESULT_JSON=...`, write local `memory/v0.json` and `baseline_report.md`, "
                "then Git commit `V0: baseline kernel`. Do not enter optimization iterations.\n\n"
                + self.evaluation.evaluator_directive()
                + "\n\n"
                + self._sandbox_directive()
            )
            recovery = self._run_session(
                self.workspace,
                recovery_prompt,
                timeout=self.setup_timeout,
                agent_cli=self.agent_cli,
                reasoning_effort="high",
                environment=self.agent_environment(),
                remote=True,
            )
            self._assert_generalized_inputs_are_private()
            self._account(recovery, "setup recovery")
            if recovery.exit_status != 0 and recovery.tokens == 0:
                raise RuntimeError(
                    f"setup recovery failed immediately (exit={recovery.exit_status}, tokens=0) — "
                    f"{auth_hint(self.agent_cli)}."
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

    def _run_v0_evaluator_with_correctness_prefetch(self) -> dict:
        """Measure V0 while prefetching the source-only V1 correctness review.

        The review packet deliberately excludes V0 measurement artifacts, so its
        digest is stable before and after the evaluator writes memory/v0.json and
        baseline_report.md. Review failure remains non-fatal here: the ordinary V1
        entry point validates the cache and retries synchronously when necessary.
        """
        if not self._framework_baseline_correctness_reviewers():
            return self.evaluation.run_v0_evaluator()
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
                return self.evaluation.run_v0_evaluator()
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
        self.evaluation.ensure_framework_baseline_solution_manifest()

        if action == "run":
            self._link_runtime()
            self._restore_framework_baseline_candidate(v0_blob)
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
            if res.exit_status != 0 and res.tokens == 0:
                raise RuntimeError(
                    "the framework baseline Agent exited without usable output; "
                    f"inspect {framework_baseline_progress_path(self.workspace)}"
                )
            recovery_used = False
            problem = self.evaluation.framework_baseline_problem(
                v0_blob, baseline_commit, include_policy_review=False
            )
            if problem:
                self._recover_framework_baseline(
                    problem, v0_blob, baseline_commit, pre_head
                )
                recovery_used = True
                self._warn_restored_baseline_paths(baseline_commit)
                problem = self.evaluation.framework_baseline_problem(
                    v0_blob, baseline_commit, include_policy_review=False
                )
        else:  # adopt: our own interrupted run already committed the kernel
            recovery_used = False
            self._sync_framework_baseline_live(phase="framework_baseline")
            self._warn_restored_baseline_paths(baseline_commit)
            problem = self.evaluation.framework_baseline_problem(
                v0_blob, baseline_commit, include_policy_review=False
            )
        result: Optional[dict] = None
        if not problem:
            result, problem = self._framework_baseline_external_gates(n)
        numerical_repairs = 0
        while (
            problem
            and not problem.startswith(SUPPLEMENTAL_PENDING_PREFIX)
            and (
                not recovery_used
                or (
                    problem.startswith(SUPPLEMENTAL_REPAIR_PREFIX)
                    and numerical_repairs < 2
                )
            )
        ):
            # The implementation Agent intentionally runs only a bounded smoke subset.
            # Ordinary failures retain one repair turn. Supplemental counterexamples
            # get their own bounded repair budget even if bring-up already used it.
            if problem.startswith(SUPPLEMENTAL_REPAIR_PREFIX):
                numerical_repairs += 1
            self._recover_framework_baseline(
                problem, v0_blob, baseline_commit, pre_head
            )
            recovery_used = True
            self._warn_restored_baseline_paths(baseline_commit)
            problem = self.evaluation.framework_baseline_problem(
                v0_blob, baseline_commit, include_policy_review=False
            )
            if not problem:
                result, problem = self._framework_baseline_external_gates(n)
        if problem:
            # Undo any session-authored commits while retaining every worktree file.  A hard reset
            # here used to erase the exact candidate and debugging experience needed by a restart.
            if pre_head and git_head(self.workspace) != pre_head:
                subprocess.run(
                    ["git", "reset", "--mixed", pre_head],
                    cwd=str(self.workspace),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._warn_restored_baseline_paths(baseline_commit)
            self._capture_framework_baseline_rejection(problem)
            self._record_framework_baseline_failure(problem)
            self._sync_framework_baseline_live(
                phase="failed",
                state="blocked",
                accepted=False,
                outcome={"summary": problem, "next_directions": []},
            )
            outcome = (
                "validation blocked"
                if problem.startswith(SUPPLEMENTAL_PENDING_PREFIX)
                else "rejected"
            )
            raise RuntimeError(f"framework baseline v{n} {outcome}: {problem}")

        commit = self._commit_framework_baseline(n, result or {})
        try:
            mark_framework_baseline_accepted(
                self.workspace,
                commit=commit,
                latency_us=(result or {}).get("latency_us_geomean"),
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

    def _framework_baseline_external_gates(self, n: int):
        try:
            return self.evaluation.framework_baseline_external_gates(n)
        except EvaluationUnavailable as exc:
            status = "rejected" if exc.failure.candidate_repairable else "validation_blocked"
            self._capture_framework_baseline_rejection(
                exc.failure.message, status=status
            )
            raise

    def _capture_framework_baseline_rejection(
        self, problem: str, *, status: str | None = None
    ) -> None:
        if problem.startswith(SUPPLEMENTAL_PENDING_PREFIX):
            reason = "supplemental-validation-pending"
            status = status or "validation_blocked"
        elif problem.startswith(SUPPLEMENTAL_REPAIR_PREFIX):
            reason = "supplemental-candidate-failure"
            status = status or "rejected"
        elif "self-contained" in problem:
            reason = "production-policy-rejected"
            status = status or "rejected"
        elif "combined validation" in problem:
            reason = "combined-validation-failed"
            status = status or "rejected"
        elif "candidate request rejected" in problem:
            reason = "candidate-request-rejected"
            status = status or "rejected"
        else:
            reason = "framework-baseline-rejected"
            status = status or "rejected"
        try:
            capture_framework_baseline_rejection(
                self.workspace,
                version=FRAMEWORK_BASELINE_VERSION,
                status=status,
                reason=reason,
                diagnostic=self.evaluation.framework_baseline_rejection_diagnostic(
                    problem
                ),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                "[orchestrator] WARNING: could not retain terminal framework-baseline "
                f"diagnostic: {type(exc).__name__}: {exc}",
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
            from aka.legacy.workflow.journal import sync_live_memory
            from aka.legacy.workflow.store import CampaignStore

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

        progressed = not head_kernel_is_initial_baseline(self.workspace)
        dirty_candidate = git_worktree_blob(
            self.workspace, "kernel.py"
        ) != git_path_blob(self.workspace, "HEAD", "kernel.py")
        restart_journal = framework_baseline_progress_path(self.workspace).is_file()
        # A compliant dirty candidate must never cause V0 itself to be pinned.  It is an
        # interrupted V1 worktree and needs validation/recovery first.
        if restart_journal:
            if self.resume_framework_candidate:
                return "adopt", "validate the retained framework candidate"
            return "run", "resuming the unexpected-exit V1 handoff"
        if not progressed and dirty_candidate:
            if self.resume_framework_candidate:
                return "adopt", "validate the retained framework candidate"
            return "run", "resuming preserved framework-baseline work"

        if self.framework_baseline == "auto" and self.optimization_mode != "production":
            return (
                "skip",
                "leaderboard mode keeps the permissive V0 (use --framework-baseline always)",
            )
        if not progressed:
            if not self.source_checks_enabled:
                return "pin", "multi-GPU Agate source checks are disabled; retain the validated V0"
            structural = production_structure_violations(self.workspace, self.framework)
            if any(
                value.startswith("unsupported production framework")
                for value in structural
            ):
                return "skip", "; ".join(structural)
            baseline_commit = self._v0_baseline_commit()
            reference_blob = git_path_blob(
                self.workspace, baseline_commit, "reference.py"
            )
            known_reference_wrapper = (
                bool(
                    reference_blob
                    and reference_blob
                    == git_path_blob(self.workspace, baseline_commit, "kernel.py")
                )
                or self._loaded_problem().initial_source.kind == "sol"
            )
            if known_reference_wrapper:
                return (
                    "run",
                    f"the supervisor-seeded V0 is the PyTorch reference wrapper, not "
                    f"a self-contained {self.framework} kernel",
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
        if self.resume_framework_candidate:
            return "adopt", "validate the retained framework candidate"
        return "skip", (
            "HEAD has progressed beyond V0 without a framework-baseline pin; "
            "leaving this campaign on its existing baseline"
        )

    def _v0_baseline_commit(self) -> str:
        commit = v0_baseline_commit(self.workspace)
        if not commit:
            raise RuntimeError("framework baseline requires a committed V0 kernel.py")
        return commit

    def _framework_baseline_supervisor_order(self) -> tuple[str, ...]:
        """Runtime order for the post-exit progress-saving supervisor only."""
        ordered: list[str] = []
        for agent_cli in (self.agent_cli, "codex", "qodercli"):
            if agent_cli not in ordered:
                ordered.append(agent_cli)
        return tuple(ordered)

    def _save_framework_baseline_exit_progress(
        self,
        *,
        crash_progress: dict,
    ) -> None:
        """Ask a separate read-only Agent to turn crash artifacts into a restart handoff."""
        exits = crash_progress.get("unexpected_exits")
        if not isinstance(exits, list) or not exits or not isinstance(exits[-1], dict):
            return
        crash_record = exits[-1]
        snapshot = crash_record.get("snapshot")
        snapshot_root = (
            self.workspace / str(snapshot.get("root"))
            if isinstance(snapshot, dict) and snapshot.get("root")
            else None
        )
        order = self._framework_baseline_supervisor_order()
        prompt = (PROMPTS_DIR / "framework_baseline_exit_supervisor.md").read_text(
            encoding="utf-8"
        )
        for index, supervisor_cli in enumerate(order):
            with tempfile.TemporaryDirectory(
                prefix="atrex-v1-exit-supervisor-"
            ) as directory:
                review_workspace = Path(directory)
                if snapshot_root is not None and snapshot_root.is_dir():
                    shutil.copytree(snapshot_root, review_workspace / "candidate")
                (review_workspace / "crash_record.json").write_text(
                    json.dumps(crash_record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(
                    "[orchestrator] V1 exit progress supervisor: launching "
                    f"{supervisor_cli} ({index + 1}/{len(order)})",
                    flush=True,
                )
                try:
                    result = self._run_session(
                        review_workspace,
                        prompt,
                        timeout=600,
                        agent_cli=supervisor_cli,
                        reasoning_effort="high",
                        agent_plugins=False,
                    )
                except Exception as exc:
                    print(
                        "[orchestrator] V1 exit progress supervisor failed: "
                        f"{supervisor_cli}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                self._account(
                    result, f"framework baseline exit supervisor ({supervisor_cli})"
                )
                if result.exit_status != 0 or result.timed_out:
                    continue
                try:
                    recovery = json.loads(
                        (review_workspace / "resume.json").read_text(encoding="utf-8")
                    )
                    destination = save_framework_baseline_recovery(
                        self.workspace, recovery, agent_cli=supervisor_cli
                    )
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    print(
                        "[orchestrator] V1 exit progress supervisor produced an invalid "
                        f"handoff via {supervisor_cli}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                print(
                    f"[orchestrator] V1 restart handoff saved: {destination}",
                    flush=True,
                )
                return
        print(
            "[orchestrator] WARNING: no progress supervisor produced a structured V1 "
            f"handoff; mechanical crash record remains at {framework_baseline_progress_path(self.workspace)}",
            file=sys.stderr,
            flush=True,
        )

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
        order = self._framework_baseline_supervisor_order()
        progress = capture_framework_baseline_exit(
            self.workspace,
            agent_cli=self.agent_cli,
            exit_status=exit_status,
            timed_out=timed_out,
            tokens=tokens,
            session_id=session_id,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=error,
            supervisor_order=order,
        )
        self._save_framework_baseline_exit_progress(crash_progress=progress)

    def _framework_baseline_correctness_context(self) -> list[Path]:
        """Return the bounded public packet shown to the two V1 correctness reviewers."""
        context: list[Path] = []
        for relative in _FRAMEWORK_BASELINE_CORRECTNESS_CONTEXT_PATHS:
            path = self.workspace / relative
            if path.is_file():
                context.append(path)
        return context

    def _framework_baseline_reference_keywords(self) -> set[str]:
        """Derive bounded path-ranking terms from the public operator contract."""
        keywords = {
            token
            for token in re.findall(
                r"[a-z0-9]+", f"{self.name} {self.framework} {self.arch}".lower()
            )
            if len(token) >= 3
        }
        public_text = ""
        for relative in ("agent_problem.json", "reference.py", "input.py"):
            path = self.workspace / relative
            if not path.is_file():
                continue
            try:
                public_text += (
                    "\n" + path.read_text(encoding="utf-8", errors="replace").lower()
                )
            except OSError:
                continue
        for term in (
            "attention",
            "backward",
            "bwd",
            "causal",
            "decode",
            "gqa",
            "mask",
            "mla",
            "nvrtc",
            "paged",
            "prefill",
            "ragged",
            "softmax",
            "varlen",
        ):
            if term in public_text:
                keywords.add(term)
        if self.arch.lower().startswith("sm_12"):
            keywords.update({"blackwell", "sm120"})
        if hardware_vendor(self.platform, self.arch) == "ppu":
            keywords.update({"ppu", "sail", "hggc", "actlize", "m890"})
        return keywords

    def _framework_baseline_reference_catalog(self) -> list[str]:
        """Rank a small exact-path catalog; reviewers may select only from this list."""
        # Knowledge stores are queried through enabled tools, not scanned as raw files.
        selected = self.plugin_registry.resource_result().resources
        sources = {}
        for resource in selected:
            if not resource.name.startswith('workflow-reference-') or resource.destination is None:
                continue
            for member in resource.files:
                relative = (Path(resource.destination)/member.path).as_posix()
                sources[relative] = resource.source/member.path
        candidates = tuple(sources)

        keywords = self._framework_baseline_reference_keywords()
        targets_ppu = hardware_vendor(self.platform, self.arch) == "ppu"
        wants_backward = bool(keywords & {"backward", "bwd"})
        ranked: list[tuple[int, str]] = []
        for raw_path in candidates:
            relative = raw_path
            source = sources[relative]
            suffix = source.suffix.lower()
            if (
                suffix not in _FRAMEWORK_BASELINE_REFERENCE_EXTENSIONS
                or not source.is_file()
            ):
                continue
            if suffix != ".md":
                try:
                    if (
                        source.stat().st_size
                        < _FRAMEWORK_BASELINE_REFERENCE_MIN_SOURCE_BYTES
                    ):
                        continue
                except OSError:
                    continue
            lowered = relative.lower().replace("-", "_")
            score = sum(10 for keyword in keywords if keyword in lowered)
            if score == 0:
                continue
            if relative.startswith("reference-projects/") and suffix != ".md":
                score += 4
            if "/sources/prs/" in lowered:
                score -= 5
            if not wants_backward and any(
                term in lowered for term in ("_bwd", "bwd_", "backward")
            ):
                score -= 15
            if any(term in lowered for term in ("paged", "varlen", "gqa", "sm120")):
                score += 3
            if any(term in lowered for term in ("_for_sail", "actlize", "hggc", "ppu")):
                if not targets_ppu:
                    continue
                score += 3
            ranked.append((score, relative))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        per_project: dict[str, int] = {}
        catalog: list[str] = []
        for _score, relative in ranked:
            if len(catalog) >= _FRAMEWORK_BASELINE_REFERENCE_CATALOG_LIMIT:
                break
            project = "/".join(relative.split("/")[:2])
            if (
                per_project.get(project, 0)
                >= _FRAMEWORK_BASELINE_REFERENCE_PER_PROJECT_LIMIT
            ):
                continue
            per_project[project] = per_project.get(project, 0) + 1
            catalog.append(relative)
        return catalog

    def _framework_baseline_reference_catalog_text(self) -> str:
        catalog = self._framework_baseline_reference_catalog()
        lines = [
            "# Bounded implementation reference catalog",
            "",
            "Select at most two exact paths from this list. A path is navigational evidence only; "
            "do not claim facts about file contents you have not read.",
            "",
        ]
        lines.extend(f"- `{path}`" for path in catalog)
        return "\n".join(lines) + "\n"

    def _framework_baseline_review_references(
        self, guidance: str
    ) -> list[dict[str, str]]:
        catalog = set(self._framework_baseline_reference_catalog())
        references: list[dict[str, str]] = []
        in_section = False
        for line in guidance.splitlines():
            stripped = line.strip()
            if stripped == "TARGETED_REFERENCES:":
                in_section = True
                continue
            if in_section and stripped.endswith(":") and not stripped.startswith("-"):
                break
            if not in_section:
                continue
            match = re.fullmatch(r"- path: ([^|]+?)\s*\|\s*purpose: (.+)", stripped)
            if match is None:
                continue
            path = match.group(1).strip().strip("`")
            purpose = " ".join(match.group(2).split())[:500]
            if path not in catalog or any(item["path"] == path for item in references):
                continue
            references.append({"path": path, "purpose": purpose})
            if len(references) >= _FRAMEWORK_BASELINE_SELECTED_REFERENCE_LIMIT:
                break
        return references

    def _framework_baseline_selected_references(
        self, reviews: dict[str, object]
    ) -> list[dict[str, object]]:
        """Choose at most two reviewer-nominated paths, preferring consensus then rank."""
        catalog = self._framework_baseline_reference_catalog()
        rank = {path: index for index, path in enumerate(catalog)}
        nominations: dict[str, dict[str, object]] = {}
        for reviewer in self._framework_baseline_correctness_reviewers():
            record = reviews.get(reviewer)
            if not isinstance(record, dict):
                continue
            references = record.get("references")
            if not isinstance(references, list):
                continue
            for item in references:
                if not isinstance(item, dict) or item.get("path") not in rank:
                    continue
                path = str(item["path"])
                current = nominations.setdefault(
                    path,
                    {"path": path, "purpose": str(item.get("purpose", "")), "votes": 0},
                )
                current["votes"] = int(current["votes"]) + 1
        ordered = sorted(
            nominations.values(),
            key=lambda item: (-int(item["votes"]), rank[str(item["path"])]),
        )
        return ordered[:_FRAMEWORK_BASELINE_SELECTED_REFERENCE_LIMIT]

    def _framework_baseline_correctness_context_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            f"v1-correctness-review-v{_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_SCHEMA_VERSION}".encode()
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
        digest.update(b"\0reference-catalog\0")
        digest.update(self._framework_baseline_reference_catalog_text().encode())
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
        selected_references = value.get("selected_references")
        catalog = set(self._framework_baseline_reference_catalog())
        if (
            not isinstance(selected_references, list)
            or len(selected_references) > _FRAMEWORK_BASELINE_SELECTED_REFERENCE_LIMIT
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or item["path"] not in catalog
                or not isinstance(item.get("purpose"), str)
                or not isinstance(item.get("votes"), int)
                for item in selected_references
            )
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
                catalog_path = context_root / "reference_catalog.md"
                catalog_path.parent.mkdir(parents=True, exist_ok=True)
                catalog_path.write_text(
                    self._framework_baseline_reference_catalog_text(),
                    encoding="utf-8",
                )
                source_hashes["reference_catalog.md"] = hashlib.sha256(
                    catalog_path.read_bytes()
                ).hexdigest()

                prompt = render_prompt(
                    PROMPTS_DIR / "framework_baseline_correctness_review.md",
                    REVIEWER=label,
                    FRAMEWORK=self.framework,
                    PLATFORM=self.platform,
                    ARCH=self.arch or "the runtime GPU architecture",
                )
                result = self._run_session(
                    review_workspace,
                    prompt,
                    timeout=_FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_TIMEOUT_S,
                    agent_cli=agent_cli,
                    reasoning_effort="max",
                    agent_plugins=False,
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
                            "reason": "malformed guidance; missing "
                            + ", ".join(missing),
                            "session_id": result.session_id,
                        },
                        result,
                    )
                return (
                    {
                        "status": "ok",
                        "guidance": guidance[:16000],
                        "references": self._framework_baseline_review_references(
                            guidance
                        ),
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
            "selected_references": self._framework_baseline_selected_references(
                reviews
            ),
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

    @staticmethod
    def _framework_baseline_guidance_without_reference_nominations(
        guidance: str,
    ) -> str:
        """Hide raw nominations so V1 sees only the supervisor's final shortlist."""
        lines: list[str] = []
        skipping = False
        for line in guidance.splitlines():
            stripped = line.strip()
            if stripped == "TARGETED_REFERENCES:":
                skipping = True
                continue
            if skipping and stripped in _FRAMEWORK_BASELINE_CORRECTNESS_REVIEW_MARKERS:
                skipping = False
            if not skipping:
                lines.append(line)
        return "\n".join(lines).strip()

    def _framework_baseline_correctness_guidance_text(self) -> str:
        reviewers = self._framework_baseline_correctness_reviewers()
        if not reviewers:
            return (
                "## External pre-implementation correctness guidance\n\n"
                "External V1 correctness review is disabled by configuration. Derive every "
                "correctness requirement directly from the public contract and immutable "
                "reference before editing.\n"
            )
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
            + " reviewed the bounded public packet. Before editing `kernel.py`, reconcile the "
            "available guidance against the immutable reference. Shared requirements are "
            "mandatory; reviewer suggestions never override the public contract.\n",
        ]
        for reviewer in reviewers:
            label = "Qoder" if reviewer == "qodercli" else "Codex"
            record = reviews.get(reviewer)
            record = record if isinstance(record, dict) else {}
            sections.append(f"\n### {label}\n")
            if record.get("status") == "ok":
                guidance = (
                    self._framework_baseline_guidance_without_reference_nominations(
                        str(record.get("guidance", ""))
                    )
                )
                sections.append(guidance + "\n")
            else:
                sections.append(
                    "Reviewer unavailable: "
                    + str(record.get("reason", "no valid response"))
                    + "\n"
                )
        sections.append("\n### Supervisor-selected implementation references\n")
        selected_references = value.get("selected_references")
        if isinstance(selected_references, list) and selected_references:
            for item in selected_references:
                assert isinstance(item, dict)
                sections.append(
                    f"- `{item['path']}` — {item['purpose']} "
                    f"(nominated by {item['votes']}/{len(reviewers)} configured reviewers)\n"
                )
            sections.append(
                "Read only these exact files as static design evidence. Do not open sibling "
                "files, follow imports or links recursively, execute/import the reference, "
                "delegate computation to it, or copy a prebuilt implementation. Raw reviewer "
                "nominations were reconciled and are not additional authorization.\n"
            )
        else:
            sections.append(
                "- None selected. Do not broaden research unless the bounded fallback in Step B "
                "is needed for framework/toolchain syntax.\n"
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
        """Run the configured outer V1 Agent; other CLIs only summarize an exit."""
        prompt = self._framework_baseline_prompt(n)
        if rejection:
            prompt = (
                "# Repair the preserved V1 candidate\n\n"
                f"The supervisor rejected the current candidate: **{rejection}**\n"
                "Keep sound work, fix this exact rejection, and finish V1. If a prior targeted "
                "smoke passed but the supervisor's combined full-workload validation failed, "
                "you may run exactly one full base-seed evaluator while repairing; do not run "
                "multi-seed or a separate benchmark because the supervisor will repeat the "
                "combined authoritative gate.\n\n" + prompt
            )
        if framework_baseline_progress_path(self.workspace).is_file():
            prompt = (
                "# Resume V1 from an unexpected-exit handoff\n\n"
                "Do not start over. Read "
                "`.atrex_long_horizon/framework_baseline/resume.json` when present, then inspect "
                "the existing candidate and debug files. Continue at the recorded `next_step` and "
                "avoid repeating completed research.\n\n" + prompt
            )
        try:
            result = self._run_session(
                self.workspace,
                prompt,
                timeout=self.framework_baseline_timeout,
                agent_cli=self.agent_cli,
                reasoning_effort="max",
                environment=self.agent_environment(),
                remote=True,
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
                supervisor_order=self._framework_baseline_supervisor_order(),
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
                supervisor_order=self._framework_baseline_supervisor_order(),
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

    def _restore_framework_baseline_candidate(self, v0_blob: str) -> None:
        """Restore an ignored snapshot only when the worktree fell back to the V0 wrapper."""
        if git_worktree_blob(self.workspace, "kernel.py") != v0_blob:
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
        if restored and git_worktree_blob(self.workspace, "kernel.py") != v0_blob:
            print(
                "[orchestrator] restored interrupted V1 candidate from local checkpoint: "
                + ", ".join(restored),
                flush=True,
            )

    def _framework_baseline_sandbox_directive(self) -> str:
        return self.execution.framework_baseline_directive()

    def _framework_baseline_prompt(self, n: int) -> str:
        smoke_command, smoke_scope = self.evaluation.framework_baseline_smoke_command(n)
        return render_prompt(
            PROMPTS_DIR / "framework_baseline.md",
            PLUGINS=self.plugin_directive("framework_baseline"),
            WORKSPACE=str(self.workspace),
            N=n,
            PREV=n - 1,
            PLATFORM=self.platform,
            FRAMEWORK=self.framework,
            ARCH=self.arch or "the runtime GPU arch",
            NOTES=self.notes,
            AGENT_RUNTIME=_agent_runtime_directive(self.agent_cli),
            HARDWARE=hardware_directive(self.platform, self.arch),
            SANDBOX=self._framework_baseline_sandbox_directive(),
            EVALUATOR=self.evaluation.evaluator_directive(),
            CORRECTNESS_GUIDANCE=self._framework_baseline_correctness_guidance_text(),
            SMOKE_COMMAND=smoke_command,
            SMOKE_SCOPE=smoke_scope,
            MODE_POLICY=self._mode_directive(),
        )

    def _warn_restored_baseline_paths(self, baseline_commit: str) -> None:
        restored = self.evaluation.restore_immutable_baseline_paths(baseline_commit)
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
        """Run one recovery session; unexpected exits invoke the progress supervisor."""
        print(
            f"[orchestrator] framework baseline needs repair ({problem}); "
            "starting one recovery session",
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
            self._run_framework_baseline_agent(
                n=FRAMEWORK_BASELINE_VERSION,
                label="framework baseline recovery",
                rejection=problem,
            )
        finally:
            self._warn_restored_baseline_paths(baseline_commit)

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
        validation_incomplete = problem.startswith(SUPPLEMENTAL_PENDING_PREFIX)
        memory["correctness"] = {
            "status": "UNKNOWN" if validation_incomplete else "FAIL"
        }
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
                "error_type": (
                    "validation_incomplete"
                    if validation_incomplete
                    else (
                        "production_policy"
                        if "self-contained" in problem
                        else "correctness"
                    )
                ),
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
            "validated_stages": ["combined-base-performance+multi-seed-5"],
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

    def numerical_feedback(self, workspace, *, standard_correctness_passed=False):
        return self.services.numerical_feedback(workspace, standard_correctness_passed=standard_correctness_passed)

    def _run_session(self, workspace, prompt, timeout, agent_cli="claude", **options):
        runtime = (
            self.agent_runtime
            if agent_cli == self.agent_cli
            else self.agent.runtime(agent_cli)
        )
        return self.execution.run_session(
            workspace, prompt, timeout, agent_cli=agent_cli, runtime=runtime, **options
        )

    def prepare(self, *, recovery_ready=None) -> None:
        if latest_version(self.workspace) < 0:
            self.setup_baseline()
        else:
            print(
                f"[orchestrator] resuming workspace at v{latest_version(self.workspace)}",
                flush=True,
            )
            self._link_runtime()
        baseline_coverage_problem = self._generalized_memory_coverage_problem(
            read_memory(self.workspace, 0)
        )
        if baseline_coverage_problem:
            raise RuntimeError(
                "generalized campaign baseline is incompatible with authoritative per-shape "
                f"memory: {baseline_coverage_problem}; start a fresh workspace"
            )
        if recovery_ready is not None:
            recovery_ready()
        self.ensure_framework_baseline()

    def run(self) -> str:
        return self.workflow(self)

    def _finish(self, reason: str) -> str:
        print(f"\n[orchestrator] STOP — {reason}", flush=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m", "aka.legacy.workflow.memory_manager",
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
                        "-m", "aka.legacy.workflow.sol_finalize",
                        "--workspace",
                        str(self.workspace),
                    ],
                    check=False,
                )
            except OSError:
                pass
        return reason
