"""Application adapter: public module inputs, private light Campaign mechanics."""
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import tempfile

from aka.contracts.agent import AgentRunRequest
from aka.contracts.workspace import SourceRequest
from aka.light.workflow.bootstrap import Campaign
from aka.light.workflow.session_io import SessionResult


class ManagedCampaign(Campaign):
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    @staticmethod
    def _source_request(request, problem):
        return SourceRequest(
            initial=problem.initial_source,
            name=request.name,
            framework=(
                request.settings.framework
                if request.settings.specify_dsl
                else "Agent-selected supported GPU DSL"
            ),
            platform=request.settings.platform,
        )

    @staticmethod
    def _content_files(content) -> dict[str, bytes]:
        files = {}
        for entry in content.files:
            if entry.path in files:
                raise ValueError(f"Initial source contains duplicate path: {entry.path}")
            files[entry.path] = (
                entry.source.read_bytes()
                if entry.source is not None
                else entry.text.encode("utf-8")
            )
        return files

    @classmethod
    def _prepare_campaign_source(
        cls, request, problem, workflow, *, state_root: Path
    ):
        initial = problem.initial_source
        native_eval = initial.kind != "sol" and problem.evaluation.runtime_root is not None
        options = workflow.candidate.source_options(workflow.options.source_tree)
        if initial.kind == "sol":
            if options is not None:
                raise ValueError(
                    "Source trees require native Atrex-Bench Eval, not SOL"
                )
            if workflow.candidate.file_system not in (None, "single_file"):
                raise ValueError("SOL requires the single_file candidate file system")
            from aka.light.workspace.sources import select

            system = select("single_file")
        else:
            system = workflow.candidate.select_file_system(
                framework=request.settings.framework,
                source_tree=options is not None,
            )
        content = workflow.candidate.prepare_source(
            system,
            cls._source_request(request, problem),
            workflow.source,
            options=options,
            state_root=state_root,
            native_eval=native_eval,
        )
        files = cls._content_files(content)
        if system.name == "source_tree":
            try:
                snapshot = json.loads(
                    (state_root / "snapshot.json").read_text(encoding="utf-8")
                )
                contract = snapshot["contract"]
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    "Initial source-tree snapshot did not produce a valid contract"
                ) from error
        else:
            contract = system.default_policy
        system.validate_policy(contract)
        system.validate(files, contract)
        from aka.light.workspace.sources import pack

        packed = pack(files, contract)
        return content, packed, {
            "file_system": system.name,
            "contract": contract,
        }

    @classmethod
    def create(cls, request, problem, workflow, evaluation):
        from aka.light.runtime.paths import initialize_campaign_root
        from aka.bootstrap.continuation import launch_composition, read

        payload = read(os.environ)
        if payload is None:
            raise RuntimeError(
                "AKA Light Campaign creation requires the frozen launch composition"
            )

        campaign_root = Path(request.workspace).expanduser().resolve()
        campaign_root.parent.mkdir(parents=True, exist_ok=True)
        # SourceProvider codecs may need a bounded staging directory while they
        # normalize the initial tree and derive its immutable contract.  This
        # is preparation state, not Campaign evidence: the authoritative V0
        # source is archived under measurements/kernels/<Agent-commit>/ after
        # its full Evaluate succeeds.
        with tempfile.TemporaryDirectory(
            prefix=f".{campaign_root.name}-source-",
            dir=campaign_root.parent,
        ) as staged_directory:
            source_state_root = Path(staged_directory) / "initial-source"
            initial_content, initial_source, source_binding = (
                cls._prepare_campaign_source(
                    request,
                    problem,
                    workflow,
                    state_root=source_state_root,
                )
            )
            campaign_root = initialize_campaign_root(
                campaign_root,
                name=request.name,
                launch_composition=launch_composition(payload),
                source=source_binding,
            )
        # The user-selected workspace is the Campaign container.  Candidate
        # Git exists only in lineage Agent workspaces.
        agent_workspace = (
            campaign_root / "lineages" / "t0001" / "agent-workspace"
        )
        request = replace(request, workspace=agent_workspace)
        settings = request.settings
        common = {f.name: getattr(settings, f.name) for f in fields(Campaign)
                  if f.init and hasattr(settings, f.name)}
        transport = {"sandbox_" + name: getattr(evaluation, name) for name in
                     ("hardware", "profile", "url", "timeout", "ssh", "ssh_init", "ssh_gpu", "health_command")}
        campaign = cls(**common, **transport, name=request.name,
            kernel_demo=str(problem.initial_source.entry),
            work_dir=str(request.workspace.parent),
            atrex_bench_root=str(problem.evaluation.runtime_root or ""),
            verify_repeats=evaluation.verify_repeats,
            evaluation_step_timeout=evaluation.evaluation_step_timeout,
            shape_batch_size=evaluation.shape_batch_size,
            gpu_clock_mhz=evaluation.gpu_clock_mhz,
            min_improvement_pct=evaluation.min_improvement_pct,
            problem_generation_timeout=settings.setup_timeout,
            agent_sandbox=workflow.agent.options.sandbox,
            bwrap_executable=workflow.agent.options.bwrap_executable,
            agent_read_only_paths=workflow.agent.options.read_only_paths)
        campaign.request = request
        campaign.campaign_root = campaign_root
        campaign.problem = problem
        campaign.problem_provider = workflow.problem
        campaign.agent = workflow.agent
        campaign.source_provider = workflow.source
        campaign.candidate_workspace = workflow.candidate
        campaign.module_options = workflow.options
        campaign.tool_runtime = workflow.tool_runtime
        campaign._prepared_initial_content = initial_content
        campaign._selected_initial_source = initial_source
        return campaign

    @property
    def workspace(self):
        return self.request.workspace

    @property
    def private_reference_dir(self):
        return self.problem.evaluation.private_reference_dir

    @property
    def evaluator_reference_dir(self):
        if self.problem.initial_source.kind == "sol":
            return None
        return Path(self.problem.evaluation.directory).expanduser().resolve()

    def prepare(self, *, recovery_ready=None):
        # LongHorizon.run owns the idempotent V0/resume prelude.
        self.workspace.parent.mkdir(parents=True, exist_ok=True)
        if recovery_ready is not None:
            recovery_ready()

    def start_runtime(self):
        cancel = getattr(self, "search_cancel", None)
        if cancel is not None and cancel.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError("Search coordinator stopped this trajectory")
        return super().start_runtime()

    def run(self):
        # EnvironmentUnavailable is shared with WorkflowStartup; preserve the
        # original exception and traceback through both serial and pooled runs.
        from aka.light.runtime.paths import supervisor_campaign_base_root
        from aka.light.workflow.search import COORDINATOR_STATE

        campaign_root = supervisor_campaign_base_root(self.workspace)
        if self.module_options.search is not None:
            if not self.atrex_bench_root:
                raise ValueError("Pooled search requires native Atrex-Bench Eval/ABBA")
            from .search import PooledSearch
            from .search_driver import EpisodeSearchDriver
            driver = EpisodeSearchDriver(self, campaign_root)
            try:
                reason = PooledSearch(campaign_root, self.module_options.search, driver,
                                      token_budget=self.token_budget).run()
                return self._finish(reason)
            finally:
                driver.close()
        if (campaign_root / COORDINATOR_STATE).exists():
            raise ValueError("This workspace uses pooled search; restore its search configuration to resume")
        return super().run()

    def _run_session(self, workspace, prompt, timeout, agent_cli="claude",
                     reasoning_effort="max", extra_environment=None, **legacy):
        environment = dict(extra_environment or {})
        # Every auxiliary Agent invocation gets a durable Campaign-private
        # transcript even when its caller did not request a more specific
        # Review archive. Explicit Review capture roots continue to win.
        if "ATREX_SESSION_CAPTURE_DIR" not in environment:
            import re
            from aka.light.runtime.paths import supervisor_campaign_root

            role = str(environment.get("ATREX_AGENT_WORKSPACE_ROLE") or "auxiliary")
            role = re.sub(r"[^a-zA-Z0-9_.-]+", "-", role).strip("-.") or "auxiliary"
            environment["ATREX_SESSION_CAPTURE_DIR"] = str(
                supervisor_campaign_root(self.workspace)
                / "sessions"
                / role
            )
        # Keep remote credentials/authority out of the selected Agent; its
        # session boundary obtains only a scoped ToolHost capability.
        result = self.agent.runtime(agent_cli).run(AgentRunRequest(
            workspace, prompt, timeout, reasoning_effort=reasoning_effort,
            session_id=legacy.get("session_id"),
            resume_session=bool(legacy.get("resume_session", False)),
            environment=environment))
        return SessionResult(result.exit_status, result.timed_out,
            result.terminal_usage.total_tokens or 0, result.stdout_tail, result.stderr_tail,
            result.session_id, result.terminal_usage, result.events, result.capabilities,
            result.observation_errors)

    def setup_baseline(self):
        initial = self.problem.initial_source
        content = self._prepared_initial_content
        if initial.kind == "sol":
            self.candidate_workspace.prepare_sol(self.workspace, content,
                problem_files=self.problem.agent.files)
            from aka.light.workspace.sources import Artifact
            Artifact(self._selected_initial_source).install(
                self.workspace, replace=False, bind=True
            )
            self._link_runtime()
            from aka.light.workspace.state import v0_baseline_commit
            source_commit = v0_baseline_commit(self.workspace)
            if not source_commit:
                raise RuntimeError("SOL V0 has no committed Kernel source")
            result = self._run_v0_evaluator(source_commit)
            self._finalize_v0_measurement(result, source_commit)
            return
        self.candidate_workspace.initialize_at(self.workspace, content)
        from aka.light.workspace.sources import Artifact
        Artifact(self._selected_initial_source).install(
            self.workspace, replace=False, bind=True
        )
        self.candidate_workspace.install_files(self.workspace, self.problem.agent.files)
        self._ensure_agent_problem()
        self._link_runtime()
        self._setup_baseline_native(self.problem.evaluation.directory,
                                   generalized=self.private_reference_dir is not None)
