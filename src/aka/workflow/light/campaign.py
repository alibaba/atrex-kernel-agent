"""Application adapter: public module inputs, private light Campaign mechanics."""
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import tempfile

from aka.contracts.agent import AgentRunRequest
from aka.contracts.workspace import SourceRequest
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.orchestration.session_io import SessionResult


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
        options = workflow.options.source_tree
        if initial.kind == "sol":
            if options is not None:
                raise ValueError(
                    "Source trees require native Atrex-Bench Eval, not SOL"
                )
            if workflow.options.file_system not in (None, "single_file"):
                raise ValueError("SOL requires the single_file candidate file system")
            from aka.task.source.formats import select

            system = select("single_file")
        else:
            from aka.task.source.formats import task_selection
            system = task_selection(workflow.options.file_system,
                framework=request.settings.framework, source_tree=options is not None)
        content = system.prepare_source(
            cls._source_request(request, problem),
            workflow.source,
            options=options,
            state_root=state_root,
            native_eval=native_eval,
        )
        files = cls._content_files(content)
        from .provided_bootstrap import validate_permission
        validate_permission(workflow.options.provided_bootstrap, files)
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
        from aka.task.source.formats import pack

        packed = pack(files, contract)
        return content, packed, {
            "file_system": system.name,
            "contract": contract,
        }

    @classmethod
    def create(cls, request, problem, workflow, evaluation):
        from aka.workflow.light.support.paths import initialize_campaign_root
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
        # source is archived under measurements/kernels/<Agent-commit>/ when
        # its initial Git commit is created; measurement is Agent-controlled.
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
        workflow.candidate.bind_source_contract(campaign_root, **source_binding)
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
            agent_sandbox="bwrap" if workflow.agent.capabilities.filesystem_isolation else "none")
        campaign.request = request
        campaign.campaign_root = campaign_root
        campaign.problem = problem
        campaign.problem_provider = workflow.problem
        campaign.agent = workflow.agent
        campaign.source_provider = workflow.source
        campaign.candidate_workspace = workflow.candidate
        campaign.module_options = workflow.options
        campaign.provided_bootstrap = workflow.options.provided_bootstrap
        campaign._provided_bootstrap_files = (cls._content_files(initial_content)
                                              if campaign.provided_bootstrap else {})
        from .runtime import ManagedRuntimeFactory
        from .runtime_resources import runtime_resource_registry
        campaign.tool_runtime = ManagedRuntimeFactory(workflow, campaign)
        campaign._module_resources = runtime_resource_registry(campaign_root,
            agate_tools=request.runtime.agate_tools, wiki=workflow.options.wiki,
            research_modules=campaign.research_modules)
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
        # Session startup owns idempotent source initialization and recovery.
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
        from aka.workflow.light.support.paths import supervisor_campaign_base_root
        from aka.workflow.light.search import COORDINATOR_STATE

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

    def agent_environment(self):
        # Evaluation/Execution credentials and routing remain in the host.
        from aka.execution.recovery import environment_state_file
        state = environment_state_file()
        return {"ATREX_ENVIRONMENT_STATE_FILE": str(state)} if state is not None else {}

    def agent_access(self, workspace, *, role="optimizer"):
        from pathlib import Path
        from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
        workspace = Path(workspace).resolve(strict=True)
        if role != "optimizer":
            from aka.workflow.light.orchestration.agent_workspace import WORKSPACE_LAYOUTS
            if role not in WORKSPACE_LAYOUTS:
                raise ValueError("Unknown auxiliary session role")
            inputs, _ = WORKSPACE_LAYOUTS[role]
            return FilesystemAccess((FilesystemMount(workspace, workspace, True),), hidden_paths=(workspace / ".git",),
                read_only_paths=tuple(workspace / name for name in inputs if (workspace / name).exists()))
        resources = self._module_resources.filesystem_access(workspace)
        from .runtime_resources import reference_knowledge_access
        knowledge = reference_knowledge_access(workspace, self.agent_reference_paths)
        hidden = list(resources.hidden_paths)
        private = self.private_reference_dir
        if private is not None and private.is_relative_to(workspace):
            hidden.append(private)
        protected = [workspace / name for name in
            ("PROGRESS.md", "records", "research-library", "input", "reference", "frontier.json", "atrex-bench")
            if (workspace / name).exists()]
        protected.extend(resources.read_only_paths)
        protected.extend(knowledge.read_only_paths)
        protected.extend(workspace / item.path for item in self.problem.agent.files)
        for path in self.agent_read_only_paths:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            if candidate.exists():
                protected.append(candidate)
        # A hidden subtree is already inaccessible. Do not also ask Agent to
        # expose its private context store through a read-only overlay.
        protected = [path for path in protected if not any(path.is_relative_to(root) for root in hidden)]
        from aka.contracts.filesystem_access import normalize_access
        return normalize_access(FilesystemAccess(
            (FilesystemMount(workspace, workspace, True), *resources.mounts, *knowledge.mounts),
            tuple(dict.fromkeys(hidden)), tuple(dict.fromkeys(protected))))

    def _run_session(self, workspace, prompt, timeout, agent_cli="claude",
                     reasoning_effort="max", extra_environment=None, observation_directory=None, **legacy):
        from aka.workflow.budgets import remaining_timeout
        timeout = remaining_timeout(timeout)
        environment = dict(extra_environment or {})
        role = environment.pop("ATREX_AGENT_WORKSPACE_ROLE", "optimizer")
        for key in tuple(environment):
            if key.startswith(("ATREX_SANDBOX_", "ATREX_AGENT_")) or key in {"ATREX_BWRAP_EXECUTABLE", "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_AKA_RUNTIME_OWNER"}:
                environment.pop(key, None)
        access = self.agent_access(workspace, role=role)
        def invoke(values, selected_access):
            result = self.agent.runtime(agent_cli).run(AgentRunRequest(
                workspace, prompt, timeout, reasoning_effort=reasoning_effort,
                session_id=legacy.get("session_id"),
                resume_session=bool(legacy.get("resume_session", False)),
                environment=values, access=selected_access, observation_directory=observation_directory))
            return SessionResult(result.exit_status, result.timed_out,
                result.terminal_usage.total_tokens or 0, result.stdout_tail, result.stderr_tail,
                result.session_id, result.terminal_usage, result.events, result.capabilities,
                result.observation_errors)
        if role == "optimizer":
            self.start_runtime()
            with self._supervisor_runtime.session(workspace, environment, access=access) as grant:
                return invoke(grant.environment, grant.filesystem_access)
        return invoke(environment, access)


    def initialize_source(self):
        """Prepare source and tools; initial-source evaluation is Agent-owned."""
        from .initial_source import initialize_source

        initialize_source(self)
        self._ensure_agent_problem()
        self._link_runtime()
