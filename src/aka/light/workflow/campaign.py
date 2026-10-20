"""Application adapter: public module inputs, private light Campaign mechanics."""
from dataclasses import fields

from aka.contracts.agent import AgentRunRequest
from aka.contracts.workspace import SourceRequest
from aka.light.orchestrator.campaign import Campaign
from aka.light.orchestrator.constants import REPO_ROOT
from aka.light.orchestrator.session_io import SessionResult


class ManagedCampaign(Campaign):
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    @classmethod
    def create(cls, request, problem, workflow, evaluation):
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
            verify_run_timeout=evaluation.verify_run_timeout,
            min_improvement_pct=evaluation.min_improvement_pct,
            problem_generation_timeout=min(settings.setup_timeout, 1800),
            v1_ask_codex="codex" in settings.framework_baseline_reviewers,
            v1_ask_qoder="qoder" in settings.framework_baseline_reviewers,
            agent_sandbox=workflow.agent.options.sandbox,
            bwrap_executable=workflow.agent.options.bwrap_executable,
            agent_read_only_paths=workflow.agent.options.read_only_paths)
        campaign.request = request
        campaign.problem = problem
        campaign.agent = workflow.agent
        campaign.source_provider = workflow.source
        campaign.candidate_workspace = workflow.candidate
        campaign.module_options = workflow.options
        campaign.tool_runtime = workflow.tool_runtime
        return campaign

    @property
    def workspace(self):
        return self.request.workspace

    @property
    def private_reference_dir(self):
        return self.problem.evaluation.private_reference_dir

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
        from aka.light.orchestrator.environment_recovery import EnvironmentUnavailable as LightEnvironmentUnavailable

        try:
            from aka.light.runtime.paths import supervisor_campaign_root
            search_root = supervisor_campaign_root(self.workspace) / "search"
            if self.module_options.search is not None:
                if not self.atrex_bench_root or self.sandbox_ssh:
                    raise ValueError("Pooled search requires native Atrex-Bench Eval/ABBA; SOL/SSH search is not supported")
                from .search import PooledSearch
                from .search_driver import EpisodeSearchDriver
                driver = EpisodeSearchDriver(self, search_root)
                try:
                    reason = PooledSearch(search_root, self.module_options.search, driver,
                                          token_budget=self.token_budget).run()
                    return self._finish(reason)
                finally:
                    driver.close()
            if (search_root / "state.json").exists():
                raise ValueError("This workspace uses pooled search; restore its search configuration to resume")
            return super().run()
        except LightEnvironmentUnavailable as exc:
            # The existing Application owns the recovery monitor. Translate at
            # this compatibility boundary, not by replacing either module.
            from orchestrator.environment_recovery import EnvironmentUnavailable

            raise EnvironmentUnavailable(str(exc)) from exc

    def _run_session(self, workspace, prompt, timeout, agent_cli="claude",
                     reasoning_effort="max", extra_environment=None, **legacy):
        environment = dict(extra_environment or {})
        # Keep remote credentials/authority out of the selected Agent; its
        # session boundary obtains only a scoped ToolHost capability.
        result = self.agent.runtime(agent_cli).run(AgentRunRequest(
            workspace, prompt, timeout, reasoning_effort=reasoning_effort,
            environment=environment))
        return SessionResult(result.exit_status, result.timed_out,
            result.terminal_usage.total_tokens or 0, result.stdout_tail, result.stderr_tail,
            result.session_id, result.terminal_usage, result.events, result.capabilities,
            result.observation_errors)

    def setup_baseline(self):
        initial = self.problem.initial_source
        if self.module_options.source_tree:
            if initial.kind == "sol" or not self.atrex_bench_root or self.sandbox_ssh:
                raise ValueError("Source trees require native Atrex-Bench Eval, not SOL/SSH")
            from aka.contracts.content import SourceContent, WorkspaceFile
            from aka.light.workspace.sources import initial_snapshot, unpack, POLICY
            import json
            source = initial_snapshot(self.workspace, self.module_options.source_tree)
            policy, files = unpack(source)
            content = SourceContent(tuple(WorkspaceFile(name, text=data.decode()) for name, data in files.items()) +
                                    (WorkspaceFile(POLICY, text=json.dumps(policy)),))
        else:
            content = self.source_provider.prepare(SourceRequest(
                initial=initial, name=self.campaign_name, framework=self.framework,
                platform=self.platform))
        self._selected_initial_kernel = next(
            (entry.source.read_bytes() if entry.source is not None else entry.text.encode()
             for entry in content.files if entry.path == "kernel.py"), None)
        if self._selected_initial_kernel is None:
            raise ValueError("Selected SourceProvider must supply kernel.py")
        if initial.kind == "sol":
            self.candidate_workspace.prepare_sol(self.workspace, content,
                problem_files=self.problem.agent.files, reference_dir=REPO_ROOT / "reference")
            self.candidate_workspace.commit_sol(self.workspace)
            self._link_runtime()
            from aka.light.orchestrator.workspace_state import v0_baseline_commit
            result = self._run_v0_evaluator_with_correctness_prefetch()
            self._finalize_v0_measurement(result, v0_baseline_commit(self.workspace),
                                         extra_paths=("CLAUDE.md", ".gitignore"))
            return
        self.candidate_workspace.initialize_at(self.workspace, content, REPO_ROOT / "reference")
        self.candidate_workspace.install_files(self.workspace, self.problem.agent.files)
        self._ensure_agent_problem()
        self._link_runtime()
        self._install_native_evaluator()
        self._install_profile_driver()
        self._setup_baseline_native(self.problem.evaluation.directory,
                                   generalized=self.private_reference_dir is not None)
