"""Application adapter: public module inputs, private light Campaign mechanics."""
from dataclasses import fields, replace

from aka.contracts.agent import AgentRunRequest
from aka.contracts.workspace import SourceRequest
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.orchestration.constants import REPO_ROOT
from aka.workflow.light.orchestration.session_io import SessionResult


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
            problem_generation_timeout=settings.setup_timeout,
            v1_ask_codex="codex" in settings.framework_baseline_reviewers,
            v1_ask_qoder="qoder" in settings.framework_baseline_reviewers,
            agent_sandbox="bwrap" if workflow.agent.capabilities.filesystem_isolation else "none")
        campaign.request = request
        campaign.problem = problem
        campaign.problem_provider = workflow.problem
        campaign.agent = workflow.agent
        campaign.source_provider = workflow.source
        campaign.candidate_workspace = workflow.candidate
        campaign.episode_evidence_reader = workflow.episode_evidence_reader
        campaign.episode_evidence_interpreter = workflow.episode_evidence_interpreter
        campaign.module_options = workflow.options
        from .runtime import ManagedRuntimeFactory
        campaign.tool_runtime = ManagedRuntimeFactory(workflow, campaign)
        from aka.resources.plugins import PluginRegistry
        campaign._module_resources = PluginRegistry()
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
        from aka.execution.recovery import EnvironmentUnavailable as LightEnvironmentUnavailable

        try:
            from aka.workflow.light.support.paths import supervisor_campaign_root
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
            # WorkflowStartup owns the shared recovery monitor. Translate at
            # this module boundary without reviving the retired Application.
            from aka.execution.recovery import EnvironmentUnavailable

            raise EnvironmentUnavailable(str(exc)) from exc

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
        view = getattr(self, "_supervisor_runtime", None)
        binding = view.journals.get(workspace) if view is not None else None
        authority = binding.view.worktree if binding and binding.view else workspace
        facts = self.candidate_workspace.filesystem_access(authority)
        resources = self._module_resources.filesystem_access(workspace)
        hidden = [workspace / name for name in (".git", ".atrex_long_horizon", ".atrex_versions", ".aka-agent-observations")]
        hidden.extend(resources.hidden_paths)
        private = self.private_reference_dir
        if private is not None and private.is_relative_to(workspace):
            hidden.append(private)
        protected = [workspace / path.relative_to(authority) for path in facts.read_only_paths
            if path.is_relative_to(authority) and (workspace / path.relative_to(authority)).exists()]
        protected.extend(resources.read_only_paths)
        for path in self.agent_read_only_paths:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            if candidate.exists():
                protected.append(candidate)
        return FilesystemAccess((FilesystemMount(workspace, workspace, True), *resources.mounts), tuple(dict.fromkeys(hidden)), tuple(dict.fromkeys(protected)))

    def _run_session(self, workspace, prompt, timeout, agent_cli="claude",
                     reasoning_effort="max", extra_environment=None, observation_directory=None, **legacy):
        import time
        environment = dict(extra_environment or {})
        role = environment.pop("ATREX_AGENT_WORKSPACE_ROLE", "optimizer")
        for key in tuple(environment):
            if key.startswith(("ATREX_SANDBOX_", "ATREX_AGENT_")) or key in {"ATREX_BWRAP_EXECUTABLE", "ATREX_PRIVATE_REFERENCE_DIR", "ATREX_AKA_RUNTIME_OWNER"}:
                environment.pop(key, None)
        access = self.agent_access(workspace, role=role)
        def invoke(values, selected_access):
            result = self.agent.runtime(agent_cli).run(AgentRunRequest(
                workspace, prompt, timeout, reasoning_effort=reasoning_effort,
                environment=values, access=selected_access, observation_directory=observation_directory))
            return SessionResult(result.exit_status, result.timed_out,
                result.terminal_usage.total_tokens or 0, result.stdout_tail, result.stderr_tail,
                result.session_id, result.terminal_usage, result.events, result.capabilities,
                result.observation_errors)
        if role == "optimizer":
            self.start_runtime()
            with self._supervisor_runtime.session(workspace, environment, access=access,
                    deadline=time.monotonic() + timeout) as grant:
                return invoke(grant.environment, grant.filesystem_access)
        return invoke(environment, access)

    def ensure_framework_baseline(self):
        super().ensure_framework_baseline()
        from aka.task.source.formats import POLICY, SELECTION, task_selection
        from aka.task.source.formats.registry import resolve
        from aka.contracts.content import WorkspaceFile
        import json
        source = self.candidate_workspace.read_source(self.workspace)
        requested = task_selection(self.module_options.file_system, framework=self.framework,
                                   source_tree=POLICY in source.files)
        selected, _ = resolve(self.workspace, revision="HEAD")
        if selected.name != requested.name:
            raise ValueError(f"Selected {requested.name} requires a matching candidate; found {selected.name}")
        content = (json.dumps({"name": selected.name}, sort_keys=True) + "\n").encode()
        retained = self.candidate_workspace.read_working(self.workspace, (SELECTION,)).files[SELECTION]
        if retained is not None and retained != content:
            raise ValueError("Candidate file system cannot change when resuming a Campaign")
        self.candidate_workspace.record(self.workspace,
            expected_revision=self.candidate_workspace.current_revision(self.workspace),
            files=(WorkspaceFile(SELECTION, text=content.decode()),), message="Pin candidate file system")

    def setup_baseline(self):
        initial = self.problem.initial_source
        content = self.source_provider.prepare(SourceRequest(
            initial=initial, name=self.campaign_name, framework=self.framework,
            platform=self.platform))
        from aka.task.source.formats import POLICY, task_selection
        imported_tree = any(item.path == POLICY for item in content.files)
        system = task_selection(self.module_options.file_system, framework=self.framework,
                                source_tree=imported_tree)
        if system.name != "single_file" and (initial.kind == "sol" or self.sandbox_ssh):
            raise ValueError("Compiled/source-tree candidates require native Eval")
        if system.name == "source_tree" and not imported_tree:
            from aka.workflow.light.support.paths import supervisor_campaign_root
            content = system.prepare_source(SourceRequest(initial=initial, name=self.campaign_name,
                framework=self.framework, platform=self.platform), self.source_provider,
                options=None, state_root=supervisor_campaign_root(self.workspace) / "initial-source",
                native_eval=bool(self.atrex_bench_root) and not self.sandbox_ssh)
        self._selected_initial_kernel = next(
            (entry.source.read_bytes() if entry.source is not None else entry.text.encode()
             for entry in content.files if entry.path == "kernel.py"), None)
        if self._selected_initial_kernel is None:
            raise ValueError("Selected SourceProvider must supply kernel.py")
        if initial.kind == "sol":
            from aka.contracts.content import WorkspaceFile
            from aka.evaluation.resources import profile_driver_script
            # Task owns source and harness content; this Workflow supplies its
            # selected tool and Git instructions through the public preparation.
            content = replace(content, readme=(
                "# Managed SOL optimization workspace\n\n"
                "Optimize `kernel.py` while preserving the definition's destination-passing "
                "interface. Maximize the arithmetic mean of workload speedups against the "
                "reference while every workload passes its correctness checks.\n\n"
                "Use `python3 tools/sandbox.py --kind run --no-sync`. The Supervisor owns "
                "the evaluator, its configuration, solution metadata and Git. Do not run "
                "the evaluator locally or edit those files. Submit `episode-report` through "
                "the Runtime; read the installed gpu-measurement and runtime-records Skills "
                "for request examples. Put temporary probes and requests under `scratch/`.\n"
            ))
            source_revision = self.candidate_workspace.prepare_sol(
                self.workspace, content, problem_files=self.problem.agent.files,
                harness=self.problem.evaluation.harness,
                runtime_files=(
                    WorkspaceFile("CLAUDE.md", source=REPO_ROOT / "reference" / "CLAUDE.md"),
                    WorkspaceFile("profile_driver.py", source=profile_driver_script()),
                ),
            )
            self._initial_source_paths = tuple(item.path for item in content.files)
            self._link_runtime()
            result = self._run_v0_evaluator_with_correctness_prefetch()
            self._finalize_v0_measurement(result, source_revision,
                                         extra_paths=("CLAUDE.md", ".gitignore"))
            return
        from aka.contracts.content import WorkspaceFile
        from aka.evaluation.resources import profile_driver_script
        inputs = self.problem_provider.content(self.problem, self.workspace,
            generated_digest=self._generated_agent_problem_digest)
        public = inputs.prepare_public()
        public_files = list((*inputs.operator_files(), *public.files))
        if public.ready and inputs.generated_digest and not any(
                item.path == "agent_problem.json" for item in public_files):
            saved = self.candidate_workspace.read_working(self.workspace, ("agent_problem.json",))
            public_files.append(WorkspaceFile("agent_problem.json",
                text=saved.files["agent_problem.json"].decode("utf-8")))
        harness = self.problem.evaluation.harness
        if harness is None:
            raise ValueError("Native V0 requires the frozen Task harness")
        runtime = self.problem_provider.runtime_files(self.problem)
        history = self.candidate_workspace.revision_history(self.workspace, changed_path="kernel.py")
        runtime_files = self._unique_contributions((
            *public_files, harness, *(runtime if not history else ()),
            WorkspaceFile("CLAUDE.md", source=REPO_ROOT / "reference" / "CLAUDE.md"),
            WorkspaceFile("profile_driver.py", source=profile_driver_script()),
        ))
        content = replace(content, readme=self._native_v0_readme(
            generalized=self.private_reference_dir is not None))
        self._initial_source_paths = tuple(item.path for item in content.files)
        selected_source = {item.path: (item.source.read_bytes() if item.source is not None
                                      else item.text.encode("utf-8")) for item in content.files}
        self.candidate_workspace.initialize_kernel(self.workspace, content,
            runtime_files=runtime_files,
            context_directories=("atrex-bench",) if runtime and not history else ())
        if not public.ready:
            self._author_public_problem(inputs)
        self._generated_agent_problem_digest = inputs.generated_digest
        inputs.assert_private()
        self._link_runtime()
        if self.candidate_workspace.read_working(self.workspace, self._initial_source_paths).files != selected_source:
            raise RuntimeError("native V0 source no longer matches the selected initial implementation")
        if history:
            source_revision = history[-1].revision
        else:
            source_revision = self.candidate_workspace.record(self.workspace,
                expected_revision=self.candidate_workspace.current_revision(self.workspace),
                files=tuple(WorkspaceFile(name, source=self.workspace / name)
                    for name in (*self._initial_source_paths, ".gitignore")),
                message="V0: baseline kernel")
        result = self._run_v0_evaluator_with_correctness_prefetch()
        self._finalize_v0_measurement(result, source_revision)
        inputs.assert_private()
        required = ("agent_problem.json",) if self.private_reference_dir is not None else ()
        changed = self.candidate_workspace.baseline_changes(self.workspace, source_revision,
                                                           required_paths=required)
        if changed:
            raise RuntimeError("invalid native V0 frozen input: " + ", ".join(changed))
        status = self.candidate_workspace.working_facts(self.workspace).status_text
        if status:
            import sys
            print("[orchestrator] WARNING: native V0 is continuing with a dirty "
                  f"workspace:\n{status}", file=sys.stderr, flush=True)

    @staticmethod
    def _unique_contributions(files):
        """Merge repeated public roles only when they contribute the same bytes."""
        selected = {}
        for item in files:
            previous = selected.get(item.path)
            if previous is not None:
                def payload(value):
                    return value.source.read_bytes() if value.source is not None else value.text.encode("utf-8")
                if (payload(previous) != payload(item)
                        or (previous.source is not None and item.source is not None
                            and previous.source.stat().st_mode & 0o777 != item.source.stat().st_mode & 0o777)):
                    raise ValueError(f"conflicting public input roles: {item.path}")
                continue
            selected[item.path] = item
        return tuple(selected.values())

    def _author_public_problem(self, inputs):
        """Schedule authoring; Task validates and retains its public contribution."""
        import tempfile
        from pathlib import Path
        from aka.resources.prompts import render_prompt
        prompt_source = self._module_resources.resource_result().resource(
            "task-problem-generation-prompt").source
        validation_error = ""
        with tempfile.TemporaryDirectory(prefix="aka-generalize-problem-") as directory:
            staging = Path(directory)
            self.candidate_workspace.install_files(staging, inputs.authoring_assets())
            for attempt in range(2):
                repair = ("The current agent_problem.json failed orchestrator validation. Replace it "
                          f"with a corrected file. Validation error: {validation_error}"
                          if validation_error else "Create agent_problem.json now.")
                result = self._run_session(staging,
                    render_prompt(prompt_source, REPAIR_CONTEXT=repair),
                    timeout=self.problem_generation_timeout,
                    agent_cli=self.agent_cli, reasoning_effort="max",
                    extra_environment=self.agent_boundary_environment("problem-generation"))
                self._account(result, f"agent problem generation attempt {attempt + 1}")
                generated = staging / "agent_problem.json"
                try:
                    inputs.validate_generated(generated)
                except ValueError as error:
                    validation_error = str(error)
                    if attempt == 0:
                        continue
                    detail = result.stderr_tail or result.stdout_tail
                    raise RuntimeError("AKA could not generate a valid generalized production problem "
                        f"after two attempts: {validation_error}"
                        + (f"; agent output: {detail}" if detail else "")) from error
                self.candidate_workspace.install_files(self.workspace,
                    (inputs.generated_file(generated),), immutable=True)
                inputs.record_generated()
                return

    def _finalize_v0_measurement(self, result, source_commit, *, extra_paths=()):
        """Record Workflow-owned V0 evidence through the selected Workspace transaction."""
        import json
        from aka.contracts.content import WorkspaceFile
        from aka.workflow.light.orchestration.version_records import (
            save_version_record, version_commit_paths, version_record_path,
        )
        path = version_record_path(self.workspace, 0, prefer_summary=self.generate_memory)
        memory = json.loads(path.read_text(encoding="utf-8"))
        memory["git_commit_hash"] = source_commit
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = "baseline"
        optimization["action_description"] = (
            "selected initial implementation measured by the official full-workload evaluator")
        optimization["expected_impact"] = "correctness and latency reference for later versions"
        optimization["risks_and_rollback"] = "none; immutable V0 reference baseline"
        save_version_record(self.workspace, 0, memory, generate_memory=self.generate_memory)
        problem = self._generalized_memory_coverage_problem(memory)
        if problem:
            raise RuntimeError(f"invalid native V0 measurement: {problem}")
        source_paths = getattr(self, "_initial_source_paths", ("kernel.py", "solution.json")
                               if self.problem.initial_source.kind == "sol" else ("kernel.py",))
        original = self.candidate_workspace.read_revision(self.workspace, source_commit, list(source_paths))
        working = self.candidate_workspace.read_working(self.workspace, source_paths)
        if original.files != working.files:
            raise RuntimeError("V0 candidate source differs from its recorded source commit")
        paths = version_commit_paths(self.workspace, 0, generate_memory=self.generate_memory)
        # Existing SOL ignores are updated by local runtime installation. Frozen
        # constraints already have a context identity and must not be staged again.
        if extra_paths:
            recorded = self.candidate_workspace.read_revision(self.workspace, source_commit, list(extra_paths))
            paths.extend(name for name in extra_paths
                         if recorded.files.get(name) is not None and (self.workspace / name).is_file())
        expected = self.candidate_workspace.current_revision(self.workspace)
        revision = self.candidate_workspace.record(self.workspace, expected_revision=expected,
            files=tuple(WorkspaceFile(name, source=self.workspace / name) for name in paths),
            message="V0: record baseline measurement")
        if revision == expected:
            raise RuntimeError("V0 evaluator produced no measurement metadata to commit")
