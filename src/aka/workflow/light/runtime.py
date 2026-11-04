"""Workflow-owned operation assembly borrowing the selected common modules."""
from dataclasses import dataclass, replace

from aka.contracts.evaluation import EvaluationConfiguration
from aka.contracts.execution import ExecutionContext
from aka.execution.binding import SandboxExecutor, task_environment
from aka.resources.wiki.config import WikiConfig


@dataclass(frozen=True)
class RuntimeServices:
    agent: object
    executor: object
    experiments: object
    evaluator: object
    command_evaluator: object
    evaluation_input: object
    configuration: EvaluationConfiguration
    resources: object
    workspace_provider: object
    journal: object
    evaluation_identity: str


class ManagedRuntimeFactory:
    """Private Workflow assembly; no version-specific module protocol."""
    def __init__(self, workflow, campaign):
        self.workflow, self.campaign = workflow, campaign
        self.wiki = workflow.options.wiki
        self.research_records = workflow.options.research_records
        self.measurement_repetitions = workflow.evaluation_defaults.measurement_repetitions

    def open(self, configuration, *, workspace_provider):
        from .journal import JournalFactory
        from aka.workflow.light.orchestration.supervisor_runtime import SupervisorRuntime
        c, w = self.campaign, self.workflow
        evaluation_input = w.problem.content(c.problem, c.workspace,
            generated_digest=getattr(c, "_generated_agent_problem_digest", None)).evaluation
        environment = dict(w.execution_context.environment)
        environment.update(task_environment(evaluation_input.worker_requirements))
        if evaluation_input.private_reference_dir is not None:
            environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(evaluation_input.private_reference_dir)
        executor = SandboxExecutor(w.executor, ExecutionContext(environment, configuration.hardware,
            configuration.timeout))
        selected = replace(w.evaluation_defaults, timeout=configuration.timeout,
            repeats=c.verify_repeats, per_run_timeout=c.verify_run_timeout,
            hardware=configuration.hardware, environment=environment,
            native=bool(evaluation_input.runtime_root),
            private_reference_dir=evaluation_input.private_reference_dir,
            max_parallelism=w.measurement_parallelism, execution_capabilities=executor.capabilities)
        evaluator = w.evaluator_factory.create(selected, candidate_workspace=workspace_provider,
            evaluation_input=evaluation_input)
        command = w.evaluator_factory.create_command(selected, evaluation_input=evaluation_input)
        services = RuntimeServices(w.agent, executor, w.experiment_factory, evaluator, command,
            evaluation_input, selected, c._module_resources, workspace_provider,
            JournalFactory(research_records=self.research_records), w.problem.snapshot(c.problem).identity)
        configured = replace(configuration, measurement_repetitions=self.measurement_repetitions,
            wiki_enabled=self.wiki, wiki_options=WikiConfig(**(w.options.wiki_options or {})),
            request_timeout=w.options.request_timeout if w.options.request_timeout is not None
                else configuration.request_timeout, queue_timeout=w.options.queue_timeout)
        runtime = SupervisorRuntime(configured, workspace_provider=workspace_provider, services=services)
        runtime.inherit_state = w.options.inherit_state
        return runtime
