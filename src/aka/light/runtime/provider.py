"""Campaign-scoped factory; business state never lives in the Core container."""
from dataclasses import dataclass, replace
from typing import Protocol


class ToolRuntimeFactory(Protocol):
    def open(self, configuration, *, workspace_provider, agent_resources=None): ...


@dataclass(frozen=True)
class ManagedRuntimeFactory:
    measurement_repetitions: int = 1
    wiki: bool = False
    request_timeout: float | None = None
    queue_timeout: float = 60
    wiki_options: dict | None = None
    services: object | None = None

    def __post_init__(self):
        if self.wiki or self.wiki_options:
            raise ValueError(
                "Light's legacy Wiki runtime has been retired; select public/internal Wiki resources in a Core profile"
            )

    @property
    def research_records(self) -> bool:
        return self.services.journal.research_records if self.services is not None else False

    def open(self, configuration, *, workspace_provider, agent_resources=None):
        from aka.light.orchestrator.supervisor_runtime import SupervisorRuntime
        return SupervisorRuntime(replace(configuration,
            measurement_repetitions=self.measurement_repetitions,
            request_timeout=self.request_timeout if self.request_timeout is not None else configuration.request_timeout,
            queue_timeout=self.queue_timeout), workspace_provider=workspace_provider,
            services=self.services, agent_resources=agent_resources)
