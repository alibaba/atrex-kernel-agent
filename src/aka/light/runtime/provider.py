"""Campaign-scoped factory; business state never lives in the Core container."""
from dataclasses import dataclass, replace
from typing import Protocol
from aka.light.wiki.config import WikiConfig


class ToolRuntimeFactory(Protocol):
    def open(self, configuration, *, workspace_provider): ...


@dataclass(frozen=True)
class ManagedRuntimeFactory:
    measurement_repetitions: int = 1
    wiki: bool = False
    request_timeout: float | None = None
    queue_timeout: float = 60
    wiki_options: dict | None = None
    services: object | None = None

    @property
    def research_records(self) -> bool:
        return self.services.journal.research_records if self.services is not None else False

    def open(self, configuration, *, workspace_provider):
        from aka.light.runtime.server import SupervisorRuntime
        return SupervisorRuntime(replace(configuration,
            measurement_repetitions=self.measurement_repetitions,
            wiki_enabled=self.wiki,
            wiki_options=WikiConfig(**(self.wiki_options or {})),
            request_timeout=self.request_timeout if self.request_timeout is not None else configuration.request_timeout,
            queue_timeout=self.queue_timeout), workspace_provider=workspace_provider,
            services=self.services)
