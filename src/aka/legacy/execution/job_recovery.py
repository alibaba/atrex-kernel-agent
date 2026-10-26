"""Selected backend recovery over confirmed job identities; no scheduler decisions."""
from aka.contracts.execution import UnsupportedExecution
from aka.contracts.execution_recovery import InfrastructureRetryPolicy, JobRecoveryCapabilities
from .gateway.jobs import GatewayJobs, selected_route


def recovery_policy(configuration):
    if type(configuration.get("idempotent_replay", False)) is not bool:
        raise ValueError("idempotent replay support must be an explicit boolean")
    return InfrastructureRetryPolicy(tuple(configuration.get("recovery_retry_delays", ())),
                                     configuration.get("recovery_retry_budget_s", 0))


class GatewayRecoveryMixin:
    def _job_transport(self):
        cached = getattr(self, "_selected_job_transport", None)
        if cached is not None:
            return cached
        configuration = getattr(self, "configuration", {})
        request = getattr(self, "request", None)
        environment = request.environment if request is not None else getattr(self, "environment", None)
        from .configuration import legacy_settings
        settings = request if request is not None else legacy_settings(environment)
        if settings.ssh:
            raise UnsupportedExecution("SSH executor does not provide durable job queries")
        endpoint = configuration.get("url") or settings.url
        profile = configuration.get("profile") or settings.profile
        if configuration.get("url"):
            profile = ""
        elif configuration.get("profile"):
            endpoint = ""
        from .gateway.sandbox import _find_agate
        executable = _find_agate()
        endpoint, profile, environment = selected_route(endpoint, profile, environment=environment,
                                                        executable=executable)
        supported = configuration.get("idempotent_replay", False)
        if supported:
            if not configuration.get("url"):
                raise UnsupportedExecution("stable-key replay needs a verified explicit gateway URL")
            executable = None
        elif endpoint:
            # Query the confirmed URL directly with frozen authentication. A later
            # CLI config change cannot redirect the job or introduce new credentials.
            executable = None
        transport = GatewayJobs(endpoint=endpoint, profile=profile, executable=executable,
                                environment=environment, idempotent_replay=bool(supported))
        self._selected_job_transport = transport
        return transport

    @property
    def recovery_capabilities(self):
        try:
            transport = self._job_transport()
        except UnsupportedExecution:
            return JobRecoveryCapabilities()
        query = bool(transport.endpoint or transport.executable)
        return JobRecoveryCapabilities(query, query, transport.idempotent_replay)

    def query_job(self, reference, *, timeout, retry=None):
        policy = recovery_policy(getattr(self, "configuration", {})) if retry is None else retry
        return self._job_transport().query(reference, timeout=timeout, retry=policy)

    def cancel_job(self, reference, *, timeout):
        return self._job_transport().cancel(reference, timeout=timeout)

    def replay_job(self, command, *, idempotency_key):
        self.capabilities.require(command)
        from dataclasses import replace
        from .environment_query import command_query, require_supported
        from .gateway.requests import bound_limits
        backend = self._environment_backend() if hasattr(self, "request") else self._environment_backend(command.environment)
        support = backend.check(command_query(command), discover=False)
        require_supported(support)
        def decorate(result):
            return replace(result, environment_snapshot=support.environment,
                           environment_identity=support.environment.identity, environment_support=support)
        def observed(result):
            command.on_observation(decorate(result))
        bound = replace(command, environment=bound_limits(command.environment, support),
                        on_observation=observed if command.on_observation is not None else None)
        return decorate(self._job_transport().replay(bound, idempotency_key=idempotency_key,
            retry=recovery_policy(getattr(self, "configuration", {}))))
