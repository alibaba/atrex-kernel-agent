"""Selected backend recovery over confirmed job identities; no scheduler decisions."""
from aka.contracts.execution import UnsupportedExecution
from aka.contracts.execution_recovery import InfrastructureRetryPolicy, JobRecoveryCapabilities
from .gateway.jobs import GatewayJobs


class GatewayRecoveryMixin:
    def _job_transport(self):
        configuration = getattr(self, "configuration", {})
        request = getattr(self, "request", None)
        if request is not None and request.ssh:
            raise UnsupportedExecution("SSH executor does not provide durable job queries")
        if request is not None:
            endpoint, profile = request.url, request.profile
            environment = request.environment
        else:
            from .configuration import legacy_settings
            settings = legacy_settings()
            endpoint = configuration.get("url", settings.url)
            profile = configuration.get("profile", settings.profile)
            environment = {}
        if not endpoint and not profile:
            # Keep CLI's selected default profile; never invent an alternate endpoint.
            profile = "prod"
        from .gateway.sandbox import _find_agate
        executable = _find_agate()
        supported = configuration.get("idempotent_replay", False) or environment.get("AKA_GATEWAY_IDEMPOTENT_REPLAY") == "true"
        if supported:
            # Explicitly configured stable-key replay uses only the unified HTTP API.
            if not endpoint:
                raise UnsupportedExecution("idempotent replay needs a verified explicit gateway URL")
            executable = None
        return GatewayJobs(endpoint=endpoint, profile=profile, executable=executable,
                           idempotent_replay=bool(supported))

    @property
    def recovery_capabilities(self):
        try:
            transport = self._job_transport()
        except UnsupportedExecution:
            return JobRecoveryCapabilities()
        return JobRecoveryCapabilities(bool(transport.endpoint or transport.executable), True,
                                       transport.idempotent_replay)

    def query_job(self, reference, *, timeout, retry=InfrastructureRetryPolicy()):
        return self._job_transport().query(reference, timeout=timeout, retry=retry)

    def cancel_job(self, reference, *, timeout):
        return self._job_transport().cancel(reference, timeout=timeout)

    def replay_job(self, command, *, idempotency_key):
        configuration = getattr(self, "configuration", {})
        policy = InfrastructureRetryPolicy(
            tuple(configuration.get("recovery_retry_delays", ())),
            configuration.get("recovery_retry_budget_s", 0),
        )
        return self._job_transport().replay(command, idempotency_key=idempotency_key, retry=policy)
