"""Transport lifecycle errors shared without importing the HTTP host."""

class RequestValidationError(ValueError):
    """Repairable input rejected before execution."""

class RequestDispatchTimeout(RuntimeError):
    """Queue/deadline expired before creating an executor."""

class GatewayExecutionUnavailable(RuntimeError):
    """A dispatched request ended without a trustworthy outcome."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "gateway_outcome_unknown",
        reason: str = "outcome_unknown",
        cancellation_requested: bool = False,
        **details,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.cancellation_requested = cancellation_requested
        self.details = details


class ResultPublicationError(RuntimeError):
    """Execution completed, but a bounded Agent-facing result could not be published."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "result_publication_failed",
        phase: str = "result_publication",
        gateway_record_id: str | None = None,
        **details,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.gateway_record_id = gateway_record_id
        self.details = details

    def bind_record(self, gateway_record_id: str) -> None:
        """Attach the durable result written before publication was attempted."""
        if self.gateway_record_id is None:
            self.gateway_record_id = gateway_record_id

class SessionRevokedError(RuntimeError):
    """The Session no longer authorizes execution."""
