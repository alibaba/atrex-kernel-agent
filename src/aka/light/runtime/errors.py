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

class SessionRevokedError(RuntimeError):
    """The Session no longer authorizes execution."""
