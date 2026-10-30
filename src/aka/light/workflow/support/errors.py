"""Transport lifecycle errors shared without importing the HTTP host."""

class RequestValidationError(ValueError):
    """Repairable input rejected before execution."""

class RequestDispatchTimeout(RuntimeError):
    """Queue/deadline expired before creating an executor."""

class GatewayExecutionUnavailable(RuntimeError):
    """A dispatched request ended without a trustworthy outcome."""

class SessionRevokedError(RuntimeError):
    """The Session no longer authorizes execution."""
