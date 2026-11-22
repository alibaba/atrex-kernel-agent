"""Shared exception identity for imported and module-entrypoint transports."""


class GatewayHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str, *, retry_after_s=None) -> None:
        self.status = status
        self.detail = detail
        self.retry_after_s = retry_after_s
        super().__init__(f"gateway HTTP {status}: {detail}")
