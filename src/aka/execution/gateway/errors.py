"""Shared exception identity for imported and module-entrypoint transports."""


class GatewayHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"gateway HTTP {status}: {detail}")
