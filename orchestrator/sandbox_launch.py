"""Own the anonymous argument file used to configure a Bubblewrap process."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import BinaryIO


@dataclass(slots=True)
class SandboxLaunch:
    command: list[str]
    environment: dict[str, str] = field(repr=False)
    _arguments: BinaryIO | None = field(default=None, repr=False)

    @classmethod
    def with_bwrap_environment(
        cls, prefix: list[str], command: list[str], environment: dict[str, str],
    ) -> SandboxLaunch:
        # Never execute `env KEY=secret ...`: even behind --args it exposes the
        # values in env's own argv. Let bwrap set them before directly execing CLI.
        options = ["--clearenv"]
        for key, value in sorted(environment.items()):
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError("Sandbox environment contains an invalid name or NUL byte")
            options.extend(("--setenv", key, value))
        payload = b"\0".join(os.fsencode(option) for option in options) + b"\0"
        # Seekable rather than a pre-filled pipe: large environments must not
        # deadlock before the reader is spawned. TemporaryFile is unlinked/anonymous.
        arguments = tempfile.TemporaryFile(mode="w+b")
        try:
            arguments.write(payload)
            arguments.flush()
            arguments.seek(0)
            return cls(
                [*prefix, "--args", str(arguments.fileno()), "--", *command],
                environment, arguments,
            )
        except BaseException:
            arguments.close()
            raise

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self._arguments.fileno(),) if self._arguments is not None else ()

    def close(self) -> None:
        arguments = self._arguments
        self._arguments = None
        if arguments is not None:
            arguments.close()
