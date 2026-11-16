"""Native execution preserves the selected command and environment."""

from .launch import SandboxLaunch


class NativeExecution:
    def launch(self, command, workspace, environment, backend, host_home, access):
        values = {
            k: v for k, v in environment.items() if k != "ATREX_AGENT_PREPARED_HOME"
        }
        return SandboxLaunch(list(command), values)
