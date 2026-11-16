from pathlib import Path
from .base import AgentBackendAdapter, BackendFiles
from ..model import AgentRuntimeCapabilities


class DshAdapter(AgentBackendAdapter):
    id = "dsh"
    settings_variable = "ATREX_DSH_SESSION_SETTINGS"
    supports_resume = True
    environment_variables = ("DSH_HOME", "ATREX_DSH_SESSION_SETTINGS", "DEEPSEEK_API_KEY",
                             "AKA_DSH_PROVIDER", "AKA_DSH_MODEL", "AKA_DSH_REASONING_EFFORT")
    files = BackendFiles((".dsh/.env", ".dsh/.credentials.yaml", ".dsh/cordis.patch.yml"),
                         (".dsh",), (), (("DSH_HOME", ".dsh"),))

    def configuration_files(self, environment):
        from ..dsh import settings_values
        profile = settings_values(environment.get(self.settings_variable)
                                  or environment.get("ATREX_SESSION_SETTINGS", "")).get("profile", "headless")
        return (*self.files.config_files, f".dsh/profiles/{profile}/package.json",
                f".dsh/profiles/{profile}/cordis.patch.yml")

    def configure_environment(self, environment):
        values = dict(environment)
        values["DSH_HOME"] = str(Path(values.get("DSH_HOME")
                               or Path(values.get("HOME") or Path.home()) / ".dsh").expanduser().resolve())
        return values

    def resume_command(self, command, session_id):
        command = list(command)
        index = command.index("--")
        command[index:index] = ["--session-id", session_id]
        return command
    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True, usage_delta=True, phase_marker_receipt=True,
    )

    def __init__(self, runtime_assets_dir: Path | None = None):
        del runtime_assets_dir

    def build_command(self, prompt, session_id, reasoning_effort, settings):
        from ..dsh import settings_values
        values = settings_values(settings)
        return ["dsh", "--profile", values.get("profile", "headless"), "--patch",
                str(Path(__file__).parent.parent / "dsh_runtime.yml"), "--json",
                "--permission-mode", "danger-full-access", "--approval-policy", "never",
                "--", prompt]

    def prepare_environment(self, environment, *, reasoning_effort):
        from ..dsh import settings_values
        settings = settings_values(environment.get(self.settings_variable)
                                   or environment.get("ATREX_SESSION_SETTINGS", ""))
        provider = settings.get("provider", "deepseek-official")
        effort = ({"medium": "high", "xhigh": "max"}.get(reasoning_effort, reasoning_effort)
                  if provider == "deepseek-official" else reasoning_effort)
        effort = settings.get("reasoning_effort", effort)
        if effort == "default":
            effort = ""
        return {
            **environment, "AKA_DSH_PROVIDER": provider,
            "AKA_DSH_MODEL": settings.get("model", "deepseek-flash"),
            "AKA_DSH_REASONING_EFFORT": effort,
        }

    def tool_environment(self, environment):
        from ..dsh import runtime_connection_environment
        return runtime_connection_environment(environment)

    def release_environment(self, environment):
        from ..dsh import release_runtime_connection
        release_runtime_connection(environment)

    def normalize_stream(self, stdout):
        from ..dsh import normalize_stream
        return normalize_stream(stdout)

    def classify_terminal_failure(self, stdout, session_id):
        from ..dsh import terminal_failure
        return terminal_failure(stdout, session_id)

    def validated_exit_status(self, stdout, exit_status):
        return exit_status or (1 if self.classify_terminal_failure(stdout, "")[0] else 0)

    def auth_hint(self):
        return "configure provider authentication in a DSH headless profile and run `dsh --profile headless --help`"
