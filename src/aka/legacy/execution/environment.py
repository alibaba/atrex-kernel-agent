"""Bind selected Execution settings to a coding-agent process environment."""


def session_environment(*, hardware="", profile="", url="", timeout=600,
                        ssh="", ssh_init="", health_command="", state_file="",
                        extra=None):
    environment = {str(k): None if v is None else str(v) for k, v in (extra or {}).items()}
    if hardware:
        environment["ATREX_SANDBOX_GPU"] = hardware
    for value, key, others in (
        (ssh, "ATREX_SANDBOX_SSH", ("ATREX_SANDBOX_URL", "ATREX_SANDBOX_PROFILE")),
        (url, "ATREX_SANDBOX_URL", ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_PROFILE")),
        (profile, "ATREX_SANDBOX_PROFILE", ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_URL")),
    ):
        if value:
            environment[key] = value
            environment.update((other, None) for other in others)
            break
    for value, key in ((ssh_init, "ATREX_SANDBOX_SSH_INIT"),
                       (health_command, "ATREX_SANDBOX_HEALTH_COMMAND"),
                       (state_file, "ATREX_ENVIRONMENT_STATE_FILE")):
        if value:
            environment[key] = value
    environment["ATREX_SANDBOX_TIMEOUT"] = str(timeout)
    return environment
