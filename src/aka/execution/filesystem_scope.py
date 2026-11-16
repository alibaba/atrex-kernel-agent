"""Keep a borrowed execution filesystem view within its host authorization."""
from dataclasses import replace

from aka.contracts.execution import UnsupportedExecution
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount


def require_scope_backend(context, capabilities):
    if context.filesystem_access is not None and not capabilities.filesystem_isolation:
        raise UnsupportedExecution("executor cannot enforce the bound file and environment scope")


def restricted_scope(bound, requested):
    """An omitted command scope inherits; an explicit scope can only narrow it."""
    if bound is None:
        return requested
    if requested is None:
        return bound
    grants = {(mount.source, mount.target): mount for mount in bound.mounts}
    mounts = []
    for mount in requested.mounts:
        allowed = grants.get((mount.source, mount.target))
        if allowed is None or mount.writable and not allowed.writable:
            raise UnsupportedExecution("command filesystem grants exceed the bound scope")
        mounts.append(FilesystemMount(mount.source, mount.target, mount.writable))
    return FilesystemAccess(tuple(mounts),
        tuple(dict.fromkeys((*bound.hidden_paths, *requested.hidden_paths))),
        tuple(dict.fromkeys((*bound.read_only_paths, *requested.read_only_paths))))


def bound_command(context, command):
    scope = restricted_scope(context.filesystem_access, command.filesystem_access)
    if scope is None:
        environment = {**context.environment, **command.environment}
        for key in ("ATREX_SANDBOX_REQUIREMENTS_FILE", "ATREX_PRIVATE_REFERENCE_DIR"):
            if key in context.environment:
                environment[key] = context.environment[key]
    else:
        # Explicit workload values (including CLI -e and PROFILE_* values) remain
        # legal. A restricted call never borrows the host's ambient credentials,
        # dependency/private-reference paths or selected runtime environment.
        environment = dict(command.environment)
    return replace(command, environment=environment,
        hardware=command.hardware or context.hardware, filesystem_access=scope)
