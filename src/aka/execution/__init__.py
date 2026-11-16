"""Execution mechanisms and providers, independent of application policy."""


def compose_execution(**options):
    """Own an Executor and its resources selected by the Execution profile."""
    from .assembly import compose_execution as compose
    return compose(**options)


def snapshot_command_inputs(workspace, selections, *, filesystem_access, max_file_bytes):
    """Read explicit inputs under the caller's existing filesystem authorization."""
    from .input_snapshot import snapshot_command_inputs as snapshot
    return snapshot(workspace, selections, filesystem_access=filesystem_access,
                    max_file_bytes=max_file_bytes)
