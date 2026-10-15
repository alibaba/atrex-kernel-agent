"""Compatibility entry for the legacy workspace initializer."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
_source_packages = root / "src"
if _source_packages.is_dir():
    sys.path.insert(0, str(_source_packages))

from aka.legacy.application.source import source_provider
from aka.legacy.application.workspace import candidate_workspace


def _usage_error(message, usage):
    print(f"Error: {message}")
    print(f"Usage: {usage}")
    return 1


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "--initialize":
        if len(arguments) < 3 or not arguments[1] or not arguments[2]:
            return _usage_error("reference directory and initializer entry are required",
                                f"{sys.argv[0]} --initialize <reference_dir> <entry> <name> <kernel_demo_path>")
        _, reference_dir, entry, *inputs = arguments
        usage = f"{entry} <name> <kernel_demo_path>"
        if not inputs or not inputs[0]:
            return _usage_error("workspace name is required", usage)
        if len(inputs) < 2 or not inputs[1]:
            return _usage_error("kernel_demo path is required", usage)
        with source_provider(kernel_demo=inputs[1]) as source:
            content = source.prepare()
        with candidate_workspace() as workspace:
            return workspace.initialize_kernel(Path(reference_dir), entry, inputs, content=content)
    usage = f"{sys.argv[0]} <kernel_demo_path> <workspace_path>"
    if not arguments or not arguments[0]:
        return _usage_error("kernel_demo path is required", usage)
    if len(arguments) < 2 or not arguments[1]:
        return _usage_error("workspace path is required", usage)
    with source_provider(kernel_demo=arguments[0]) as source:
        content = source.prepare()
    with candidate_workspace() as workspace:
        try:
            workspace.install_files(Path(arguments[1]), content.files)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
