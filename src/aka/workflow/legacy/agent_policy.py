"""Workflow host dependency and shared gateway protection policy."""

import ast
import os
import re
import shlex
from pathlib import Path

DEFAULT_PROTECTED_GATEWAY_SCREEN = "atrex-local-gateway"
DEFAULT_PROTECTED_GATEWAY_STATE_NAME = "atrex-local-gateway"


def _python_entrypoint(tokens: list[str]) -> tuple[str, str]:
    """Parse interpreter options without treating script arguments as Python code."""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return "script", tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "--check-hash-based-pycs":
            index += 2
        elif token.startswith("--"):
            index += 1
        elif token.startswith("-") and token != "-":
            # Python accepts combined flags such as -uc and attached -cCODE.
            for offset, flag in enumerate(token[1:], start=1):
                if flag in {"c", "m"}:
                    value = token[offset + 1 :] or (
                        tokens[index + 1] if index + 1 < len(tokens) else ""
                    )
                    return ("code" if flag == "c" else "module"), value
                if flag in {"W", "X"}:
                    if offset == len(token) - 1:
                        index += 1
                    break
            index += 1
        else:
            return "script", token
    return "", ""


def protected_gateway_identity(
    environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve shared gateway protection targets without embedding host paths."""
    values = os.environ if environment is None else environment
    screen = values.get(
        "ATREX_PROTECTED_GATEWAY_SCREEN", DEFAULT_PROTECTED_GATEWAY_SCREEN
    )
    state_dir = values.get("ATREX_PROTECTED_GATEWAY_STATE_DIR")
    if not state_dir:
        cache_home = values.get("XDG_CACHE_HOME")
        cache_root = (
            Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        )
        state_dir = str(cache_root / DEFAULT_PROTECTED_GATEWAY_STATE_NAME)
    return screen, state_dir


def python_import_roots(code: str, *, _depth: int = 0) -> set[str]:
    """Return real imported top-level modules without matching strings/comments."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and node.args:
            target: str | None = None
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                target = "import"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                target = "import"
            if target and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                if isinstance(module, str) and module:
                    roots.add(module.split(".", 1)[0])
            if (
                _depth < 2
                and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval"}
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                roots.update(python_import_roots(node.args[0].value, _depth=_depth + 1))
    return roots


def dependency_process_violation(
    argv: list[str], environment=None, *, cwd: Path | None = None,
    trusted_sandbox_entrypoints: frozenset[Path] | None = None,
) -> str | None:
    """Report confirmed policy violations for the shared dependency guard.

    ``cwd`` must be the inspected process's directory, not the Supervisor's.
    Missing process/file evidence is inconclusive (None), not authorization or
    evidence of a violation. In particular, procfs can disappear during exit.
    """
    if not argv:
        return None
    from aka.resources.bundles import resource_path
    canonical = resource_path('tools/sandbox.py')
    trusted = frozenset({canonical}) if trusted_sandbox_entrypoints is None else trusted_sandbox_entrypoints
    transport_names = {"sandbox.py", *(path.name for path in trusted)}

    def unwrap(segment: list[str]) -> list[str]:
        result = list(segment)
        while result and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", result[0]):
            result.pop(0)
        if result and Path(result[0]).name.lower() in {"env", "command"}:
            result.pop(0)
            while result and (result[0].startswith("-") or "=" in result[0]):
                result.pop(0)
        if result and Path(result[0]).name.lower() == "timeout":
            result.pop(0)
            while result and result[0].startswith("-"):
                result.pop(0)
            if result:
                result.pop(0)
        return result

    def command_segments(process_argv: list[str]) -> list[list[str]]:
        executable = Path(process_argv[0]).name.lower()
        # A real argv already preserves argument boundaries. A quoted ';' in
        # Dev's remote command is data, not a new local shell command.
        if executable not in {"bash", "sh", "dash", "zsh", "ksh"}:
            return [process_argv]
        command_index = next(
            (index + 1 for index, value in enumerate(process_argv[:-1])
             if value.startswith("-") and "c" in value[1:]),
            -1,
        )
        if command_index < 0:
            return [process_argv]

        # Split before unquoting: shlex's punctuation tokens otherwise lose the
        # distinction between a shell operator and a literal quoted ';'/'|'.
        command = process_argv[command_index]
        boundaries = [-1]
        quote = ""
        escaped = False
        for index, character in enumerate(command):
            if escaped:
                escaped = False
            elif character == "\\" and quote != "'":
                escaped = True
            elif quote:
                if character == quote:
                    quote = ""
            elif character in "'\"`":
                quote = character
            elif character in ";&|":
                boundaries.append(index)
        boundaries.append(len(command))
        segments: list[list[str]] = []
        try:
            for left, right in zip(boundaries, boundaries[1:]):
                segment = shlex.split(command[left + 1:right], comments=True)
                if segment:
                    segments.append(segment)
        except ValueError:
            return [process_argv]
        expanded = list(segments)
        for segment in segments:
            unwrapped = unwrap(segment)
            if (
                unwrapped
                and Path(unwrapped[0]).name.lower() == "eval"
                and len(unwrapped) > 1
            ):
                expanded.extend(
                    command_segments(["sh", "-c", " ".join(unwrapped[1:])])
                )
        return expanded

    def is_installer(segment: list[str]) -> bool:
        tokens = unwrap(segment)
        if not tokens:
            return False
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        if re.fullmatch(r"pip[0-9.]*", executable):
            return len(lowered) > 1 and lowered[1] in {"install", "wheel"}
        if executable == "uv":
            return lowered[1:3] in (
                ["pip", "install"],
                ["pip", "sync"],
                ["pip", "compile"],
            )
        if executable in {"conda", "mamba", "micromamba"}:
            return len(lowered) > 1 and lowered[1] in {"install", "create"}
        if re.fullmatch(r"python[0-9.]*", executable):
            if len(lowered) > 3 and lowered[1:3] == ["-m", "pip"]:
                return lowered[3] in {"install", "wheel"}
            if len(lowered) > 2 and lowered[1:3] == ["-m", "build"]:
                return True
            for index, token in enumerate(lowered[:-1]):
                if Path(token).name == "setup.py" and lowered[index + 1] in {
                    "install",
                    "build",
                    "build_ext",
                    "bdist_wheel",
                }:
                    return True
            if "--" in lowered:
                boundary = lowered.index("--")
                return is_installer(tokens[boundary + 1 :])
        if Path(executable).name == "setup.py":
            return len(lowered) > 1 and lowered[1] in {
                "install",
                "build",
                "build_ext",
                "bdist_wheel",
            }
        return False

    def transport_status(segment: list[str]) -> str | None:
        tokens = unwrap(segment)
        if not tokens or not re.fullmatch(r"python[0-9.]*", Path(tokens[0]).name.lower()):
            return None
        mode, entry = _python_entrypoint(tokens)
        if mode != "script" or Path(entry).name not in transport_names:
            return None
        script = Path(entry)
        if not script.is_absolute():
            if cwd is None or not cwd.is_absolute():
                return "unknown"
            script = cwd / script
        from aka.core.filesystem import read_regular_bytes
        try:
            script = script.resolve(strict=True)
            if script in trusted:
                return "trusted"
            # Workspace tools are regular copies of the HTTP-only client.
            # Bound both reads; a failed read is not a content mismatch.
            canonical = resource_path('tools/sandbox.py')
            limit = 128 * 1024
            expected = read_regular_bytes(canonical, limit=limit + 1)
            if len(expected) > limit:
                return "unknown"
            actual = read_regular_bytes(script, limit=limit + 1)
            return "trusted" if actual == expected else "untrusted"
        except (OSError, ValueError, RuntimeError):
            # Includes inaccessible/deleted cwd, disappearing processes/files
            # and symlink loops. Never fall back to Path.cwd() here.
            return "unknown"

    segments = command_segments(argv)
    local_segments = []
    for segment in segments:
        status = transport_status(segment)
        if status == "untrusted":
            return "unregistered sandbox transport executed on the host"
        if status is None:
            local_segments.append(segment)
        # A verified transport's arguments belong to the remote executor. An
        # unreadable transport cannot be classified; neither permits inferring
        # a local pip/build/import from its opaque request payload.
    segments = local_segments

    def shared_gateway_mutation(segment: list[str]) -> bool:
        tokens = unwrap(segment)
        if not tokens:
            return False
        executable = Path(tokens[0]).name.lower()
        lowered = [token.lower() for token in tokens]
        protected_screen, protected_state = protected_gateway_identity(environment)
        protected_screen = protected_screen.lower()
        protected_state = protected_state.lower()

        if executable == "screen" and any(
            token == protected_screen or token.endswith("." + protected_screen)
            for token in lowered[1:]
        ):
            return True
        if executable in {"rm", "rmdir", "unlink", "shred", "truncate", "mv"} and any(
            token == protected_state
            or token.startswith(protected_state + "/")
            or token == protected_state + ".log"
            for token in lowered[1:]
        ):
            return True
        if re.fullmatch(r"python[0-9.]*", executable):
            mode, entry = _python_entrypoint(tokens)
            if (mode == "script" and Path(entry).name == "local_gateway.py") or (
                mode == "module" and entry == "aka.execution.runtime_tools.local_gateway"
            ):
                return "serve" in lowered[1:]
            if mode == "code":
                code = entry.lower()
                if protected_state in code and re.search(
                    r"(?:rmtree|unlink|remove|rename|replace|sqlite3)", code
                ):
                    return True
        if executable in {"pkill", "killall"} and any(
            "local_gateway" in token or token == protected_screen
            for token in lowered[1:]
        ):
            return True
        if executable in {"curl", "wget"} and any(
            "/v1/jobs/" in token and "/cancel" in token for token in lowered[1:]
        ):
            return True
        return False

    if any(shared_gateway_mutation(segment) for segment in segments):
        return "shared localhost gateway lifecycle/state mutation"

    if any(is_installer(segment) for segment in segments):
        return "third-party package installation/build command"

    def direct_host_gpu_action(segment: list[str]) -> str | None:
        tokens = unwrap(segment)
        if not tokens:
            return None
        lowered = [token.lower() for token in tokens]
        executable = Path(lowered[0]).name
        info_only = (
            any(token in {"--help", "-h", "--version"} for token in lowered[1:])
            or (executable == "nvcc" and "-V" in tokens[1:])
        )
        if executable in {"nvcc", "cicc", "ptxas", "fatbinary", "ninja"} and not info_only:
            return "CUDA/JIT build tool executed directly on the host"
        if executable in {"ncu", "rocprof", "rocprofv3", "compute-sanitizer"}:
            return "GPU profiler executed directly on the host"
        if re.fullmatch(r"python[0-9.]*", executable):
            mode, entry = _python_entrypoint(tokens)
            if mode == "script" and Path(entry).name in {
                "kernel.py",
                "test_kernel.py",
                "profile_driver.py",
            }:
                return "kernel/evaluator executed directly on the host"
            if mode == "module" and entry.split(".", 1)[0] in {"kernel", "test_kernel", "profile_driver"}:
                return "kernel/evaluator executed directly on the host"
            if mode == "code":
                imports = python_import_roots(entry)
                if "kernel" in imports:
                    return "kernel imported directly on the host"
                if imports & {"flashinfer", "flash_attn", "xformers", "vllm"}:
                    return "JIT-capable third-party GPU package imported directly on the host"
        if executable in {"bash", "sh", "dash", "zsh", "ksh"} and any(
            Path(token).name in {"profile_nvidia.sh", "profile_kernel.sh"}
            for token in tokens[1:]
        ):
            return "GPU profiler wrapper executed directly on the host"
        return None

    for segment in segments:
        reason = direct_host_gpu_action(segment)
        if reason is not None:
            return reason

    command = " ".join(argv).lower()
    package_build_tree = re.search(
        r"(?:^|[\s=])[^\s]*(?:pip-install-|pip-build-|pip-modern-metadata-)[^\s]*",
        command,
    )
    build_tools = {
        "cicc",
        "nvcc",
        "ninja",
        "cmake",
        "make",
        "gcc",
        "g++",
        "clang",
        "clang++",
    }
    if package_build_tree and any(
        unwrap(segment) and Path(unwrap(segment)[0]).name.lower() in build_tools
        for segment in segments
    ):
        return "compiler/build tool running in a package-manager temporary tree"
    return None
