# Sourced through BASH_ENV only inside optimizer coding-agent sessions.
# Keep shared localhost gateway lifecycle/state outside those sessions.

_atrex_gateway_screen="${ATREX_PROTECTED_GATEWAY_SCREEN:-}"
_atrex_gateway_state="${ATREX_PROTECTED_GATEWAY_STATE_DIR:-}"

_atrex_deny_gateway_action() {
    echo "atrex policy: shared localhost gateway lifecycle/state is orchestrator-owned" >&2
    return 126
}

_atrex_is_gateway_path() {
    [[ -n "${_atrex_gateway_state}" ]] || return 1
    case "$1" in
        "${_atrex_gateway_state}"|"${_atrex_gateway_state}"/*|"${_atrex_gateway_state}.log") return 0 ;;
        *) return 1 ;;
    esac
}

_atrex_is_tracked_workspace_path() {
    local value="$1" root relative
    case "$value" in
        ""|-*) return 1 ;;
    esac
    root="$(command git rev-parse --show-toplevel 2>/dev/null)" || return 1
    relative="$value"
    case "$relative" in
        "$root") relative="." ;;
        "$root"/*) relative="${relative#"$root"/}" ;;
        /*) return 1 ;;
    esac
    command git -C "$root" ls-files -- "$relative" 2>/dev/null | command grep -q .
}

_atrex_deny_tracked_delete() {
    echo "atrex policy: deleting or moving Git-tracked optimizer state is forbidden" >&2
    return 126
}

screen() {
    local value
    if [[ -n "${_atrex_gateway_screen}" ]]; then
        for value in "$@"; do
            case "$value" in
                "${_atrex_gateway_screen}"|*."${_atrex_gateway_screen}")
                    _atrex_deny_gateway_action
                    return $?
                    ;;
            esac
        done
    fi
    command screen "$@"
}

rm() {
    local value
    for value in "$@"; do
        if _atrex_is_gateway_path "$value"; then
            _atrex_deny_gateway_action
            return $?
        fi
        if _atrex_is_tracked_workspace_path "$value"; then
            _atrex_deny_tracked_delete
            return $?
        fi
    done
    command rm "$@"
}

rmdir() {
    local value
    for value in "$@"; do
        if _atrex_is_gateway_path "$value"; then
            _atrex_deny_gateway_action
            return $?
        fi
        if _atrex_is_tracked_workspace_path "$value"; then
            _atrex_deny_tracked_delete
            return $?
        fi
    done
    command rmdir "$@"
}

mv() {
    local value
    for value in "$@"; do
        if _atrex_is_gateway_path "$value"; then
            _atrex_deny_gateway_action
            return $?
        fi
        if _atrex_is_tracked_workspace_path "$value"; then
            _atrex_deny_tracked_delete
            return $?
        fi
    done
    command mv "$@"
}

truncate() {
    local value
    for value in "$@"; do
        if _atrex_is_gateway_path "$value"; then
            _atrex_deny_gateway_action
            return $?
        fi
        if _atrex_is_tracked_workspace_path "$value"; then
            _atrex_deny_tracked_delete
            return $?
        fi
    done
    command truncate "$@"
}

pkill() {
    case " $* " in
        *local_gateway*)
            _atrex_deny_gateway_action
            return $?
            ;;
    esac
    if [[ -n "${_atrex_gateway_screen}" && " $* " == *" ${_atrex_gateway_screen} "* ]]; then
        _atrex_deny_gateway_action
        return $?
    fi
    command pkill "$@"
}

killall() {
    case " $* " in
        *local_gateway*)
            _atrex_deny_gateway_action
            return $?
            ;;
    esac
    if [[ -n "${_atrex_gateway_screen}" && " $* " == *" ${_atrex_gateway_screen} "* ]]; then
        _atrex_deny_gateway_action
        return $?
    fi
    command killall "$@"
}

_atrex_guarded_python() {
    local executable="$1"
    shift
    local -a original=("$@")
    local value code="" code_index=-1 index
    for ((index = 0; index < ${#original[@]}; index++)); do
        value="${original[index]}"
        case "$value" in
            -c)
                code_index="$index"
                code="${original[index + 1]:-}"
                break
                ;;
            --)
                break
                ;;
            -*)
                shift
                ;;
            *)
                # A script path comes before any nested command arguments.
                break
                ;;
        esac
    done
    if _atrex_python_code_imports_blocked_module "$code"; then
        echo "atrex policy: host kernel/JIT-capable GPU imports must run through tools/sandbox.py" >&2
        return 126
    fi
    if (( code_index >= 0 )); then
        # Feed allowed snippets through stdin so an older, already-running
        # optimizer's process guard cannot regex-match prose embedded in the
        # command line.  Preserve interpreter flags before -c and user args
        # after the code string.  Nested sandbox commands never enter this
        # branch because their first argument is tools/sandbox.py.
        local -a prefix=("${original[@]:0:code_index}")
        local -a trailing=("${original[@]:code_index + 2}")
        printf '%s\n' "$code" | command "$executable" "${prefix[@]}" - "${trailing[@]}"
        return $?
    fi
    command "$executable" "${original[@]}"
}

_atrex_python_code_imports_blocked_module() {
    # Parse syntax instead of grepping the whole command.  Memory/search-log
    # updates often contain prose such as "import overhead kernel"; those
    # strings must not be mistaken for executable import statements.
    command python3 - "$1" <<'PY'
import ast
import sys

try:
    tree = ast.parse(sys.argv[1])
except (SyntaxError, ValueError, TypeError):
    raise SystemExit(1)

blocked = {"kernel", "flashinfer", "flash_attn", "xformers", "vllm"}

def import_roots(source, depth=0):
    try:
        parsed = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return set()
    roots = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and node.args:
            is_dynamic_import = (
                isinstance(node.func, ast.Name) and node.func.id == "__import__"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            )
            if is_dynamic_import and isinstance(node.args[0], ast.Constant):
                module = node.args[0].value
                if isinstance(module, str) and module:
                    roots.add(module.split(".", 1)[0])
            if (
                depth < 2
                and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval"}
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                roots.update(import_roots(node.args[0].value, depth + 1))
    return roots

roots = import_roots(sys.argv[1])
raise SystemExit(0 if roots & blocked else 1)
PY
}

python() {
    _atrex_guarded_python python "$@"
}

python3() {
    _atrex_guarded_python python3 "$@"
}

python3.10() {
    _atrex_guarded_python python3.10 "$@"
}

readonly -f _atrex_deny_gateway_action _atrex_is_gateway_path
readonly -f _atrex_is_tracked_workspace_path _atrex_deny_tracked_delete
readonly -f screen rm rmdir mv truncate pkill killall
readonly -f _atrex_python_code_imports_blocked_module
readonly -f _atrex_guarded_python python python3 python3.10
