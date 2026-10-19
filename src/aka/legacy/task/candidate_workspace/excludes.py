"""Install existing workspace/episode ignore rules without changing tracked files."""
from pathlib import Path
import subprocess
RUNTIME_DIR = '.atrex_long_horizon'
VERIFY_DIR = 'verification_artifacts/.atrex_long_horizon_verify'
LIVE_MEMORY_FILE = 'memory/live.json'
STALL_STATE_FILE = ".orchestrator_state.json"

def ensure_excluded(workspace: Path, *, wiki_trace_only: bool = False) -> None:
    """Install supervisor excludes, or the early runtime subset.

    Workspace setup happens before the framework baseline commit.  At that
    point frozen evaluator files, Wiki telemetry and private Agent runtime
    files must be excluded; the complete supervisor set would hide baseline plans.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "info/exclude"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cannot locate git exclude file: {result.stderr.strip()}")
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = workspace / path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    wiki_trace_rules = (
        "/.gpu_wiki_profile/",
        "/trace-retention-manifest.json",
    )
    agent_runtime_rules = (
        "/.aka-agent-observations/",
        "/.aka-agent-sessions/",
        "/.aka-agent-session-errors/",
    )
    # Reinstalled from the frozen Problem snapshot in every episode worktree.
    # Content/mode integrity is checked by install_files(immutable=True).
    evaluator_runtime_rules = ("/atrex-bench/",)
    supervisor_rules = (
        f"/{RUNTIME_DIR}/",
        f"/{VERIFY_DIR}/",
        f"/{STALL_STATE_FILE}",
        f"/{LIVE_MEMORY_FILE}",
        *wiki_trace_rules,
        *agent_runtime_rules,
        *evaluator_runtime_rules,
        # Episode evidence is archived by the supervisor and must never
        # become part of the candidate commit.
        "/plans/",
        "/profiles/",
        "/.humanize/",
    )
    rules = (
        (*wiki_trace_rules, *agent_runtime_rules, *evaluator_runtime_rules)
        if wiki_trace_only else supervisor_rules
    )
    missing = [rule for rule in rules if rule not in text.splitlines()]
    if missing:
        suffix = ("" if not text or text.endswith("\n") else "\n") + "\n".join(missing) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(suffix)

