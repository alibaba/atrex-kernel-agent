"""Existing candidate edit and protected-file rules."""
IMMUTABLE_BASELINE_PATHS = (
    "test_kernel.py",
    "reference.py",
    "input.py",
    "agent_problem.json",
    "shapes.json",
    "metadata.json",
    "roofline.json",
    "workload.jsonl",
    "definition.json",
    "valid.py",
    # The profiling entry lives outside kernel.py so no candidate rewrite can silently
    # remove the ability to profile; it is ground truth like the evaluator harness.
    "profile_driver.py",
    "memory/v0.json",
)

PROTECTED_PATHS = frozenset(
    {
        *IMMUTABLE_BASELINE_PATHS,
        "definition.json",
        "reference.py",
        "workload.jsonl",
        "test_kernel.py",
        "config.json",
        "input.py",
        "agent_problem.json",
        "shapes.json",
        "metadata.json",
        "roofline.json",
        "valid.py",
        "CLAUDE.md",
        "README.md",
        ".gitignore",
    }
)


PROTECTED_PREFIXES = (
    "memory/",
    "eval/",
    ".claude/",
    ".qoder/",
    ".agents/",
    ".atrex_",
)


EPISODE_EVIDENCE_PREFIXES = ("plans/", "profiles/", ".humanize/")


CANDIDATE_PATHS = frozenset({"kernel.py", "solution.json"})


TIMELINE_PROBE_MARKERS = (
    "cute.experimental.iket.",
    "atrex_timeline.cuh",
    "atrex::timeline::Recorder",
    "ATREX_TIMELINE_ENABLED",
    "ppu_timeline.cuh",
    "ppu_acu_profile::timeline",
    "PPU_TIMELINE_ENABLED",
)


