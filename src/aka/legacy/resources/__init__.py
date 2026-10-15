"""Module contributions and workspace resource installation."""


def validate_runtime(root):
    """Validate the declared checkout resource tree without mirroring private paths."""
    from pathlib import Path
    root = Path(root).resolve()
    required = ("tools/sandbox.py", "reference/CLAUDE.md", "3rdparty/ncu-report-skill/SKILL.md")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("runtime resources are incomplete: " + ", ".join(missing))
