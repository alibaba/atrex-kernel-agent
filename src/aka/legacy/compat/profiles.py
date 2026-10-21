"""Named optimizer launch profiles; explicit selections remain independent."""
from importlib.resources import files
from pathlib import Path

DEFAULT_LAUNCH_PROFILE = "light"
LIGHT_PROFILES = frozenset({"observed", "sandbox", "light", "light-sandbox"})


def launch_profile_path(value=None) -> Path:
    """Resolve a fresh launch, never a recorded continuation or recovery choice."""
    name = DEFAULT_LAUNCH_PROFILE if value is None or value == "" else str(value)
    if name == "workflow":
        return Path(str(files("aka.legacy.workflow").joinpath("profiles/workflow.json")))
    if name in LIGHT_PROFILES:
        try:
            path = Path(str(files("aka.light").joinpath("profiles", f"{name}.json")))
        except ImportError as exc:
            raise ValueError(
                f"Launch profile {name!r} requires atrex-aka-light from the matching "
                "revision. Install it or explicitly select --profile workflow for Legacy; "
                "no automatic fallback is performed."
            ) from exc
        if not path.is_file():
            raise ValueError(f"Packaged launch profile is missing: {path}; reinstall matching AKA packages")
        return path
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"launch profile does not exist: {path}")
    return path
