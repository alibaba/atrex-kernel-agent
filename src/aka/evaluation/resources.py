"""Standalone GPU scripts, shared by legacy and Light evaluators.

Builds package the original checkout scripts verbatim, so they also work on a
remote GPU worker without an AKA installation.
"""
from pathlib import Path


def _script(name: str, checkout_path: str, installed_path: str) -> Path:
    packaged = Path(__file__).with_name("assets") / name
    if packaged.is_file():
        return packaged
    installed = Path(__file__).parents[1] / installed_path
    if installed.is_file():
        return installed
    source = Path(__file__).resolve().parents[3] / checkout_path
    if source.is_file():
        return source
    raise FileNotFoundError(f"Missing packaged evaluation script: {name}")


def remote_abba_script() -> Path:
    return _script("remote_abba.py", "long_horizon/remote_abba.py",
                   "evaluation/drivers/remote_abba.py")


def profile_driver_script() -> Path:
    return _script("profile_driver.py", "reference/profile_driver.py",
                   "evaluation/drivers/profile_driver.py")


def ncu_profile_script() -> Path:
    """Same standalone parser for legacy/observed and Light remote jobs."""
    return Path(__file__).with_name("ncu_profile.py")


def contribute_resources(ctx):
    from aka.contracts.agent_resources import WorkspaceResourceContribution
    ctx.contribute("workspace_resources", "evaluation-native-harness", WorkspaceResourceContribution(
        "evaluation-native-harness", native_harness_script(), visibility="runtime"))
    resource = WorkspaceResourceContribution(
        "evaluation-numerical-driver",
        Path(__file__).with_name("drivers") / "remote_numerical.py",
        visibility="runtime",
    )
    ctx.contribute("workspace_resources", resource.name, resource)
    ctx.contribute("workspace_resources", "evaluation-numerical-inputs", WorkspaceResourceContribution(
        "evaluation-numerical-inputs", Path(__file__).with_name("drivers") / "numerical_inputs.py",
        visibility="runtime"))

    ctx.contribute("workspace_resources", "execution-profile-driver", WorkspaceResourceContribution(
        "execution-profile-driver", Path(__file__).with_name("drivers") / "profile_driver.py", visibility="runtime"))


def native_harness_script() -> Path:
    """The Evaluation-owned, standalone Atrex-Bench script method."""
    return Path(__file__).with_name("drivers") / "native_harness.py"


def evaluator_harness(evaluation_input):
    """Select a method adapter, preserving an explicitly frozen Task override."""
    from aka.contracts.content import WorkspaceFile
    if evaluation_input is None:
        return None
    if evaluation_input.harness is not None:
        return evaluation_input.harness
    if evaluation_input.adapter == "legacy-native-v1":
        return WorkspaceFile("test_kernel.py", source=native_harness_script())
    return None


def validate_harness(workspace, evaluation_input):
    """Reject altered installed adapters; native typed requests need no local copy."""
    item = evaluator_harness(evaluation_input)
    if item is None:
        return
    installed = workspace / item.path
    expected = item.source.read_bytes() if item.source is not None else item.text.encode("utf-8")
    if not installed.exists() and evaluation_input.harness is None:
        return  # Evaluation can supply its own script as a prepared worker input.
    if not installed.is_file() or installed.read_bytes() != expected:
        raise ValueError("candidate evaluator harness differs from selected evaluation content")
