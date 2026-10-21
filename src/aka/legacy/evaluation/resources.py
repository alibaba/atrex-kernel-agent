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
    source = Path(__file__).resolve().parents[4] / checkout_path
    if source.is_file():
        return source
    raise FileNotFoundError(f"Missing packaged evaluation script: {name}")


def remote_abba_script() -> Path:
    return _script("remote_abba.py", "long_horizon/remote_abba.py",
                   "evaluation/drivers/remote_abba.py")


def profile_driver_script() -> Path:
    return _script("profile_driver.py", "reference/profile_driver.py",
                   "execution/profiling/profile_driver.py")


def ncu_profile_script() -> Path:
    """Same standalone parser for legacy/observed and Light remote jobs."""
    return Path(__file__).with_name("ncu_profile.py")
