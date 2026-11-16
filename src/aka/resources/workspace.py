"""Install selected resources and detach verified installed assets for Workspace.

Package managers may hardlink wheel files into an installation cache. These
installation inputs are distinct from Workspace's mutable file identities:
only a declared distribution's RECORD-verified assets may be detached here.
Unknown inputs and existing Workspace files retain their original validation.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import base64
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import stat
import tempfile

from aka.core.filesystem import _open_regular


_ASSETS = {
    "atrex-aka-framework": ("aka/resources/", "aka/evaluation/drivers/", "aka/evaluation/assets/", "aka/task/problem/"),
    "atrex-aka-legacy": ("aka/evaluation/drivers/", "aka/evaluation/assets/", "aka/task/problem/"),
}
_MAX_BYTES = 16 * 1024 * 1024


def _owned_assets(owners=None):
    selected = dict(_ASSETS)
    for name, prefixes in (owners or {}).items():
        if not isinstance(name, str) or not name or not isinstance(prefixes, (tuple, list)):
            raise ValueError("installed resource owners require distribution names and path prefixes")
        for prefix in prefixes:
            if (not isinstance(prefix, str) or not prefix.endswith("/") or "\\" in prefix
                    or not Path(prefix).parts or Path(prefix).is_absolute()
                    or any(part in (".", "..") for part in prefix.split("/")[:-1])
                    or Path(prefix).as_posix() + "/" != prefix):
                raise ValueError("installed resource prefixes must be package-relative directories")
        selected[name] = (*selected.get(name, ()), *prefixes)
    result = {}
    for name, prefixes in selected.items():
        try:
            owner = distribution(name)
        except PackageNotFoundError:
            continue  # Editable/source installs retain the original Workspace policy.
        for entry in owner.files or ():
            if (not entry.is_absolute() and ".." not in entry.parts
                    and any(str(entry).startswith(prefix) for prefix in prefixes)):
                result[Path(owner.locate_file(entry)).absolute()] = entry
    return result


def _read_asset(source, entry):
    signature = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                                value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    initial = source.lstat()
    with _open_regular(source) as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode)
                or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
                or before.st_size > _MAX_BYTES or signature(initial) != signature(before)):
            raise ValueError(f"invalid installed workspace resource: {source}")
        content = stream.read(_MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(content) > _MAX_BYTES or len(content) != before.st_size
            or signature(before) != signature(after) or signature(before) != signature(source.lstat())):
        raise ValueError(f"installed workspace resource changed while reading: {source}")
    expected = entry.hash
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
    if (expected is None or expected.mode != "sha256" or expected.value != digest
            or entry.size is not None and entry.size != len(content)):
        raise ValueError(f"installed workspace resource differs from wheel RECORD: {source}")
    return content, stat.S_IMODE(before.st_mode)


@contextmanager
def independent_resources(*groups, owners=None):
    """Yield independent copies of hardlinked, RECORD-verified package assets.

    Each group is an iterable of WorkspaceFile contributions. ``owners`` may
    extend the first-party policy with explicit distribution-to-directory-prefix
    mappings, e.g. ``{"my-plugin": ("my_plugin/assets/",)}``. This is an
    application-owned trust declaration, not a way to relax Workspace checks.
    Copies exist only during the synchronous call; installed/cache files are
    never modified. Text, unlinked files and undeclared paths pass unchanged.
    """
    owned = None
    with ExitStack() as scope:
        stage = None
        selected = []
        for group in groups:
            contributions = []
            for item in group:
                source = item.source
                if source is not None and source.lstat().st_nlink > 1:
                    if owned is None:
                        owned = _owned_assets(owners)
                    entry = owned.get(source.absolute())
                    if entry is not None:
                        content, mode = _read_asset(source, entry)
                        if stage is None:
                            stage = Path(scope.enter_context(tempfile.TemporaryDirectory(
                                prefix="aka-workspace-resources-"))).resolve()
                        target = stage / str(len(selected)) / str(len(contributions))
                        target.parent.mkdir(exist_ok=True)
                        with target.open("xb") as stream:
                            stream.write(content)
                        target.chmod(mode)
                        item = replace(item, source=target)
                contributions.append(item)
            selected.append(tuple(contributions))
        yield tuple(selected)


def install_workspace_resources(workspace: Path, *, plugin_registry):
    workspace = Path(workspace)
    plugin_registry.check_lock(workspace)
    plugin_registry.environment(workspace)
    return plugin_registry.install(workspace)
