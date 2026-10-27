"""Pure path authorization for a declared filesystem view; performs no IO."""
from pathlib import Path

from .filesystem import FilesystemAccess


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("scoped paths must be absolute without parent traversal")
    return path


def _grant(access, path):
    matches = [mount for mount in access.mounts if path.is_relative_to(_absolute(mount.target))]
    if not matches:
        raise PermissionError("path is outside the declared filesystem view")
    depth = max(len(mount.target.parts) for mount in matches)
    matches = [mount for mount in matches if len(mount.target.parts) == depth]
    if len(matches) != 1:
        raise PermissionError("path has ambiguous filesystem grants")
    mount = matches[0]
    return mount, _absolute(mount.source) / path.relative_to(mount.target)


def _restriction(access, path):
    path = _absolute(path)
    sources = []
    for mount in access.mounts:
        target, source = _absolute(mount.target), _absolute(mount.source)
        if path.is_relative_to(target):
            sources.append(source / path.relative_to(target))
        elif target.is_relative_to(path):
            sources.append(source)
    return path, tuple(sources) or (path,)


def require_path_access(access: FilesystemAccess | None, path: Path, *, write: bool = False) -> Path:
    """Return the authorized source path, or raise PermissionError.

    Mount targets describe the caller's view and map to source paths. Hidden and
    read-only paths protect their mapped sources through every alias. Callers
    still own race-safe IO and symlink/hardlink checks at the publication boundary.
    An unspecified view preserves the caller's existing trusted IO behavior.
    """
    if type(write) is not bool:
        raise TypeError("path write permission must be boolean")
    if access is None:
        return Path(path)
    if not isinstance(access, FilesystemAccess):
        raise TypeError("path authorization requires FilesystemAccess")
    path = _absolute(path)
    mount, physical = _grant(access, path)
    for hidden in access.hidden_paths:
        virtual_hidden, source_hidden = _restriction(access, hidden)
        if path.is_relative_to(virtual_hidden) or any(physical.is_relative_to(source) for source in source_hidden):
            raise PermissionError("path is hidden by the declared filesystem view")
    if write:
        if not mount.writable:
            raise PermissionError("filesystem grant is read-only")
        for readonly in access.read_only_paths:
            virtual_readonly, source_readonly = _restriction(access, readonly)
            if path.is_relative_to(virtual_readonly) or any(physical.is_relative_to(source) for source in source_readonly):
                raise PermissionError("path is protected by a read-only restriction")
        if any(not grant.writable and physical.is_relative_to(_absolute(grant.source)) for grant in access.mounts):
            raise PermissionError("path aliases a read-only filesystem source")
    return physical
