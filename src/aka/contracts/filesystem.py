"""Module-neutral descriptions of caller-authorized filesystem views."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilesystemMount:
    source: Path
    target: Path
    writable: bool = False

    def __post_init__(self):
        if not isinstance(self.source, Path) or not isinstance(self.target, Path):
            raise TypeError("Filesystem mounts require Path source and target")
        if type(self.writable) is not bool:
            raise TypeError("Filesystem mount writable flag must be boolean")


@dataclass(frozen=True)
class FilesystemAccess:
    mounts: tuple[FilesystemMount, ...] = ()
    hidden_paths: tuple[Path, ...] = ()
    read_only_paths: tuple[Path, ...] = ()

    def __post_init__(self):
        for name in ("mounts", "hidden_paths", "read_only_paths"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(not isinstance(item, FilesystemMount) for item in self.mounts):
            raise TypeError("Filesystem access requires mount descriptions")
        if any(not isinstance(p, Path) for p in (*self.hidden_paths, *self.read_only_paths)):
            raise TypeError("Filesystem restrictions require Path values")
