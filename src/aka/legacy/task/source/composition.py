"""Own a selected Source provider without selecting CandidateWorkspace."""
from pathlib import Path
from aka.core.boot import compose, boot
from .plugin import SOURCE

PROFILES_DIR = Path(__file__).with_name("profiles")


class SourceComposition:
    def __init__(self, report):
        self.report = report
        self.disposed = False

    @property
    def source(self):
        if self.disposed:
            raise RuntimeError("source composition has been disposed")
        return self.report.service(SOURCE.name)

    def dispose(self):
        if not self.disposed:
            self.disposed = True
            self.report.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.dispose()


def compose_source(kind="kernel", config=None, *, profile="legacy", profiles_dir=PROFILES_DIR,
                   patches=(), patch_files=()):
    values = {"kind": kind, **(config or {})}
    resolved = compose(profile, profiles_dir=profiles_dir, patch_files=patch_files,
                       patches=({"id": "source", "config": values}, *patches))
    return SourceComposition(boot(resolved, tokens=(SOURCE,), required_services=(SOURCE.name,)))

