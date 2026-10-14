"""Own an independent CandidateWorkspace plugin tree."""
from pathlib import Path
from aka.core.boot import compose, boot
from .plugin import CANDIDATE

PROFILES_DIR = Path(__file__).with_name("profiles")


class CandidateComposition:
    def __init__(self, report):
        self.report = report
        self.disposed = False

    @property
    def candidate(self):
        if self.disposed:
            raise RuntimeError("candidate composition has been disposed")
        return self.report.service(CANDIDATE.name)

    def dispose(self):
        if not self.disposed:
            self.disposed = True
            self.report.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.dispose()


def compose_candidate(*, profile="legacy", profiles_dir=PROFILES_DIR, patches=(), patch_files=()):
    resolved = compose(profile, profiles_dir=profiles_dir, patches=patches, patch_files=patch_files)
    return CandidateComposition(boot(resolved, tokens=(CANDIDATE,), required_services=(CANDIDATE.name,)))

