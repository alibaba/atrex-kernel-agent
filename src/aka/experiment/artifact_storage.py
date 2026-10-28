"""Private content objects with ordinary artifact paths for existing readers."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import uuid


class ArtifactCorrupt(ValueError):
    pass


@contextmanager
def _directory(path, *, create=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact directories require absolute paths")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _read(directory, name, digest, size):
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ArtifactCorrupt("invalid artifact object") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ArtifactCorrupt("invalid artifact object")
        content = bytearray()
        while True:
            data = os.read(descriptor, 64 * 1024)
            if not data:
                break
            content.extend(data)
            if len(content) > size:
                raise ArtifactCorrupt("artifact content mismatch")
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if ((current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
                or len(content) != size or hashlib.sha256(content).hexdigest() != digest):
            raise ArtifactCorrupt("artifact content mismatch")
        return bytes(content), (before.st_dev, before.st_ino)
    finally:
        os.close(descriptor)


def read(path, digest, size):
    """Old independent files and new shared files have identical read semantics."""
    path = Path(path)
    with _directory(path.parent) as directory:
        return _read(directory, path.name, digest, size)[0]


class ArtifactStorage:
    """Share new immutable content inside one configured recording root.

    Object locks serialize cooperative publishers on this host. Hard links keep
    old record formats and reader paths usable. Missing logical artifacts never
    fall back to an object, and corrupted objects are never silently rewritten.
    Unreferenced objects are retained; reclamation needs a separate policy.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.objects = self.root / ".artifact-objects" / "sha256"
        self.locks = self.root / ".artifact-objects" / "locks"

    @contextmanager
    def _object(self, content, digest):
        with _directory(self.locks, create=True) as locks:
            try:
                lock = os.open(digest, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=locks)
            except FileExistsError:
                lock = os.open(digest, os.O_RDWR | os.O_NOFOLLOW, dir_fd=locks)
            try:
                identity = os.fstat(lock)
                if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                    raise ValueError("invalid artifact object lock")
                fcntl.flock(lock, fcntl.LOCK_EX)
                with _directory(self.objects, create=True) as objects:
                    try:
                        existing, identity = _read(objects, digest, digest, len(content))
                        if (existing != content
                                or os.stat(digest, dir_fd=objects, follow_symlinks=False).st_mode & 0o222):
                            raise ArtifactCorrupt("artifact content mismatch")
                    except FileNotFoundError:
                        temporary = ".pending-" + uuid.uuid4().hex
                        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                             0o600, dir_fd=objects)
                        try:
                            with os.fdopen(descriptor, "wb") as output:
                                output.write(content)
                                output.flush()
                                os.fchmod(output.fileno(), 0o400)
                                os.fsync(output.fileno())
                                created = os.fstat(output.fileno())
                                identity = (created.st_dev, created.st_ino)
                            os.link(temporary, digest, src_dir_fd=objects, dst_dir_fd=objects,
                                    follow_symlinks=False)
                            os.fsync(objects)
                        finally:
                            os.unlink(temporary, dir_fd=objects)
                    yield objects, identity
            finally:
                os.close(lock)

    def publish(self, destination, content, digest):
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("artifact content identity mismatch")
        destination = Path(destination)
        with self._object(content, digest) as (objects, identity), \
                _directory(destination.parent, create=True) as directory:
            # Never replace an existing logical identity, including a UUID
            # collision or an unexpected file left by a previous operation.
            os.link(digest, destination.name, src_dir_fd=objects, dst_dir_fd=directory,
                    follow_symlinks=False)
            try:
                actual = os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
                if (actual.st_dev, actual.st_ino) != identity:
                    raise ArtifactCorrupt("artifact object changed during publication")
                os.fsync(directory)
            except BaseException:
                os.unlink(destination.name, dir_fd=directory)
                raise
            return identity

    @staticmethod
    def discard(destination, identity):
        """Remove only this publication's link after its record failed to commit."""
        destination = Path(destination)
        with _directory(destination.parent) as directory:
            try:
                actual = os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                return
            if (actual.st_dev, actual.st_ino) != identity:
                raise ArtifactCorrupt("artifact changed before publication rollback")
            os.unlink(destination.name, dir_fd=directory)
            os.fsync(directory)
