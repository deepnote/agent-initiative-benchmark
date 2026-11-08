"""Small local transport for the existing immutable result-store contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace


class StorageError(RuntimeError):
    pass


class StorageConflict(StorageError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalObjects:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise StorageError("Result path cannot contain symlinks")
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = SimpleNamespace(multipart_chunk=1024 * 1024)

    def _path(self, bucket, key):
        parts = key.split("/")
        if not parts or any(
            p in {"", ".", ".."} or "\\" in p or ":" in p for p in parts
        ):
            raise StorageError("Invalid result path")
        path = self.root.joinpath(*parts)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise StorageError("Result path cannot contain symlinks")
        return path

    def _head(self, bucket, key):
        path = self._path(bucket, key)
        return (
            {"ContentLength": path.stat().st_size, "sha256": file_hash(path)}
            if path.is_file()
            else None
        )

    @staticmethod
    def _metadata_sha(head):
        return head["sha256"]

    def _read_range(self, bucket, key, start, end):
        with self._path(bucket, key).open("rb") as stream:
            stream.seek(start)
            return stream.read(end - start + 1)

    def _put(self, bucket, key, writer, expected):
        target = self._path(bucket, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".writing-", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            if file_hash(temporary) != expected:
                raise StorageError("Result source changed during copying")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if not target.is_file() or file_hash(target) != expected:
                    raise StorageConflict(
                        "An existing immutable result has different bytes"
                    )
        finally:
            temporary.unlink(missing_ok=True)

    def put_file_immutable(self, bucket, key, path, expected):
        def copy(stream):
            with Path(path).open("rb") as source:
                shutil.copyfileobj(source, stream, 1024 * 1024)

        self._put(bucket, key, copy, expected)

    def put_bytes_immutable(self, bucket, key, value, **kwargs):
        self._put(
            bucket,
            key,
            lambda stream: stream.write(value),
            hashlib.sha256(value).hexdigest(),
        )

    def verify_object(self, bucket, key, size, digest, **kwargs):
        path = self._path(bucket, key)
        if (
            not path.is_file()
            or path.stat().st_size != size
            or file_hash(path) != digest
        ):
            raise StorageError("Saved result checksum mismatch")

    def download_verified(self, bucket, key, destination, size, digest):
        self.verify_object(bucket, key, size, digest)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._path(bucket, key).open("rb") as source, destination.open(
            "xb"
        ) as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if file_hash(destination) != digest:
            raise StorageError("Result copy checksum mismatch")

    def list_keys(self, bucket, prefix):
        root = self.root.joinpath(*prefix.rstrip("/").split("/"))
        return (
            sorted(
                str(p.relative_to(self.root)) for p in root.rglob("*") if p.is_file()
            )
            if root.exists()
            else []
        )
