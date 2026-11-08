"""Pinned local source selection. No source-service or cloud connection."""

from __future__ import annotations
import copy
import hashlib
import math
import re
import stat
import json
import gzip
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from .contracts import build_ref as validate_build_ref
from .failure_causes import safe_text
from .local_objects import StorageError

CHUNK_BYTES = 1024 * 1024


class BaselineAccessError(StorageError):
    """Bounded owner diagnostic; transport exception text is never copied."""

    def __init__(self, code, reason, **details):
        self.code = code
        self.details = {
            key: safe_text(value) if isinstance(value, str) else value
            for key, value in details.items()
        }
        super().__init__(
            f"{code}: {reason}" + (f" ({self.details})" if details else "")
        )


def _path(value):
    try:
        length = len(value.encode("utf-8")) if isinstance(value, str) else 0
    except UnicodeError:
        raise BaselineAccessError("invalid_manifest", "Invalid Unicode path") from None
    if (
        not isinstance(value, str)
        or not value
        or length > 800
        or any(ord(c) < 32 or ord(c) == 127 or c in "\\:" for c in value)
        or any(not p or p.startswith(".") for p in value.split("/"))
    ):
        raise BaselineAccessError("invalid_manifest", "Unsafe relative path")
    return value


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise BaselineAccessError("invalid_limits", "Invalid integer", field=name)
    return value


@dataclass(frozen=True)
class CacheLimits:
    max_cache_bytes: int
    max_file_bytes: int
    min_free_bytes: int
    lock_wait_seconds: float = 10.0
    ready_max_bytes: int = 64 * 1024
    manifest_max_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name in (
            "max_cache_bytes",
            "max_file_bytes",
            "ready_max_bytes",
            "manifest_max_bytes",
        ):
            _integer(getattr(self, name), name)
        _integer(self.min_free_bytes, "min_free_bytes", minimum=0)
        if (
            type(self.lock_wait_seconds) not in (int, float)
            or not math.isfinite(self.lock_wait_seconds)
            or self.lock_wait_seconds <= 0
        ):
            raise BaselineAccessError(
                "invalid_limits", "lock_wait_seconds must be finite and positive"
            )


def _unlinked(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise BaselineAccessError("unsafe_cache", "Cache path contains a symlink")


@dataclass(frozen=True)
class FilePlan:
    paths: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class VerifiedPaths:
    root: Path
    build_ref: dict
    files: tuple[str, ...]
    size_bytes: int
    receipt: dict


class PinnedBaseline:
    def __init__(self, dataset, metadata_bytes, source=None, metadata_files=()):
        self.source = copy.deepcopy(source)
        self._dataset = dataset
        self._metadata_bytes = metadata_bytes
        self._metadata_files = tuple(metadata_files)
        self._metadata_file_bytes = sum(p.stat().st_size for p in self._metadata_files)
        self._ref = {
            "build_id": dataset.build_id,
            "manifest_sha256": dataset.manifest_sha256,
        }
        self._metadata = {
            "enabled_services": dataset.manifest["enabled_services"],
            "history": dataset.manifest.get("history", {}),
            "counts": dataset.manifest.get("counts", {}),
            **({"source": self.source} if self.source else {}),
        }

    @property
    def build_ref(self):
        return dict(self._ref)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def plan_service(self, service):
        _path(service)
        if "/" in service or service not in self.metadata["enabled_services"]:
            raise BaselineAccessError(
                "missing_required_file", "Service is not enabled", service=service
            )
        paths = sorted(
            path for path in self._dataset.files if path.startswith(service + "/")
        )
        if service == "slack":
            from benchmark.tools.slack.slack import PARQUET_SCHEMAS

            required = {
                f"slack/{name}.parquet"
                for name in PARQUET_SCHEMAS
                if name != "messages"
            }
            messages = {
                p
                for p in paths
                if p.startswith("slack/messages/") and p.endswith(".parquet")
            }
            if not required.issubset(paths) or not messages:
                raise BaselineAccessError(
                    "missing_required_file",
                    "Slack native inventory is incomplete",
                    service=service,
                    missing_files=sorted(required - set(paths)),
                    message_partitions_missing=not bool(messages),
                )
            paths = sorted(required | messages)
        if not paths:
            raise BaselineAccessError(
                "missing_required_file", "Service has no files", service=service
            )
        return self._plan(paths)

    def _plan(self, paths):
        paths = tuple(sorted(set(_path(p) for p in paths)))
        if not paths:
            raise BaselineAccessError("missing_required_file", "Empty file selection")
        for path in paths:
            if path not in self._dataset.files:
                raise BaselineAccessError(
                    "missing_required_file",
                    "File is not in the pinned manifest",
                    path=path,
                )
        return FilePlan(paths, sum(self._dataset.files[p].size_bytes for p in paths))

    def check_limits(self, paths):
        """Check the entire required view before starting any model work."""
        records = [self._dataset.files[p] for p in self._plan(paths).paths]
        limits = self._dataset.cache.limits
        for record in records:
            needed = max(record.size_bytes, record.object_size_bytes)
            if needed > limits.max_file_bytes:
                raise BaselineAccessError(
                    "file_too_large",
                    "Required file or stored object exceeds limit",
                    path=record.path,
                    needed=needed,
                    limit=limits.max_file_bytes,
                )
        stored = {
            r.object_key: r.object_size_bytes
            for r in records
            if r.storage_encoding is not None
        }
        needed = (
            sum(r.size_bytes for r in records)
            + sum(stored.values())
            + self.metadata_size_bytes
        )
        if needed > limits.max_cache_bytes:
            raise BaselineAccessError(
                "working_set_too_large",
                "Required files, metadata and decode scratch exceed cache",
                needed=needed,
                limit=limits.max_cache_bytes,
            )

    def acquire_service(self, service):
        return self.acquire_files(self.plan_service(service).paths)

    def file_inventory(self, paths):
        """Exact selected records from the raw hash-verified manifest."""
        return [
            self._file_receipt(self._dataset.files[p]) for p in self._plan(paths).paths
        ]

    def _file_receipt(self, record):
        result = {
            "path": record.path,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
        }
        if self.source:
            result.update(
                object_key=record.object_key,
                storage_encoding=record.storage_encoding,
                storage_size_bytes=record.object_size_bytes,
                storage_sha256=record.object_sha256,
                storage_offset_bytes=record.storage_offset_bytes,
            )
        return result

    @property
    def metadata_size_bytes(self):
        return (
            sum(len(v) for v in self._metadata_bytes.values())
            + self._metadata_file_bytes
        )

    @property
    def metadata_file_count(self):
        return len(self._metadata_bytes) + len(self._metadata_files)

    def verify_receipt(self, receipt):
        """A receipt is a claim; compare every field with the pinned manifest."""
        if (
            not isinstance(receipt, dict)
            or set(receipt)
            != (
                {"build_ref", "scope", "files"} | ({"source"} if self.source else set())
            )
            or receipt["build_ref"] != self.build_ref
            or receipt.get("source") != self.source
            or receipt["scope"] != "selected_files"
            or not isinstance(receipt["files"], list)
            or not receipt["files"]
        ):
            raise BaselineAccessError(
                "receipt_mismatch", "Invalid original baseline receipt identity"
            )
        paths = []
        for record in receipt["files"]:
            if (
                not isinstance(record, dict)
                or set(record)
                != (
                    {"path", "size_bytes", "sha256"}
                    | (
                        {
                            "object_key",
                            "storage_encoding",
                            "storage_size_bytes",
                            "storage_sha256",
                            "storage_offset_bytes",
                        }
                        if self.source
                        else set()
                    )
                )
                or type(record["size_bytes"]) is not int
                or self.source
                and (
                    type(record["storage_size_bytes"]) is not int
                    or type(record["storage_offset_bytes"]) is not int
                )
            ):
                raise BaselineAccessError(
                    "receipt_mismatch", "Invalid original file receipt"
                )
            paths.append(_path(record["path"]))
        if len(set(paths)) != len(paths):
            raise BaselineAccessError(
                "receipt_mismatch", "Duplicate original file receipt"
            )
        expected = self.file_inventory(paths)
        if sorted(receipt["files"], key=lambda r: r["path"]) != expected:
            raise BaselineAccessError(
                "receipt_mismatch",
                "Original file sizes/hashes disagree with pinned manifest",
            )
        from .service_registry import receipt_paths_valid

        if not receipt_paths_valid(self, paths):
            raise BaselineAccessError(
                "receipt_mismatch",
                "Original receipt must cover required catalogs and selected supported native files only",
            )
        return expected

    @contextmanager
    def acquire_files(self, paths):
        plan = self._plan(paths)
        self.check_limits(plan.paths)
        for path in plan.paths:
            record = self._dataset.files[path]
            self._dataset.cache.check(self._dataset.root / path, record.sha256)
        receipt = {
            "build_ref": self.build_ref,
            "scope": "selected_files",
            "files": self.file_inventory(plan.paths),
        }
        yield VerifiedPaths(
            self._dataset.root, self.build_ref, plan.paths, plan.size_bytes, receipt
        )


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(value, ref):
    if value is not None:
        raise BaselineAccessError(
            "invalid_source", "Use the pinned public dataset manifest"
        )
    return None


class BaselineCache:
    """Memoize verified file identities without copying company data."""

    shared_local = True

    def __init__(self, root, limits):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.max_bytes = limits.max_cache_bytes
        self._batch_depth = 0
        self.database = sqlite3.connect(self.root / "verified-files.sqlite")
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, expected TEXT, identity TEXT)"
        )

    def check(self, path, expected, size=None):
        path = Path(path)
        _unlinked(path)
        before = path.stat()
        if size is not None and before.st_size != size:
            raise BaselineAccessError(
                "checksum_mismatch", "Local source size changed", path=str(path)
            )
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BaselineAccessError(
                "unsafe_source", "Source must be a single regular file", path=str(path)
            )
        identity = json.dumps(
            [
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ]
        )
        saved = self.database.execute(
            "SELECT expected, identity FROM files WHERE path=?", (str(path),)
        ).fetchone()
        if saved != (expected, identity):
            after_hash = _file_hash(path)
            after = path.stat()
            after_identity = json.dumps(
                [
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ]
            )
            if after_hash != expected or after_identity != identity:
                raise BaselineAccessError(
                    "checksum_mismatch", "Local source changed", path=str(path)
                )
            self.database.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?)",
                (str(path), expected, identity),
            )
            if not self._batch_depth:
                self.database.commit()

    @contextmanager
    def batch(self):
        """Commit a dataset check once, without changing normal runtime checks."""
        self._batch_depth += 1
        try:
            yield self
        except BaseException:
            if self._batch_depth == 1:
                self.database.rollback()
            raise
        else:
            if self._batch_depth == 1:
                self.database.commit()
        finally:
            self._batch_depth -= 1

    def _remember_verified(self, path, expected):
        """Record a file just written and SHA-256 checked by the asset extractor."""
        path = Path(path)
        _unlinked(path)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise BaselineAccessError("unsafe_source", "Expected a single regular file")
        identity = json.dumps(
            [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        )
        self.database.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?)",
            (str(path), expected, identity),
        )
        if not self._batch_depth:
            self.database.commit()

    def close(self):
        self.database.close()


def open_baseline(ref, root, cache, *, manifest_sha256):
    validate_build_ref(ref, "build_ref", "$")
    root = Path(root).resolve()
    _unlinked(root)
    if cache.root.is_relative_to(root) or root.is_relative_to(cache.root):
        raise BaselineAccessError(
            "unsafe_cache", "Cache must be separate from company data"
        )
    manifest_path = root / "manifest.json"
    if _file_hash(manifest_path) != manifest_sha256:
        raise BaselineAccessError(
            "checksum_mismatch", "Dataset manifest differs from the pinned release"
        )
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("source_snapshot_sha256") != ref["manifest_sha256"]:
        raise BaselineAccessError(
            "pin_mismatch", "Dataset source differs from task source"
        )
    index_path = root / _path(manifest["file_index"]["path"])
    cache.check(index_path, manifest["file_index"]["sha256"])
    records = {}
    with gzip.open(index_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            path = _path(row["path"])
            if (
                path in records
                or type(row["size_bytes"]) is not int
                or row["size_bytes"] < 0
            ):
                raise BaselineAccessError(
                    "invalid_manifest", "Invalid or duplicate file index entry"
                )
            if not isinstance(row["sha256"], str) or not re.fullmatch(
                "[a-f0-9]{64}", row["sha256"]
            ):
                raise BaselineAccessError("invalid_manifest", "Invalid file checksum")
            records[path] = SimpleNamespace(
                path=path,
                size_bytes=row["size_bytes"],
                sha256=row["sha256"],
                object_key=path,
                object_size_bytes=row["size_bytes"],
                storage_encoding=None,
                object_sha256=row["sha256"],
                storage_offset_bytes=0,
            )
    if len(records) != manifest["file_count"]:
        raise BaselineAccessError(
            "invalid_manifest", "File index count differs from manifest"
        )
    dataset = SimpleNamespace(
        root=root,
        cache=cache,
        files=records,
        build_id=ref["build_id"],
        manifest_sha256=ref["manifest_sha256"],
        manifest={"enabled_services": sorted({p.split("/")[0] for p in records})},
    )
    return PinnedBaseline(dataset, {"manifest.json": raw}, metadata_files=(index_path,))
