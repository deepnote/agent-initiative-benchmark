"""Save immutable completed benchmark bundles on local disk."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable

from .local_objects import LocalObjects, StorageConflict, StorageError

COMPLETE = "_complete.json"
PAYLOAD = "_files"
MAX_MARKER_BYTES = 4 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024


def _safe_reason(error: Exception, secrets: tuple[bytes, ...] = ()) -> str:
    """Keep validation reasons or SDK codes, never arbitrary SDK messages."""
    chain, seen = [], set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__

    def redact(text: str) -> str:
        text = re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+", "[URL redacted]", text)
        text = re.sub(
            r"[^\s]*[?&](?:X-Amz-|X-Goog-|Signature=)[^\s]*",
            "[URL redacted]",
            text,
            flags=re.IGNORECASE,
        )
        for secret in secrets:
            text = text.replace(secret.decode("utf-8"), "[credential redacted]")
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        return text if len(text) <= 512 else text[:255] + "..." + text[-254:]

    def identifier(value: Any) -> str | None:
        if (
            isinstance(value, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value)
            and redact(value) == value
        ):
            return value
        return None

    for cause in reversed(chain):
        response = getattr(cause, "response", None)
        details = response.get("Error") if isinstance(response, dict) else None
        code = identifier(details.get("Code")) if isinstance(details, dict) else None
        if code:
            return code
    cause = chain[-1]
    kind = identifier(type(cause).__name__) or "Exception"
    if isinstance(cause, StorageError):
        return redact(str(cause))
    if isinstance(cause, json.JSONDecodeError):
        return "Invalid JSON: " + redact(cause.msg)
    if isinstance(cause, OSError) and cause.strerror:
        return kind + ": " + redact(cause.strerror)
    return kind


def _parent_directories(paths: list[str]) -> set[str]:
    return {
        "/".join(path.split("/")[:i])
        for path in paths
        for i in range(1, len(path.split("/")))
    }


def _directories(record: dict[str, Any]) -> list[str]:
    # Version 1 stored no directories; only file parents can be recovered.
    if "directories" in record:
        return record["directories"]
    return sorted(_parent_directories([f["path"] for f in record["files"]]))


def _relative(value: str, *, unit: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise StorageError("Expected a nonempty relative path of at most 800 bytes.")
    try:
        length = len(value.encode("utf-8"))
    except UnicodeError:
        raise StorageError("Relative paths must be valid UTF-8.") from None
    if length > 800:
        raise StorageError("Expected a nonempty relative path of at most 800 bytes.")
    parts = value.split("/")
    if (
        any(part in {"", ".", "..", COMPLETE} for part in parts)
        or any(ord(c) < 32 or ord(c) == 127 or c in "\\:" for c in value)
        or (unit and PAYLOAD in parts)
    ):
        raise StorageError("Unsafe or reserved relative path.")
    return value


def _canonical(value: Any) -> bytes:
    def check_keys(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise StorageError("JSON object keys must be strings.")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)

    check_keys(value)
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise StorageError(
            "Expected JSON data with finite numbers and string keys."
        ) from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StorageError("Duplicate JSON key in completion record.")
        result[key] = value
    return result


def validate_completion_manifest(
    record: Any,
    unit: str,
    *,
    prefix: str = "benchmark/",
    secrets: tuple[bytes, ...] = (),
) -> dict[str, Any]:
    """Validate an imported marker without constructing or contacting a store."""

    def check_key(path):
        if len((prefix + unit + "/" + PAYLOAD + "/" + path).encode("utf-8")) > 1024:
            raise StorageError("Artifact object key exceeds 1024 bytes.")

    if not isinstance(record, dict):
        raise StorageError("Invalid completion record fields.")
    version = record.get("schema_version")
    if (
        type(version) is not int
        or version not in (1, 2)
        or record.get("unit_key") != unit
    ):
        raise StorageError("Invalid completion version or unit identity.")
    fields = {"schema_version", "unit_key", "metadata", "files"}
    if version == 2:
        fields.add("directories")
    if set(record) != fields:
        raise StorageError("Invalid completion record fields.")
    if not isinstance(record["metadata"], dict) or not isinstance(
        record["files"], list
    ):
        raise StorageError("Completion metadata must be an object and files a list.")
    paths: list[str] = []
    for entry in record["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise StorageError("Invalid completion file fields.")
        path = _relative(entry["path"])
        check_key(path)
        if (
            type(entry["bytes"]) is not int
            or entry["bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise StorageError("Invalid completion file size or SHA-256.")
        paths.append(path)
    known = set(paths)
    if paths != sorted(known) or any(
        "/".join(path.split("/")[:i]) in known
        for path in paths
        for i in range(1, len(path.split("/")))
    ):
        raise StorageError(
            "Completion paths must be sorted, unique, and have no file/directory conflicts."
        )
    if version == 2:
        directories = record["directories"]
        if not isinstance(directories, list):
            raise StorageError("Completion directories must be a list.")
        for path in directories:
            _relative(path)
            check_key(path)
        known_dirs = set(directories)
        if directories != sorted(known_dirs):
            raise StorageError("Completion directories must be sorted and unique.")
        if (
            known & known_dirs
            or not _parent_directories(paths + directories) <= known_dirs
        ):
            raise StorageError(
                "Completion files and directories conflict or have missing parents."
            )
    encoded = _canonical(record)
    if any(secret in encoded for secret in secrets):
        raise StorageError("Refusing artifact data containing a configured credential.")
    return record


class ArtifactStore:
    """One writer per unit; a verified deterministic marker commits the payload."""

    def __init__(self, root, prefix="benchmark/", *, secrets=()):
        self.store = LocalObjects(root)
        self.bucket = "local"
        self.prefix = _relative(prefix.removesuffix("/"), unit=True) + "/"
        self._secrets = tuple(
            s.encode() if isinstance(s, str) else s for s in secrets if s
        )

    def _no_secrets(self, value: bytes) -> None:
        if any(secret in value for secret in self._secrets):
            raise StorageError(
                "Refusing artifact data containing a configured credential."
            )

    def _unit(self, unit_key: str) -> str:
        unit = _relative(unit_key, unit=True)
        self._no_secrets(unit.encode("utf-8"))
        return unit

    def _key(self, unit: str, path: str = COMPLETE) -> str:
        key = self.prefix + unit + "/" + path
        if len(key.encode("utf-8")) > 1024:
            raise StorageError("Artifact object key exceeds 1024 bytes.")
        return key

    def _call(
        self, operation: str, unit: str, path: str, call: Callable[[], Any]
    ) -> Any:
        try:
            return call()
        except Exception as error:
            reason = _safe_reason(error, self._secrets)
            raise StorageError(
                f"{operation} failed for unit {unit}, path {path}: {reason}"
            ) from None

    def _inventory(
        self, root: Path, unit: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if root.is_symlink() or not root.is_dir():
            raise StorageError(
                f"publish failed for unit {unit}: source must be a directory, not a symlink."
            )
        files, directories = [], []

        def walk_error(error: OSError) -> None:
            reason = _safe_reason(error, self._secrets)
            raise StorageError(
                f"inspect failed for unit {unit}: cannot read source directory: {reason}"
            ) from None

        for directory, dirs, names in os.walk(
            root, followlinks=False, onerror=walk_error
        ):
            for name in sorted(dirs + names):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                self._no_secrets(relative.encode("utf-8", errors="replace"))
                self._call("inspect", unit, relative, lambda: _relative(relative))
                self._key(unit, PAYLOAD + "/" + relative)
                mode = self._call(
                    "inspect", unit, relative, lambda: path.lstat().st_mode
                )
                if stat.S_ISDIR(mode):
                    directories.append(relative)
                    continue
                if not stat.S_ISREG(mode):
                    raise StorageError(
                        f"inspect failed for unit {unit}, path {relative}: only regular files allowed."
                    )
                size, digest = self._call(
                    "inspect", unit, relative, lambda: self._hash_file(path)
                )
                files.append({"path": relative, "bytes": size, "sha256": digest})
        return sorted(files, key=lambda item: item["path"]), sorted(directories)

    def _hash_file(self, path: Path) -> tuple[int, str]:
        digest, size, previous = hashlib.sha256(), 0, b""
        overlap = max((len(secret) for secret in self._secrets), default=1) - 1
        before = path.stat()
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                self._no_secrets(previous + chunk)
                previous = chunk[-overlap:] if overlap else b""
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise StorageError("Source file changed during inspection.")
        return size, digest.hexdigest()

    def _validate(self, record: Any, unit: str) -> dict[str, Any]:
        return validate_completion_manifest(
            record, unit, prefix=self.prefix, secrets=self._secrets
        )

    def publish_directory(
        self,
        local_dir: str | Path,
        unit_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        unit, root = self._unit(unit_key), Path(local_dir)
        # Roundtrip makes metadata independent of later caller mutation.
        metadata = json.loads(_canonical({} if metadata is None else metadata))
        files, directories = self._inventory(root, unit)
        record = self._validate(
            {
                "schema_version": 2,
                "unit_key": unit,
                "metadata": metadata,
                "files": files,
                "directories": directories,
            },
            unit,
        )
        marker = _canonical(record)
        if len(marker) > MAX_MARKER_BYTES:
            raise StorageError(
                f"publish failed for unit {unit}: completion exceeds 4 MiB."
            )
        existing = self.read_completion(unit)
        if existing is not None:
            comparable = {
                **existing,
                "schema_version": 2,
                "directories": _directories(existing),
            }
            if _canonical(comparable) != marker:
                raise StorageConflict(
                    f"publish conflict for completed unit {unit}: files, directories, or metadata differ."
                )
        for entry in record["files"]:
            relative = entry["path"]
            key = self._key(unit, PAYLOAD + "/" + relative)
            if existing is None:
                self._call(
                    "upload",
                    unit,
                    relative,
                    lambda: self.store.put_file_immutable(
                        self.bucket, key, root / relative, entry["sha256"]
                    ),
                )
            # Reused objects must be checked by content, not just stored metadata.
            self._call(
                "verify",
                unit,
                relative,
                lambda: self.store.verify_object(
                    self.bucket, key, entry["bytes"], entry["sha256"], full=True
                ),
            )
        if self._inventory(root, unit) != (files, directories):
            raise StorageError(
                f"publish failed for unit {unit}: source directory changed."
            )
        if existing is None:
            self._call(
                "commit",
                unit,
                COMPLETE,
                lambda: self.store.put_bytes_immutable(
                    self.bucket,
                    self._key(unit),
                    marker,
                    content_type="application/json",
                ),
            )
        return existing if existing is not None else record

    def read_completion(self, unit_key: str) -> dict[str, Any] | None:
        unit = self._unit(unit_key)
        key = self._key(unit)

        def read() -> dict[str, Any] | None:
            head = self.store._head(self.bucket, key)
            if head is None:
                return None
            size = int(head.get("ContentLength", -1))
            if not 0 < size <= MAX_MARKER_BYTES:
                raise StorageError("Completion size is outside the allowed range.")
            chunks = []
            chunk_size = min(self.store.config.multipart_chunk, HASH_CHUNK_BYTES)
            for start in range(0, size, chunk_size):
                chunks.append(
                    self.store._read_range(
                        self.bucket, key, start, min(size - 1, start + chunk_size - 1)
                    )
                )
            value = b"".join(chunks)
            if hashlib.sha256(value).hexdigest() != self.store._metadata_sha(head):
                raise StorageError("Completion checksum mismatch.")
            return self._validate(
                json.loads(value, object_pairs_hook=_unique_object), unit
            )

        return self._call("read completion", unit, COMPLETE, read)

    def is_complete(self, unit_key: str) -> bool:
        # Malformed or unreadable markers raise, rather than look like missing work.
        return self.read_completion(unit_key) is not None

    def download_directory(
        self, unit_key: str, destination: str | Path, *, before_file=None
    ) -> dict[str, Any]:
        unit = self._unit(unit_key)
        record = self.read_completion(unit)
        if record is None:
            raise StorageError(
                f"download failed for unit {unit}: completion is missing."
            )
        target = Path(os.path.abspath(destination))
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise StorageError(
                f"download failed for unit {unit}: destination contains a symlink."
            )
        if target.exists():
            raise StorageError(
                f"download failed for unit {unit}: destination already exists."
            )
        self._call(
            "prepare download",
            unit,
            "destination",
            lambda: target.parent.mkdir(parents=True, exist_ok=True),
        )
        staging = self._call(
            "prepare download",
            unit,
            "destination",
            lambda: Path(
                tempfile.mkdtemp(prefix=".benchmark-download-", dir=target.parent)
            ),
        )
        try:
            for relative in _directories(record):
                self._call(
                    "create directory",
                    unit,
                    relative,
                    lambda: (staging / relative).mkdir(parents=True, exist_ok=True),
                )
            for entry in record["files"]:
                if before_file is not None:
                    before_file(entry)
                relative = entry["path"]
                key = self._key(unit, PAYLOAD + "/" + relative)
                self._call(
                    "inspect download",
                    unit,
                    relative,
                    lambda: self.store.verify_object(
                        self.bucket, key, entry["bytes"], entry["sha256"]
                    ),
                )
                self._call(
                    "download",
                    unit,
                    relative,
                    lambda: self.store.download_verified(
                        self.bucket,
                        key,
                        staging / relative,
                        entry["bytes"],
                        entry["sha256"],
                    ),
                )
            # A same-filesystem rename exposes the directory as one completed bundle.
            if target.exists() or target.is_symlink():
                raise StorageError(
                    f"download failed for unit {unit}: destination appeared during download."
                )
            self._call(
                "finish download", unit, "destination", lambda: staging.rename(target)
            )
        finally:
            if staging.exists():
                self._call(
                    "cleanup download", unit, "staging", lambda: shutil.rmtree(staging)
                )
        return record

    def list_completed(self, group_prefix: str = "") -> list[dict[str, Any]]:
        group = self._unit(group_prefix.removesuffix("/")) if group_prefix else ""
        prefix = self.prefix + (group + "/" if group else "")
        keys = self._call(
            "list",
            group or "(all)",
            COMPLETE,
            lambda: self.store.list_keys(self.bucket, prefix),
        )
        results = []
        for key in keys:
            if not key.startswith(prefix) or not key.endswith("/" + COMPLETE):
                continue
            unit = key[len(self.prefix) : -len("/" + COMPLETE)]
            # Payload names cannot masquerade as unit completion markers.
            record = self.read_completion(unit)
            if record is not None:
                results.append(record)
        return sorted(results, key=lambda item: item["unit_key"])
