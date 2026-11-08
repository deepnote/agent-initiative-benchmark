"""Verify frozen task packages and their exact native delta files."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from .contracts import (
    array,
    build_ref,
    config_ref,
    fail,
    load_config,
    load_json,
    load_task,
    object_fields,
    payload_path,
    schema_version,
    sha256,
    timestamp,
)

CHUNK_BYTES = 1024 * 1024


def hash_file(path: str | Path) -> tuple[int, str]:
    """Hash raw bytes using a fixed-size buffer; reject linked/special files."""
    path = Path(path)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            fail(path, "$", "expected a regular file without symlinks or hard links")
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                fail(path, "$", "file changed while opening")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(CHUNK_BYTES):
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        signature = lambda s: (
            s.st_dev,
            s.st_ino,
            s.st_size,
            s.st_mtime_ns,
            s.st_ctime_ns,
        )
        if signature(before) != signature(after) or signature(after) != signature(
            current
        ):
            fail(path, "$", "file changed while hashing")
        return size, digest.hexdigest()
    except OSError as exc:
        fail(path, "$", str(exc))


def _payload_files(root: Path) -> dict[str, Path]:
    """Scan without following links, and reject every unexpected root entry."""
    files = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                fail(path, "$", "symlinks are forbidden")
            if directory == root and relative == "manifest.json":
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    fail(path, "$", "manifest must be a regular, unlinked file")
                continue
            if relative == "delta" and stat.S_ISDIR(info.st_mode):
                visit(path)
                continue
            payload_path(relative, path, "$.path")
            if stat.S_ISDIR(info.st_mode) and relative.startswith("delta/"):
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                files[relative] = path
            else:
                fail(path, "$", "expected a regular file without hard links")

    try:
        if root.is_symlink() or not root.is_dir():
            fail(root, "$", "expected a package directory without symlinks")
        visit(root)
        if "task.json" not in files:
            fail(root / "task.json", "$", "required payload is missing")
        if not (root / "delta").is_dir():
            fail(root / "delta", "$", "required directory is missing")
    except OSError as exc:
        fail(root, "$", str(exc))
    return files


def check_task_config(task: dict, task_file: Path, config_path: str | Path) -> None:
    """Check the caller-supplied config, without fetching the task's URI."""
    config = load_config(config_path)
    _, digest = hash_file(config_path)
    meta = task["meta"]
    if digest != meta["config_ref"]["sha256"]:
        fail(
            task_file,
            "$.meta.config_ref.sha256",
            "does not match supplied config bytes",
        )
    if meta["build_ref"] != config["build_ref"]:
        fail(task_file, "$.meta.build_ref", "does not match supplied config")
    if timestamp(meta["snapshot_time"], task_file, "$.meta.snapshot_time") != timestamp(
        config["snapshot_time"], config_path, "$.snapshot_time"
    ):
        fail(task_file, "$.meta.snapshot_time", "does not match supplied config")


def _check_inventory(task: dict, task_file: Path, files: dict[str, Path]) -> None:
    for i, change in enumerate(task["changes"]):
        if change["file"] not in files:
            fail(task_file, f"$.changes[{i}].file", "referenced delta file is missing")


def validate_manifest(value: Any, file: str | Path = "manifest.json") -> dict:
    object_fields(
        value, {"schema_version", "build_ref", "config_ref", "files"}, file, "$"
    )
    schema_version(value["schema_version"], file, "$.schema_version")
    build_ref(value["build_ref"], file, "$.build_ref")
    config_ref(value["config_ref"], file, "$.config_ref")
    seen = set()
    for i, entry in enumerate(array(value["files"], file, "$.files")):
        field = f"$.files[{i}]"
        object_fields(entry, {"path", "bytes", "sha256"}, file, field)
        relative = payload_path(entry["path"], file, f"{field}.path")
        if relative in seen:
            fail(file, f"{field}.path", "duplicate path")
        seen.add(relative)
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            fail(file, f"{field}.bytes", "expected a non-negative integer")
        sha256(entry["sha256"], file, f"{field}.sha256")
    if "task.json" not in seen:
        fail(file, "$.files", "task.json must be listed")
    return value


def verify_package(
    package: str | Path, *, config_path: str | Path | None = None
) -> dict:
    root = Path(package)
    actual = _payload_files(root)
    manifest_file = root / "manifest.json"
    manifest = validate_manifest(load_json(manifest_file), manifest_file)
    expected = {entry["path"] for entry in manifest["files"]}
    if expected - actual.keys():
        fail(
            manifest_file,
            "$.files",
            f"missing payloads: {', '.join(sorted(expected - actual.keys()))}",
        )
    if actual.keys() - expected:
        fail(
            manifest_file,
            "$.files",
            f"unlisted payloads: {', '.join(sorted(actual.keys() - expected))}",
        )
    for i, entry in enumerate(manifest["files"]):
        size, digest = hash_file(actual[entry["path"]])
        field = f"$.files[{i}]"
        if size != entry["bytes"]:
            fail(manifest_file, f"{field}.bytes", f"size mismatch for {entry['path']}")
        if digest != entry["sha256"]:
            fail(manifest_file, f"{field}.sha256", f"hash mismatch for {entry['path']}")
    task_file = root / "task.json"
    task = load_task(task_file)
    _check_inventory(task, task_file, actual)
    for key in ("build_ref", "config_ref"):
        if manifest[key] != task["meta"][key]:
            fail(manifest_file, f"$.{key}", "must match task.json meta")
    if config_path is not None:
        check_task_config(task, task_file, config_path)
    return manifest
