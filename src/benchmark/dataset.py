"""Prepare the pinned company snapshot once and share it across all tasks."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

import httpx

from .asset_download import (
    CHUNK,
    check_file,
    copy_checked,
    download,
    hub_url,
    regular,
    require,
)
from .runtime.baseline_access import BaselineCache, CacheLimits, _path, _unlinked
from .search_assets import check_search, prepare_search

DEFAULT_DATA = Path("data/pagemantle")
DEFAULT_CACHE = Path(".cache/benchmark")
FREE_RESERVE = 512 * 1024**2


class Release:
    def __init__(self, lock=None, manifest=None):
        self.lock = lock or json.loads(
            Path(__file__).with_name("assets.lock.json").read_bytes()
        )
        self.raw = (
            manifest
            if manifest is not None
            else Path(__file__).with_name("dataset-manifest.json").read_bytes()
        )
        require(
            hashlib.sha256(self.raw).hexdigest()
            == self.lock["dataset"]["manifest_sha256"],
            "Dataset manifest differs from the asset lock",
        )
        self.manifest = json.loads(self.raw)
        m = self.manifest
        require(
            m.get("schema_version") == 1
            and m["source_snapshot_sha256"]
            == self.lock["dataset"]["build_ref"]["manifest_sha256"],
            "Dataset snapshot differs from the task data",
        )
        self.packages = {}
        for row in m["packages"]:
            service = row["service"]
            require(
                re.fullmatch("[a-z][a-z0-9_]*", service)
                and service not in self.packages,
                "Invalid dataset service",
            )
            require(
                row["path"] == f"data/{service}.tar.gz"
                and row["extraction_root"] == ".",
                "Invalid archive path",
            )
            for key in ["size_bytes", "unpacked_bytes", "file_count"]:
                require(
                    type(row[key]) is int and row[key] > 0,
                    "Invalid dataset package size",
                )
            require(
                re.fullmatch("[a-f0-9]{64}", row["sha256"]), "Invalid archive checksum"
            )
            self.packages[service] = row
        for total, key in [
            ("archive_bytes", "size_bytes"),
            ("unpacked_bytes", "unpacked_bytes"),
            ("file_count", "file_count"),
        ]:
            require(
                m[total] == sum(row[key] for row in self.packages.values()),
                "Dataset totals differ",
            )
        require(
            m["file_index"]["path"] == "file-index.jsonl.gz"
            and m["file_index"]["sha256"] == self.lock["dataset"]["file_index_sha256"],
            "Dataset file index differs from the lock",
        )

    def index(self, root):
        path = Path(root) / "file-index.jsonl.gz"
        check_file(
            path,
            self.lock["dataset"]["file_index_bytes"],
            self.lock["dataset"]["file_index_sha256"],
        )
        rows, seen, totals = defaultdict(dict), set(), Counter()
        with gzip.open(path, "rt") as stream:
            for _ in range(self.manifest["file_count"]):
                line = stream.readline(4097)
                require(
                    line and len(line) <= 4096,
                    "Invalid dataset index row or file count",
                )
                row = json.loads(line)
                name = _path(row["path"])
                service = name.split("/")[0]
                require(
                    service in self.packages and "/" in name and name not in seen,
                    "Invalid or duplicate indexed path",
                )
                require(
                    type(row["size_bytes"]) is int
                    and row["size_bytes"] >= 0
                    and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
                    "Invalid file size or checksum",
                )
                seen.add(name)
                rows[service][name] = row
                totals[service] += row["size_bytes"]
            require(not stream.read(1), "Dataset index exceeds the pinned file count")
        for name, row in self.packages.items():
            require(
                len(rows[name]) == row["file_count"]
                and totals[name] == row["unpacked_bytes"],
                f"File index totals differ for {name}",
            )
        return rows

    def source(self, repository=None, revision=None):
        configured = self.lock["dataset"].get("huggingface", {})
        repository, revision = repository or configured.get(
            "repository"
        ), revision or configured.get("revision")
        require(
            repository and revision,
            "Dataset download source is not configured. Set the repository and full revision in assets.lock.json, pass --repository and --revision, or use data prepare --from PATH",
        )
        hub_url(repository, revision, "manifest.json", dataset=True)
        return repository, revision


@contextmanager
def cache_at(root):
    cache = BaselineCache(
        root,
        CacheLimits(
            max_cache_bytes=32 * 1024**3, max_file_bytes=4 * 1024**3, min_free_bytes=0
        ),
    )
    try:
        with cache.batch():
            yield cache
    finally:
        cache.close()


@contextmanager
def preparation_lock(root):
    root = Path(root).absolute()
    _unlinked(root)
    folder = root / ".benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    _unlinked(folder)
    path = folder / "prepare.lock"
    if path.exists():
        regular(path)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Dataset preparation is already running in this folder"
            ) from None
        try:
            yield root
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_service(root, rows, cache, *, full=False):
    for row in rows.values():
        path = Path(root) / row["path"]
        if full:
            check_file(path, row["size_bytes"], row["sha256"])
            cache._remember_verified(path, row["sha256"])
        else:
            cache.check(path, row["sha256"], row["size_bytes"])


def extract_service(archive, root, service, rows, cache):
    """Read only indexed regular members. Publish one complete service atomically."""
    root = Path(root)
    target = root / service
    require(
        not target.exists(),
        f"Existing {service} data is incomplete or changed. Use a fresh --dir to preserve it",
    )
    stage = root / ".benchmark" / "staging" / service
    _unlinked(stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    seen = set()
    try:
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                name = _path(member.name)
                require(
                    member.isfile()
                    and not member.issparse()
                    and name in rows
                    and name not in seen,
                    f"Unexpected or unsafe archive member: {name}",
                )
                row = rows[name]
                require(
                    member.size == row["size_bytes"],
                    f"Archive member size differs: {name}",
                )
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with stream.extractfile(member) as source, destination.open(
                    "xb"
                ) as output:
                    while block := source.read(CHUNK):
                        size += len(block)
                        require(
                            size <= row["size_bytes"],
                            f"Archive member exceeds its size: {name}",
                        )
                        output.write(block)
                        digest.update(block)
                require(
                    size == row["size_bytes"] and digest.hexdigest() == row["sha256"],
                    f"Extracted file checksum differs: {name}",
                )
                seen.add(name)
        require(seen == rows.keys(), f"Archive is missing indexed files: {service}")
        os.replace(stage / service, target)
        for row in rows.values():
            cache._remember_verified(root / row["path"], row["sha256"])
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def check_space(root, required):
    path = Path(root).absolute()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    require(
        free >= required + FREE_RESERVE,
        f"Not enough space. Need {(required+FREE_RESERVE)/1e9:.2f} GB free, have {free/1e9:.2f} GB. Choose another --dir",
    )


def prepare_data(
    root=DEFAULT_DATA,
    *,
    source=None,
    repository=None,
    revision=None,
    search_source=None,
    cache_dir=DEFAULT_CACHE,
    release=None,
    client=None,
    progress=print,
    dry_run=False,
):
    release = release or Release()
    m = release.manifest
    search_bytes = sum(row["bytes"] for row in release.lock["search"]["files"])
    progress(
        f'Dataset {m["version"]}: {m["unpacked_bytes"]/1e9:.2f} GB unpacked, {m["archive_bytes"]/1e9:.2f} GB download'
    )
    peak = (
        m["unpacked_bytes"]
        + search_bytes
        + (0 if source else max(row["size_bytes"] for row in release.packages.values()))
    )
    progress(
        f"Folder: {Path(root).absolute()} | fresh setup needs up to {(peak+FREE_RESERVE)/1e9:.2f} GB free"
    )
    if dry_run:
        return {
            "unpacked_bytes": m["unpacked_bytes"],
            "peak_free_bytes": peak + FREE_RESERVE,
        }
    remote = release.source(repository, revision) if source is None else None
    source = Path(source).absolute() if source is not None else None
    require(
        not Path(cache_dir).absolute().is_relative_to(Path(root).absolute())
        and not Path(root).absolute().is_relative_to(Path(cache_dir).absolute()),
        "Cache must be outside the dataset folder",
    )
    owned = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        with preparation_lock(root) as root, cache_at(cache_dir) as cache:
            for name, size, checksum in [
                (
                    "manifest.json",
                    len(release.raw),
                    release.lock["dataset"]["manifest_sha256"],
                ),
                (
                    "file-index.jsonl.gz",
                    release.lock["dataset"]["file_index_bytes"],
                    release.lock["dataset"]["file_index_sha256"],
                ),
            ]:
                if source is not None:
                    copy_checked(source / name, root / name, size, checksum)
                else:
                    download(
                        client,
                        hub_url(*remote, name, dataset=True),
                        root / name,
                        size,
                        checksum,
                        progress=progress,
                    )
            rows = release.index(root)
            for number, (service, package) in enumerate(release.packages.items(), 1):
                progress(f"[{number}/{len(release.packages)}] {service}: checking")
                if (root / service).exists():
                    verify_service(root, rows[service], cache)
                    cache.database.commit()
                    progress(f"{service}: ready")
                    continue
                saved_archive = (
                    root / ".benchmark/downloads" / Path(package["path"]).name
                )
                saved_partial = saved_archive.with_name(saved_archive.name + ".part")
                saved_bytes = (
                    regular(saved_archive).st_size
                    if saved_archive.exists()
                    else regular(saved_partial).st_size if saved_partial.exists() else 0
                )
                check_space(
                    root,
                    package["unpacked_bytes"]
                    + (0 if source else max(0, package["size_bytes"] - saved_bytes)),
                )
                if source is not None:
                    archive = source / package["path"]
                    check_file(archive, package["size_bytes"], package["sha256"])
                else:
                    archive = root / ".benchmark/downloads" / Path(package["path"]).name
                    download(
                        client,
                        hub_url(*remote, package["path"], dataset=True),
                        archive,
                        package["size_bytes"],
                        package["sha256"],
                        progress=progress,
                    )
                progress(f"{service}: extracting and verifying")
                extract_service(archive, root, service, rows[service], cache)
                cache.database.commit()
                if source is None:
                    archive.unlink()
            missing_search = sum(
                row["bytes"]
                for row in release.lock["search"]["files"]
                if not (
                    root
                    / ".benchmark/search"
                    / release.lock["search"]["revision"]
                    / row["path"]
                ).exists()
            )
            if missing_search:
                check_space(root, missing_search)
            prepare_search(
                root,
                client,
                release.lock["search"],
                source=search_source,
                progress=progress,
            )
            check_search(root, release.lock["search"], cache=cache)
            progress(
                f'Ready: {root} ({m["file_count"]:,} dataset files and pinned search model)'
            )
            return {
                "root": str(root),
                "files": m["file_count"],
                "unpacked_bytes": m["unpacked_bytes"],
            }
    finally:
        if owned:
            client.close()


def check_data(
    root=DEFAULT_DATA,
    *,
    cache_dir=DEFAULT_CACHE,
    full=False,
    release=None,
    progress=print,
):
    release = release or Release()
    root = Path(root).absolute()
    check_file(
        root / "manifest.json",
        len(release.raw),
        release.lock["dataset"]["manifest_sha256"],
    )
    rows = release.index(root)
    require(
        not Path(cache_dir).absolute().is_relative_to(root)
        and not root.is_relative_to(Path(cache_dir).absolute()),
        "Cache must be outside the dataset folder",
    )
    with cache_at(cache_dir) as cache:
        for service in release.packages:
            progress(f"Checking {service}")
            verify_service(root, rows[service], cache, full=full)
        check_search(root, release.lock["search"], cache=cache, full=full)
    progress(
        f'Verified {release.manifest["file_count"]:,} dataset files and search model: {root}'
    )
    return {"root": str(root), "files": release.manifest["file_count"]}
