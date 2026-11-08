"""Bounded, resumable public file downloads. No global Hub cache."""

import hashlib
import os
from pathlib import Path
import re
import stat
from urllib.parse import quote

import httpx

from .runtime.baseline_access import _unlinked
from .runtime.failure_causes import RuntimeInputError

CHUNK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeInputError(message)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    _unlinked(path)
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        f"Expected a regular file: {path.name}",
    )
    return info


def check_file(path, size, checksum):
    before = regular(path)
    require(before.st_size == size, f"File size differs: {path.name}")
    require(hash_file(path) == checksum, f"File checksum differs: {path.name}")
    require(
        identity(before) == identity(regular(path)),
        f"File changed while checking: {path.name}",
    )


def identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def hub_url(repository, revision, name, *, dataset=False):
    require(
        isinstance(repository, str)
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository
        ),
        "Set a Hugging Face repository as organization/name",
    )
    require(
        isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}", revision),
        "Set the full 40-character Hugging Face commit revision",
    )
    return f'https://huggingface.co/{"datasets/" if dataset else ""}{repository}/resolve/{revision}/{quote(name, safe="/")}'


def copy_checked(source, target, size, checksum):
    source, target = Path(source), Path(target)
    if target.exists():
        check_file(target, size, checksum)
        return
    check_file(source, size, checksum)
    target.parent.mkdir(parents=True, exist_ok=True)
    _unlinked(target)
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        regular(temporary)
        temporary.unlink()
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as src, temporary.open("xb") as dst:
        while block := src.read(CHUNK):
            count += len(block)
            require(count <= size, f"Source exceeds expected size: {source.name}")
            digest.update(block)
            dst.write(block)
    require(
        count == size and digest.hexdigest() == checksum,
        f"Source changed while copying: {source.name}",
    )
    os.replace(temporary, target)


def finish_download(partial, target, size, checksum):
    try:
        check_file(partial, size, checksum)
    except RuntimeInputError:
        partial.unlink()
        raise RuntimeInputError(
            f"Download checksum differs for {target.name}. The corrupt partial was removed. Run the same command to download again"
        ) from None
    os.replace(partial, target)


def download(client, url, target, size, checksum, *, progress=lambda value: None):
    """Keep a partial file on interruption. Publish only the complete pinned bytes."""
    target = Path(target)
    _unlinked(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check_file(target, size, checksum)
        return
    partial = target.with_name(target.name + ".part")
    offset = regular(partial).st_size if partial.exists() else 0
    require(
        offset <= size, f"Partial download is too large. Remove {partial} and retry"
    )
    if offset == size:
        finish_download(partial, target, size, checksum)
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        progress(f"Resuming {target.name} at {offset:,} bytes")
    try:
        with client.stream(
            "GET", url, headers=headers, follow_redirects=True, timeout=60
        ) as response:
            require(
                response.status_code in {200, 206},
                f"Download failed for {target.name}: HTTP {response.status_code}. Run the same command to resume",
            )
            if response.status_code == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("content-range", ""),
                )
                require(
                    match is not None
                    and int(match[1]) == offset
                    and int(match[3]) == size
                    and offset <= int(match[2]) < size,
                    f"Invalid resumed response for {target.name}",
                )
            else:
                offset = 0
            require(
                response.headers.get("content-encoding", "identity") == "identity",
                f"Unexpected HTTP content encoding for {target.name}",
            )
            mode = "ab" if offset else "wb"
            count = offset
            last = count
            with partial.open(mode) as stream:
                for block in response.iter_bytes(CHUNK):
                    count += len(block)
                    require(
                        count <= size, f"Download exceeds expected size: {target.name}"
                    )
                    stream.write(block)
                    if count - last >= 64 * CHUNK:
                        progress(f"{target.name}: {count / size:.0%}")
                        last = count
        require(
            count == size,
            f"Download interrupted for {target.name}. Run the same command to resume",
        )
    except httpx.HTTPError:
        raise RuntimeInputError(
            f"Download interrupted for {target.name}. Run the same command to resume"
        ) from None
    finish_download(partial, target, size, checksum)
