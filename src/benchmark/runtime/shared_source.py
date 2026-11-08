"""Writable derived files live apart from the read-only company snapshot."""

import os
from pathlib import Path
from .packages import hash_file


def derived_directory(namespace):
    if (
        not isinstance(namespace, str)
        or not namespace
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in namespace)
    ):
        raise ValueError("Invalid cache namespace")
    root = Path(os.environ.get("BENCHMARK_CACHE_DIR", ".cache/benchmark")).resolve()
    target = root / namespace
    target.mkdir(parents=True, exist_ok=True)
    return target


def selected_file_hash(path):
    return hash_file(path)[1]
