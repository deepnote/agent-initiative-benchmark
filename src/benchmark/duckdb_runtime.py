"""Disk-backed DuckDB scratch databases with conservative memory defaults.

Scratch files are private to one connection and removed on close.
Durable outputs are ordinary Parquet files.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import json
import subprocess
import threading
import gc
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import duckdb as _duckdb

DEFAULT_BUFFER_CAP = 512 * 1024**2


def _cgroup_memory_files():
    """Include ancestor limits: a job can share its parent's allowance."""
    files = {
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    }
    try:
        cgroups = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        cgroups = []
    for line in cgroups:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if not controllers:
            root, filename = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            root, filename = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        directory = (root / relative.lstrip("/")).resolve()
        while directory.is_relative_to(root):
            files.add(directory / filename)
            if directory == root:
                break
            directory = directory.parent
    return files


def _command_output(*arguments):
    try:
        return subprocess.check_output(
            arguments, text=True, timeout=2, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def _machine_memory_bytes():
    """Use physical RAM and the smallest container limit, with a safe fallback."""
    limits = []
    try:
        physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if physical > 0:
            limits.append(physical)
    except (ValueError, OSError):
        pass
    if sys.platform == "darwin" and not limits:
        value = _command_output("/usr/sbin/sysctl", "-n", "hw.memsize").strip()
        if value.isdigit() and int(value) > 0:
            limits.append(int(value))
    for path in _cgroup_memory_files():
        try:
            value = path.read_text().strip()
            if value != "max" and int(value) > 0:
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return min(limits) if limits else 1024**3


def _available_memory_bytes():
    """Count usable RAM, container headroom, and remaining address space."""
    limits = [_machine_memory_bytes()]
    measured = False
    if sys.platform == "linux":
        try:
            match = re.search(
                r"^MemAvailable:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text(), re.M
            )
            if match:
                limits.append(int(match[1]) * 1024)
                measured = True
        except OSError:
            pass
    elif sys.platform == "darwin":
        output = _command_output("/usr/bin/vm_stat")
        page = re.search(r"page size of (\d+) bytes", output)
        counts = re.findall(
            r"^Pages (?:free|inactive|speculative):\s+(\d+)\.", output, re.M
        )
        if page and counts:
            limits.append(sum(map(int, counts)) * int(page[1]))
            measured = True
    for path in _cgroup_memory_files():
        usage_name = (
            "memory.current" if path.name == "memory.max" else "memory.usage_in_bytes"
        )
        try:
            maximum = path.read_text().strip()
            used = int(path.with_name(usage_name).read_text().strip())
            if maximum != "max" and int(maximum) > 0:
                # Cgroup usage includes reclaimable file cache from large input
                # scans and checkpoints. Count only clean inactive file pages,
                # never anonymous memory, dirty pages or pending writes.
                limits.append(
                    max(0, int(maximum) - used + _reclaimable_cgroup_cache(path, used))
                )
                measured = True
        except (OSError, ValueError):
            pass
    if sys.platform == "linux":
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            virtual = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            limits.extend(
                max(0, value - virtual)
                for value in (soft, hard)
                if value != resource.RLIM_INFINITY
            )
        except (OSError, ValueError, IndexError):
            pass
    if not measured:
        limits.append(1024**3)
    return min(limits)


def _reclaimable_cgroup_cache(limit_path, used):
    try:
        values = {
            key: int(value)
            for key, value in (
                line.split()
                for line in limit_path.with_name("memory.stat").read_text().splitlines()
            )
        }
        prefix = "" if limit_path.name == "memory.max" else "total_"
        inactive = values.get(prefix + "inactive_file", 0)
        dirty = values.get(prefix + "file_dirty", values.get(prefix + "dirty", 0))
        writeback = values.get(
            prefix + "file_writeback", values.get(prefix + "writeback", 0)
        )
        return max(0, min(used, inactive - dirty - writeback))
    except (OSError, ValueError):
        return 0


@dataclass(frozen=True)
class MemoryAllowance:
    working_bytes: int
    limit_bytes: int


class _MemoryPool:
    """Reservations shared by every scratch database in this process.

    Plan for half of available RAM. Reserve at most three quarters for buffers,
    including overruns. A connection cannot consume another connection's share.
    Recheck free RAM on admission and start a fresh pool when all owners close.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.reservations = {}

    def acquire(self, requested=None):
        with self.lock:
            available = _available_memory_bytes()
            if not self.reservations:
                self.working = available // 2
                self.ceiling = available * 3 // 4
            try:
                workers = int(os.environ.get("BENCHMARK_SERVICE_CONCURRENCY", "3"))
            except ValueError as error:
                raise ValueError(
                    "BENCHMARK_SERVICE_CONCURRENCY must be an integer"
                ) from error
            if not 1 <= workers <= 8:
                raise ValueError(
                    "BENCHMARK_SERVICE_CONCURRENCY must be between 1 and 8"
                )
            remaining = min(
                self.ceiling - sum(self.reservations.values()), available * 3 // 4
            )
            # Leave half of unclaimed buffers for another or nested owner.
            # Waiting for a slot here could deadlock a job opening a second DB.
            working = min(self.working // workers, remaining // 3)
            ceiling = min(working * 3 // 2, remaining)
            if requested is not None:
                ceiling = _memory_setting_bytes(requested)
                if ceiling > remaining:
                    raise MemoryError(
                        "Requested DuckDB memory exceeds the remaining shared allowance."
                    )
                working = min(working, ceiling * 2 // 3)
            else:
                ceiling = min(ceiling, DEFAULT_BUFFER_CAP)
                working = min(working, ceiling * 2 // 3)
            if ceiling < 16 * 1024**2:
                raise MemoryError(
                    "Insufficient available memory for a scratch database; "
                    f"available={available}B shared_remaining={remaining}B "
                    f"active_connections={len(self.reservations)}. Saved work is unchanged."
                )
            token = object()
            self.reservations[token] = ceiling
            return token, MemoryAllowance(working, ceiling)

    def release(self, token):
        with self.lock:
            self.reservations.pop(token, None)


def _memory_setting_bytes(value):
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*", str(value), re.I
    )
    if not match:
        raise ValueError(
            "DuckDB memory_limit must be a positive size in bytes, KB/MB/GB or KiB/MiB/GiB."
        )
    unit = match[2].upper()
    powers = {
        "B": 1,
        **{prefix + "B": 1000**power for power, prefix in enumerate("KMGT", 1)},
        **{prefix + "IB": 1024**power for power, prefix in enumerate("KMGT", 1)},
    }
    result = int(float(match[1]) * powers[unit])
    if result <= 0:
        raise ValueError("DuckDB memory_limit must be positive.")
    return result


_memory_pool = _MemoryPool()


def _reclaim_process_memory():
    """Release dead Python objects and free allocator pages before one retry."""
    gc.collect()
    if sys.platform == "linux":
        try:
            import ctypes

            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
        except (AttributeError, OSError):
            pass


def limit_process_memory():
    """Reserve one quarter of the machine or container RAM for other work.

    This is a final address-space ceiling, not a substitute for streaming.
    Keep any stricter limit imposed by the environment.
    """
    if sys.platform != "linux":
        return
    import resource

    ceiling = _machine_memory_bytes() * 3 // 4
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        ceiling = min(ceiling, hard)
    if soft != resource.RLIM_INFINITY:
        ceiling = min(ceiling, soft)
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))


def __getattr__(name):
    return getattr(_duckdb, name)


def _check_scratch_disk(parent):
    """A tmpfs spill directory consumes RAM and defeats the memory budget."""
    mounts = Path("/proc/self/mountinfo")
    if sys.platform != "linux" or not mounts.is_file():
        return
    destination = Path(parent).resolve()
    selected = None
    for line in mounts.read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), before.split()[4])
        if destination.is_relative_to(mount) and (
            selected is None or len(mount) > selected[0]
        ):
            selected = (len(mount), after.split()[0])
    if selected and selected[1] in {"tmpfs", "ramfs"}:
        raise RuntimeError(
            "DuckDB scratch is on a RAM-backed filesystem. Set "
            "BENCHMARK_SCRATCH_DIR to a local disk directory."
        )


class Connection:
    def __init__(self, connection, scratch, token, allowance):
        self._connection = connection
        self._scratch = scratch
        self._explicit_transaction = False
        self._memory_token = token
        self.memory_allowance = allowance

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __enter__(self):
        return self

    def execute(self, query, parameters=None, **kwargs):
        self._connection.execute(query, parameters or [], **kwargs)
        command = query.strip().split(None, 1)[0].rstrip(";").upper()
        if command in {"BEGIN", "START"}:
            self._explicit_transaction = True
        elif command in {"COMMIT", "ROLLBACK", "END"}:
            self._explicit_transaction = False
        return self

    def executemany(self, query, parameters):
        # Disk-backed inserts otherwise commit and fsync once for every row.
        if self._explicit_transaction:
            self._connection.executemany(query, parameters)
            return self
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.executemany(query, parameters)
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return self

    def commit(self):
        self._connection.commit()
        self._explicit_transaction = False
        return self

    def rollback(self):
        self._connection.rollback()
        self._explicit_transaction = False
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._memory_token is None:
            return
        try:
            self._connection.close()
        finally:
            try:
                if self._scratch is not None:
                    self._scratch.cleanup()
            finally:
                self._scratch = None
                _memory_pool.release(self._memory_token)
                self._memory_token = None

    def __del__(self):
        if getattr(self, "_memory_token", None) is not None:
            self.close()


def connect(database=":memory:", *, read_only=False, config=None, **kwargs):
    """Share an adaptive working budget and bounded reserve across connections.

    Use disk for temporary tables too: a DuckDB memory limit alone cannot make
    large in-memory tables spill. Preserve row order for deterministic snapshots.
    """
    parent = os.environ.get("BENCHMARK_SCRATCH_DIR")
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    _check_scratch_disk(parent or tempfile.gettempdir())
    settings = dict(config or {})
    requested = settings.pop("max_memory", settings.get("memory_limit"))
    if "memory_limit" in settings and requested is not None:
        requested = f'{min(_memory_setting_bytes(requested), _memory_setting_bytes(settings["memory_limit"]))}B'
    reclaimed = False
    try:
        token, allowance = _memory_pool.acquire(requested)
    except MemoryError:
        _reclaim_process_memory()
        reclaimed = True
        token, allowance = _memory_pool.acquire(requested)
    if (
        requested is None
        and allowance.limit_bytes < DEFAULT_BUFFER_CAP
        and not reclaimed
    ):
        # Large projections can leave dead Python objects and allocator pages
        # behind. Reclaim before freezing a needlessly small SQL budget, not
        # only when even the minimum reservation would fail. Admission and the
        # original shared/default caps still apply to the fresh measurement.
        _memory_pool.release(token)
        _reclaim_process_memory()
        token, allowance = _memory_pool.acquire(requested)
    scratch = None
    connection = None
    try:
        scratch = tempfile.TemporaryDirectory(prefix="pagemantle-duckdb-", dir=parent)
        settings["memory_limit"] = f"{allowance.limit_bytes}B"
        settings.setdefault("threads", 2)
        settings.setdefault("preserve_insertion_order", True)
        settings["temp_directory"] = str(Path(scratch.name) / "spill")
        if str(database) == ":memory:":
            database = str(Path(scratch.name) / "work.duckdb")
        connection = _duckdb.connect(
            database=database, read_only=read_only, config=settings, **kwargs
        )
        connection.execute("SET enable_progress_bar=false")
        return Connection(connection, scratch, token, allowance)
    except BaseException:
        try:
            if connection is not None:
                connection.close()
            if scratch is not None:
                scratch.cleanup()
        finally:
            _memory_pool.release(token)
        raise


def query_rows(query, parameters=None):
    """Small result sets only; callers must bound large results in SQL."""
    with connect() as connection:
        return connection.execute(query, parameters or []).fetchall()


def copy_query(connection, query, parameters, destination):
    """Stream a computed table to disk without a second materialized copy."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = path.resolve().as_posix().replace("'", "''")
    try:
        connection.execute(
            f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            parameters,
        )
    except Exception as error:
        raise RuntimeError(f"Parquet write failed for {path}: {error}") from error


def insert_json_rows(connection, table_name, columns, records):
    """Bulk-load normalized rows through bounded local scratch, with fixed types."""

    def encode(value):
        if isinstance(value, (date, datetime, time)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError("unsupported native JSON value: " + type(value).__name__)

    with tempfile.TemporaryDirectory(
        dir=os.environ.get("BENCHMARK_SCRATCH_DIR")
    ) as scratch:
        source = Path(scratch) / "records.jsonl"
        count = 0
        with source.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, separators=(",", ":"), default=encode))
                stream.write("\n")
                count += 1
        if not count:
            return
        relation = connection.read_json(
            str(source), columns=dict(columns), format="newline_delimited"
        )
        relation.create_view("_native_bulk_input", replace=True)
        quoted = '"' + table_name.replace('"', '""') + '"'
        try:
            connection.execute(f"INSERT INTO {quoted} SELECT * FROM _native_bulk_input")
        finally:
            connection.execute("DROP VIEW _native_bulk_input")
