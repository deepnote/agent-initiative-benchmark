"""A single sequential Slack owner process, stopped before final-state capture."""

from __future__ import annotations

import multiprocessing
import asyncio
import json
import os
import signal
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from .baseline_access import (
    BaselineAccessError,
    BaselineCache,
    CacheLimits,
    open_baseline,
    _unlinked,
)
from .slack_environment import SlackEnvironment, _encoded
from .failure_causes import exception_cause, safe_text, RuntimeInputError
from .contracts import load_config, load_task
from .packages import verify_package, hash_file
from . import service_registry as services


def _serve(connection, factory, arguments, ipc, transport_factory):
    def respond(value):
        temporary = ipc / "result.tmp"
        temporary.write_bytes(_encoded(value))
        os.replace(temporary, ipc / "result.json")
        # Only a small notification goes through the pipe. Receiving a partial
        # large pickle must not defeat a tool deadline.
        connection.send(True)

    environment = None
    lifetime = ExitStack()
    preparing = True
    settings = None
    try:
        arguments = dict(arguments)
        operator_preparation = arguments.pop("operator_preparation", False)
        read_only = arguments.pop("read_only", False)
        settings = arguments.pop("baseline_settings", None)
        # This runs in the spawned native child. Host index paths must not make
        # independently leased workers share writable Notion index files.
        index_root = (
            Path(settings["cache_dir"])
            if settings is not None
            else Path(arguments["work_dir"])
        )
        os.environ["BENCHMARK_CACHE_DIR"] = str(index_root)
        os.environ["BENCHMARK_NOTION_RUNTIME_DIR"] = str(index_root / "notion-indexes")
        os.environ["NOTION_EMBEDDING_THREADS"] = os.environ.get("OMP_NUM_THREADS", "2")
        judge_receipt = arguments.pop("judge_baseline_receipt", None)
        judge_limits = arguments.pop("judge_baseline_limits", None)
        local_known = arguments.pop("judge_local_files", {})
        local_targets = {}
        accounted = set()
        selected = {}
        checked_selection = None
        selected_root = None
        linear_baseline_dir = arguments.pop("linear_baseline_dir", None)
        notion_baseline_dir = arguments.pop("notion_baseline_dir", None)
        service_baseline_dirs = arguments.pop("service_baseline_dirs", None) or {}
        stripe_baseline_dir = arguments.pop("stripe_baseline_dir", None)
        config = load_config(arguments["config_path"])
        task = load_task(Path(arguments["package"]) / "task.json")
        manifest = verify_package(
            arguments["package"], config_path=arguments["config_path"]
        )
        baseline = None
        readers = {}

        def save_receipt(local):
            nonlocal selected_root
            selected_root = local.root
            selected.update({r["path"]: r for r in local.receipt["files"]})
            receipt_path = ipc / "baseline-receipt.json"
            receipt = {
                **local.receipt,
                "files": [
                    {**record, "path": safe_text(record["path"])}
                    for record in local.receipt["files"]
                ],
            }
            if receipt_path.exists():
                previous = json.loads(receipt_path.read_bytes())
                prior_paths = {record["path"] for record in previous["files"]}
                receipt["files"] = previous["files"] + [
                    record
                    for record in receipt["files"]
                    if record["path"] not in prior_paths
                ]
            temporary = ipc / "baseline-receipt.tmp"
            temporary.write_bytes(_encoded(receipt))
            os.replace(temporary, receipt_path)

        def verify_selected():
            if baseline is not None and getattr(
                baseline._dataset.cache, "shared_local", False
            ):
                for path, record in selected.items():
                    baseline._dataset.cache.check(
                        selected_root / path, record["sha256"], record["size_bytes"]
                    )
                return
            if judge_limits is None:
                return
            for path, record in selected.items():
                target = (
                    local_targets[path]
                    if path in local_targets
                    else selected_root / path
                )
                try:
                    info = target.lstat()
                except FileNotFoundError:
                    raise BaselineAccessError(
                        "missing_required_file",
                        "Leased baseline disappeared",
                        path=path,
                    ) from None
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or info.st_nlink != 1
                    or hash_file(target) != (record["size_bytes"], record["sha256"])
                ):
                    raise BaselineAccessError(
                        "checksum_mismatch",
                        "Selected baseline changed while leased",
                        path=path,
                    )

        def check_selection(paths):
            nonlocal checked_selection
            subset = set(paths) <= selected.keys()
            # Pinned inventory and cache policy cannot change in a worker.
            # Recheck when a selection grows or the judge's allowance changes.
            signature = (
                len(selected),
                (
                    tuple(sorted(judge_limits.items()))
                    if judge_limits is not None
                    else None
                ),
            )
            if subset and checked_selection == signature:
                return
            # Include earlier leases before any later linked-file transfer.
            paths = set(selected) | set(paths)
            baseline.check_limits(paths)
            if judge_limits is None:
                checked_selection = signature if subset else None
                return
            records = baseline.file_inventory(paths)
            for record in records:
                if (
                    max(record["size_bytes"], record.get("storage_size_bytes", 0))
                    > judge_limits["file_bytes"]
                ):
                    raise BaselineAccessError(
                        "source_file_limit",
                        "Selected baseline exceeds judge file bound",
                    )
            stored = {
                r["object_key"]: r["storage_size_bytes"]
                for r in records
                if r.get("storage_encoding")
            }
            needed = (
                sum(r["size_bytes"] for r in records)
                + sum(stored.values())
                + baseline.metadata_size_bytes
                + len(_encoded(records)) * 2
            )
            if needed > judge_limits["bytes"]:
                raise BaselineAccessError(
                    "source_limit",
                    "Selected files and metadata exceed cumulative judge source bound",
                )
            if len(records) + 2 > judge_limits["count"]:
                raise BaselineAccessError(
                    "source_count_limit",
                    "Selected files exceed cumulative judge source count",
                )
            checked_selection = signature if subset else None

        def acquire(service, paths=None, *, inventory=False, max_file_bytes=None):
            if inventory:
                if baseline is not None:
                    return tuple(
                        p[len(service.NAME) + 1 :]
                        for p in baseline.plan_service(service.NAME).paths
                    )
                root = Path(local_roots[service.NAME]).resolve()
                return tuple(
                    p.relative_to(root).as_posix()
                    for p in root.rglob("*")
                    if p.is_file()
                    and service.local_file(p.relative_to(root).as_posix())
                )
            if baseline is None:
                lifetime.callback(readers[service.NAME].close)
                root = Path(local_roots[service.NAME]).resolve()
                if judge_limits is not None:
                    prefix = service.NAME + "/"
                    chosen = paths or tuple(
                        p
                        for p in local_known
                        if p.startswith(prefix) and service.local_file(p[len(prefix) :])
                    )
                    records = dict(selected)
                    for path in chosen:
                        if not path.startswith(
                            service.NAME + "/"
                        ) or not service.local_file(path[len(service.NAME) + 1 :]):
                            raise BaselineAccessError(
                                "unsafe_path", "Invalid local linked selection"
                            )
                        target = root / path[len(service.NAME) + 1 :]
                        if any(p.is_symlink() for p in (target, *target.parents)):
                            raise BaselineAccessError(
                                "unsafe_path", "Linked source contains a symlink"
                            )
                        size = target.stat().st_size
                        if size > judge_limits["file_bytes"]:
                            raise BaselineAccessError(
                                "source_file_limit",
                                "Linked source exceeds judge file bound",
                            )
                        record = dict(path=path, size_bytes=size, sha256="0" * 64)
                        records[path] = record
                        local_targets[path] = target
                    added = [
                        r
                        for p, r in records.items()
                        if p not in accounted and p not in local_known
                    ]
                    if (
                        sum(r["size_bytes"] for r in added)
                        + len(_encoded(list(records.values()))) * 2
                        > judge_limits["bytes"]
                    ):
                        raise BaselineAccessError(
                            "source_limit",
                            "Linked files exceed cumulative judge source bound",
                        )
                    if len(added) > judge_limits["count"]:
                        raise BaselineAccessError(
                            "source_count_limit",
                            "Linked files exceed judge source count",
                        )
                    for path in chosen:
                        record = records[path]
                        record["sha256"] = hash_file(local_targets[path])[1]
                        if path in local_known and local_known[path] != record:
                            raise BaselineAccessError(
                                "checksum_mismatch", "Local catalog changed"
                            )
                    from types import SimpleNamespace

                    save_receipt(
                        SimpleNamespace(
                            root=root.parent,
                            receipt=dict(
                                build_ref=config["build_ref"],
                                scope="selected_files",
                                files=list(records.values()),
                            ),
                        )
                    )
                return root
            paths = service.required_files(baseline) if paths is None else paths
            # Adapter bounds describe decoded documents. Shared pack/gzip object
            # bytes remain bounded by the cache, transfer and judge policies.
            if max_file_bytes is not None and any(
                r["size_bytes"] > max_file_bytes for r in baseline.file_inventory(paths)
            ):
                raise BaselineAccessError(
                    "source_file_limit",
                    "Linked file exceeds adapter file bound",
                    service=service.NAME,
                )
            check_selection(paths)
            if set(paths) <= selected.keys():
                # The worker still owns these leases and their cache lock.
                # Reverify selected bytes, without rescanning the entire cache
                # or rewriting an identical cumulative receipt for every PDF.
                for path in paths:
                    target = selected_root / path
                    if getattr(baseline._dataset.cache, "shared_local", False):
                        record = selected[path]
                        baseline._dataset.cache.check(
                            target, record["sha256"], record["size_bytes"]
                        )
                        continue
                    _unlinked(target)
                    record = selected[path]
                    if (
                        not target.is_file()
                        or target.stat().st_nlink != 1
                        or hash_file(target) != (record["size_bytes"], record["sha256"])
                    ):
                        raise BaselineAccessError(
                            "checksum_mismatch", "Leased baseline changed", path=path
                        )
                return selected_root / service.NAME
            local = lifetime.enter_context(baseline.acquire_files(paths))
            save_receipt(local)
            # Reader closes before its lease. Both run on this worker thread.
            lifetime.callback(readers[service.NAME].close)
            return local.root / service.NAME

        local_roots = {
            "linear": linear_baseline_dir,
            "notion": notion_baseline_dir,
            "stripe": stripe_baseline_dir,
            **service_baseline_dirs,
        }
        for service in services.SERVICES:
            if service.NAME != services.CONTEXT_SERVICE:

                def service_acquire(paths=None, service=service, **bounds):
                    return acquire(service, paths, **bounds)

                service_acquire.inventory = lambda service=service: (
                    baseline.plan_service(service.NAME).paths
                    if baseline is not None
                    else tuple(
                        service.NAME
                        + "/"
                        + p.relative_to(local_roots[service.NAME]).as_posix()
                        for p in Path(local_roots[service.NAME]).rglob("*")
                        if p.is_file()
                    )
                )
                service_acquire.file_inventory = lambda paths: (
                    baseline.file_inventory(paths) if baseline is not None else None
                )
                readers[service.NAME] = service.create(
                    arguments, manifest, task, config, service_acquire
                )

        if settings is not None and "hubspot" in readers:
            readers["hubspot"].configure_document_cache(
                Path(settings["cache_dir"]) / "hubspot-text", config["build_ref"]
            )
        if settings is not None:
            cache = BaselineCache(
                Path(settings["cache_dir"]), CacheLimits(**settings["limits"])
            )
            lifetime.callback(cache.close)
            baseline = open_baseline(
                arguments["baseline_build_ref"],
                settings["root"],
                cache,
                manifest_sha256=settings["manifest_sha256"],
            )
            baseline.check_limits(
                services.required_files(baseline, config["tool_scope"])
            )
            if judge_limits is not None:
                baseline.verify_receipt(judge_receipt)
                planned = services.required_files(baseline, config["tool_scope"])
                records = baseline.file_inventory(planned)
                for record in records:
                    if (
                        max(record["size_bytes"], record.get("storage_size_bytes", 0))
                        > judge_limits["file_bytes"]
                    ):
                        raise BaselineAccessError(
                            "source_file_limit",
                            "Selected baseline exceeds judge file bound",
                        )
                stored = {
                    r["object_key"]: r["storage_size_bytes"]
                    for r in records
                    if r.get("storage_encoding")
                }
                needed = (
                    sum(r["size_bytes"] for r in records)
                    + sum(stored.values())
                    + baseline.metadata_size_bytes
                    + len(_encoded(records)) * 2
                )
                if needed > judge_limits["bytes"]:
                    raise BaselineAccessError(
                        "source_limit",
                        "Selected baseline and metadata exceed judge source bound",
                    )
                if (
                    len(records) + max(2, baseline.metadata_file_count)
                    > judge_limits["count"]
                ):
                    raise BaselineAccessError(
                        "source_count_limit",
                        "Selected baseline exceeds judge source count",
                    )
            local = lifetime.enter_context(baseline.acquire_service("slack"))
            arguments["baseline_dir"] = local.root / "slack"
            save_receipt(local)
        environment = factory(**arguments)
        readers[services.CONTEXT_SERVICE] = environment
        inspector = None
        respond(
            {
                "ok": True,
                "context": environment.initial_context(),
                "tools": services.reader_definitions(readers, read_only=read_only),
                **(
                    {
                        "baseline_metadata_bytes": (
                            baseline.metadata_size_bytes if baseline is not None else 0
                        )
                    }
                    if judge_limits is not None
                    else {}
                ),
            }
        )
        preparing = False
        while True:
            command = connection.recv()
            if command is None:
                environment.close()
                environment = None
                lifetime.close()
                respond({"ok": True})
                break
            command_data = json.loads((ipc / "command.json").read_bytes())
            if baseline is not None and getattr(
                baseline._dataset.cache, "shared_local", False
            ):
                verify_selected()
            name, arguments = command_data[:2]
            if len(command_data) == 3 and judge_limits is not None:
                # Parent reserves source metadata/results too. Refresh the amount
                # still available before a read can acquire another linked file.
                remaining = command_data[2]
                leased = sum(r["size_bytes"] for r in selected.values())
                judge_limits["bytes"] = remaining["bytes"] + (
                    leased + baseline.metadata_size_bytes if baseline is not None else 0
                )
                judge_limits["count"] = remaining["count"] + (
                    len(selected) + 2 if baseline is not None else 0
                )
            accounted = set(selected)
            if name == "_operator_prepare_contract_search" and operator_preparation:
                result = readers["hubspot"].prepare_contract_search(**arguments)
            elif name == "_judge_verify_baseline" and judge_limits is not None:
                verify_selected()
                result = {"ok": True}
            elif name == "_judge_read_fragment" and read_only and inspector is None:
                result = environment.call(name, arguments)
            else:
                result = services.dispatch(
                    readers, name, arguments, read_only=read_only
                )
            respond({"ok": True, "result": result})
    except BaseException as error:
        # Only trusted validation text and selected fields cross this boundary.
        trusted = services.trusted_error(error) or isinstance(
            error, (BaselineAccessError, RuntimeInputError)
        )
        cause = exception_cause(error, trusted_reason=trusted)
        if isinstance(error, BaselineAccessError):
            cause["details"] = {
                key: safe_text(value) if isinstance(value, str) else value
                for key, value in error.details.items()
                if key
                in {
                    "build",
                    "service",
                    "path",
                    "key",
                    "expected",
                    "actual",
                    "needed",
                    "limit",
                    "available",
                    "field",
                }
                and (value is None or type(value) in (str, int, float))
            }
        code = (
            "initial_context_limit"
            if isinstance(error, ValueError) and "initial_context" in str(error)
            else "environment_failure"
        )
        if preparing and settings is not None:
            code = "baseline_preparation_failure"
        try:
            respond(
                {
                    "ok": False,
                    "code": code,
                    "error_type": cause["type"],
                    "cause": cause,
                    "reason": cause["reason"],
                }
            )
        except (OSError, EOFError):
            pass
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            lifetime.close()
            connection.close()


class WorkerFailure(RuntimeError):
    def __init__(self, code, reason, *, cause=None):
        self.code, self.cause = code, cause
        super().__init__(reason)


class SlackWorker:
    def __init__(
        self, arguments: dict, factory=SlackEnvironment, transport_factory=None
    ):
        self._close_lock = threading.RLock()
        self._terminal_failure = None
        self.ipc = Path(arguments["work_dir"]).parent / "worker-ipc"
        self.ipc.mkdir()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_serve,
            args=(child, factory, arguments, self.ipc, transport_factory),
            daemon=True,
        )
        self.process.start()
        child.close()

    def receive(self, timeout: float) -> dict:
        self._check_terminal()
        if not self.connection.poll(timeout):
            if not self.process.is_alive():
                raise self._crash_failure()
            error = self._remember_failure(
                WorkerFailure(
                    "worker_timeout",
                    "Isolated Slack worker exceeded its deadline; execution is cancelled",
                )
            )
            self.close(graceful=False)
            raise error
        return self._read_result()

    def _remember_failure(self, error):
        if getattr(self, "_terminal_failure", None) is None:
            self._terminal_failure = error
        return self._terminal_failure

    def _check_terminal(self):
        if getattr(self, "_terminal_failure", None) is not None:
            raise self._terminal_failure

    def _read_result(self):
        try:
            self.connection.recv()
            value = json.loads((self.ipc / "result.json").read_bytes())
            (self.ipc / "result.json").unlink()
        except (EOFError, OSError):
            raise self._crash_failure() from None
        if not value["ok"]:
            raise self._remember_failure(
                WorkerFailure(
                    value["code"],
                    f"Slack environment failed: {value['error_type']} ({value['cause']['code']}): {value['reason']}",
                    cause=value["cause"],
                )
            )
        return value

    def _crash_failure(self):
        if getattr(self, "_terminal_failure", None) is not None:
            return self._terminal_failure
        self.process.join(0.1)
        code = self.process.exitcode
        details = {"exit_code": code}
        reason = "Isolated Slack worker stopped without a result"
        if code is not None and code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = "UNKNOWN"
            details["signal"] = name
            reason += f" (signal {name}, exit code {code})"
        elif code is not None:
            reason += f" (exit code {code})"
        return self._remember_failure(
            WorkerFailure(
                "worker_crash",
                reason,
                cause={
                    "type": "WorkerProcess",
                    "code": "worker_crash",
                    "reason": reason,
                    "details": details,
                },
            )
        )

    def _send_command(self):
        try:
            self.connection.send(True)
        except (OSError, EOFError):
            raise self._crash_failure() from None

    async def receive_async(self, timeout: float) -> dict:
        self._check_terminal()
        # Cancellation never leaves a thread blocked on IPC or downloading.
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                error = self._remember_failure(
                    WorkerFailure(
                        "worker_timeout",
                        "Isolated Slack worker exceeded its deadline; execution is cancelled",
                    )
                )
                self.close(graceful=False)
                raise error
            if self.connection.poll(0):
                return self._read_result()
            if not self.process.is_alive():
                raise self._crash_failure()
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))

    async def call_async(self, name: str, arguments: dict, timeout: float):
        self._check_terminal()
        (self.ipc / "command.json").write_bytes(_encoded([name, arguments]))
        self._send_command()
        return (await self.receive_async(timeout))["result"]

    def call(self, name: str, arguments: dict, timeout: float):
        self._check_terminal()
        command = [name, arguments]
        if getattr(self, "source_remaining", None) is not None:
            command.append(self.source_remaining)
        (self.ipc / "command.json").write_bytes(_encoded(command))
        self._send_command()
        return self.receive(timeout)["result"]

    def close(self, *, graceful=True):
        # receive() may time out on a reader thread while its caller cancels.
        # Serialize shutdown so the same pipe handle is never closed twice.
        with self._close_lock:
            self._close(graceful=graceful)

    def _close(self, *, graceful):
        if graceful and self.process.is_alive():
            try:
                self.connection.send(None)
                self.receive(0.25)
            except (OSError, EOFError, WorkerFailure):
                pass
            self.process.join(0.25)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1)
        self.process.join(0)
        self.connection.close()
        if self.process.is_alive():
            raise WorkerFailure(
                "shutdown_failure",
                "Could not stop Slack worker; final state cannot be frozen",
            )
