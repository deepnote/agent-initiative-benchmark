"""One offline Slack-plus-Linear judgment, with immutable evidence and end-only publication."""

from __future__ import annotations

from benchmark.prompts import load_text, prompt_path

import asyncio
from contextlib import closing, contextmanager
import hashlib
import math
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import stat
import time

from .agent_runner import (
    Trace,
    AttemptError,
    PublicationError,
    digest,
    WRITES,
    FINISH,
    resolve_limits as agent_limits,
    baseline_settings,
    baseline_identity,
)
from .artifact_store import _relative, _safe_reason, validate_completion_manifest
from .attempt_worker import SlackWorker, WorkerFailure
from .contracts import ContractError, load_config, load_task
from .failure_causes import (
    RuntimeInputError,
    exception_cause,
    provider_body,
    provider_cause,
    safe_text,
    safe_code,
)
from .judge_contracts import (
    JudgeError,
    Judgment,
    READ_SOURCE,
    SUBMIT,
    RUBRIC_VERSION,
    CRITERIA,
    PROMPT_VERSION,
    ERROR_POLICY_VERSION,
    CITATION_FEEDBACK_VERSION,
    EVALUATION_VERSION,
    SCHEMA_VERSION,
    DEFAULT_LIMITS,
    fields,
    parse_json,
    resolve_limits,
    validate_model_judgment,
    evidence_assembly,
    error_result,
    text,
    memory_bytes,
    bound_budget_override,
    score_consistency_feedback,
    source_read_requests,
    judge_read_group_decline,
)
from .packages import hash_file, verify_package
from .responses_adapter import load_model, load_judge_model, model_response
from .slack_environment import SlackEnvironment, _encoded
from . import service_registry as services
from . import read_result_admission as read_admission
from . import judge_read_admission
from .agent_protocol import recorded_protocols, runtime_status, preflight
from .judge_anonymity import (
    VERSION as ANONYMOUS_VERSION,
    Projection,
    freeze,
    require_prepared,
    private_restore_patch,
)

PROMPT = prompt_path("judge/evaluate_agent_run.txt")
REMINDER = load_text("judge/submit_reminder.txt")
NATIVE_GUARDS = {
    "max_delta_bytes": 8388608,
    "max_page_size": 100,
    "max_related_rows": 10000,
}


def check(condition, code, reason):
    if not condition:
        raise JudgeError(code, reason)


def size_check(size, limit, code):
    check(
        size <= limit,
        code,
        f"{code}: complete bytes exceed the pinned limit; nothing was omitted",
    )


def inventory(root, limits, *, exclude_marker=False):
    root = Path(root)
    check(
        not root.is_symlink() and root.is_dir(),
        "unsafe_source",
        "Source must be an unlinked directory",
    )
    files = []

    def visit(directory):
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if exclude_marker and relative == "_complete.json":
                check(
                    stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                    "unsafe_source",
                    "Completion marker must be unlinked",
                )
                size_check(
                    info.st_size,
                    limits["judge_max_source_file_bytes"],
                    "source_file_limit",
                )
                continue
            _relative(relative)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            else:
                check(
                    stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                    "unsafe_source",
                    "Source links/special files are forbidden",
                )
                size_check(
                    info.st_size,
                    limits["judge_max_source_file_bytes"],
                    "source_file_limit",
                )
                size, sha = hash_file(path)
                files.append({"path": relative, "bytes": size, "sha256": sha})
                size_check(
                    len(files), limits["judge_max_source_count"], "source_count_limit"
                )
                size_check(
                    sum(f["bytes"] for f in files),
                    limits["judge_max_source_bytes"],
                    "source_limit",
                )

    visit(root)
    return sorted(files, key=lambda f: f["path"])


def catalog_inventory(root, limits):
    """Notion bodies are selected and bounded by the worker on first read."""
    target = Path(root) / "catalog.sqlite"
    size_check(
        target.stat().st_size,
        limits["judge_max_source_file_bytes"],
        "source_file_limit",
    )
    size, sha = hash_file(target)
    return [dict(path="catalog.sqlite", bytes=size, sha256=sha)]


def disk_bytes(root):
    size = 0
    for path in Path(root).rglob("*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue  # DuckDB may remove a completed spill while we inspect it.
        if stat.S_ISREG(info.st_mode):
            size += info.st_size
    return size


class WorkBudget:
    def __init__(self, root, limit):
        self.root, self.limit = Path(root), limit

    def reserve(self, upcoming):
        size_check(disk_bytes(self.root) + upcoming, self.limit, "work_limit")
        check(
            shutil.disk_usage(self.root).free >= upcoming,
            "disk_full",
            "Not enough free disk for the next bounded operation",
        )

    def write(self, path, raw):
        self.reserve(len(raw))
        Path(path).write_bytes(raw)


async def worker_operation(worker, operation, budget, timeout_seconds):
    """Watch native scratch/IPC growth while the cancellable worker is busy."""
    pending = asyncio.create_task(asyncio.to_thread(operation))
    started = time.monotonic()
    try:
        while not pending.done():
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise WorkerFailure(
                    "worker_timeout",
                    "Judge read/setup operation exceeded its deadline; worker stopped",
                )
            await asyncio.wait({pending}, timeout=min(0.01, remaining))
            budget.reserve(Trace.RESERVE + 65536)
        if time.monotonic() - started >= timeout_seconds:
            raise WorkerFailure(
                "worker_timeout",
                "Judge read/setup operation exceeded its deadline; result was not delivered",
            )
        return await pending
    except BaseException:
        worker.close(graceful=False)
        try:
            await pending
        except BaseException:
            pass
        if getattr(worker, "keep_baseline_receipt", False):
            receipt = worker.ipc / "baseline-receipt.json"
            if receipt.exists():
                # Keep the last atomic acquisition receipt through scratch/IPC
                # cleanup. The caller validates its bounds before retaining it.
                receipt.replace(budget.root / "baseline-receipt.pending")
        # Only the freshly created worker paths are disposable. Sources and
        # observable reads already in the trace are outside these directories.
        for name in ("environment", "worker-ipc"):
            shutil.rmtree(budget.root / name, ignore_errors=True)
        raise


class JudgeTrace(Trace):
    """Existing Trace format, with disk reservations before every append."""

    def __init__(self, path, limit, started, work):
        self.work = work
        super().__init__(path, limit, started)

    def append(self, kind, payload, **kwargs):
        self.work.reserve(len(_encoded(payload)) + 2048)
        return super().append(kind, payload, **kwargs)

    def append_native(self, payload):
        self.work.reserve(len(_encoded(payload)) + 2048)
        return super().append_native(payload)


class ReadOnlySlackEnvironment:
    """Importable composition wrapper; original config and agent stay unchanged."""

    def __init__(
        self,
        *,
        judge_work_root,
        judge_max_work_bytes,
        judge_trace_sources=None,
        **arguments,
    ):
        self.trace_sources = judge_trace_sources or {}
        self.budget = WorkBudget(judge_work_root, judge_max_work_bytes)
        # Reserve bounded IPC, the native delta compiler and DuckDB scratch.
        # A conservative reservation can reject a tiny work budget early.
        self.scratch = memory_bytes(arguments["duckdb_max_temp_directory_size"])
        self.result_limit = arguments["max_result_bytes"]
        self.budget.reserve(
            self.scratch + NATIVE_GUARDS["max_delta_bytes"] * 4 + self.result_limit * 2
        )
        self.environment = SlackEnvironment(**arguments)
        self.budget.reserve(self.result_limit * 2)

    @property
    def tool_definitions(self):
        return [
            t
            for t in self.environment.tool_definitions
            if t["name"] in services.read_names()
        ]

    def initial_context(self):
        result = self.environment.initial_context()
        self.budget.reserve(
            len(_encoded(result)) + len(_encoded(self.tool_definitions)) + 1024
        )
        return result

    def call(self, name, arguments):
        if name == "_judge_read_fragment":
            sid = arguments["source_id"]
            check(
                sid in self.trace_sources,
                "unknown_source",
                "Internal source is not registered",
            )
            source = self.trace_sources[sid]
            path, offset, length, sha = source
            with Path(path).open("rb") as stream:
                stream.seek(offset)
                body = parse_json(stream.read(length))
            raw = _encoded(body)
            check(
                hashlib.sha256(raw).hexdigest() == sha,
                "changed_input",
                "Indexed source changed during judgment",
            )
            content = raw.decode("utf-8")
            start = arguments["offset"]
            end = min(len(content), start + arguments["max_chars"])
            result = {"text": content[start:end], "end": end, "total": len(content)}
            size_check(len(_encoded(result)) + 512, self.result_limit, "result_limit")
            self.budget.reserve(len(_encoded(result)) + 512)
            return result
        if name not in services.read_names():
            return {
                "ok": False,
                "error": "forbidden_write",
                "message": "Only configured reads are allowed",
            }
        self.budget.reserve(self.result_limit * 2 + self.scratch)
        result = self.environment.call(name, arguments)
        self.budget.reserve(len(_encoded(result)) + 1024)
        return result

    def close(self):
        self.environment.close()


class Sources:
    """Harness-owned IDs, exact source bodies and lazy trace file offsets."""

    def __init__(self, limits, total_bytes=0, file_count=0, *, projection=None):
        self.limits = limits
        self.current_protocol = True
        self.entries, self.bodies, self.accessible = {}, {}, set()
        self.total_bytes, self.file_count = total_bytes, file_count
        self.cursors, self.progress = {}, {}
        self.projection, self.originals = projection, {}

    @contextmanager
    def delivery(self):
        """Stage visibility without copying source bodies. Denied reads keep
        their cumulative source byte/count charge but expose no new handles.
        """
        before = (
            self.entries.copy(),
            self.bodies.copy(),
            self.originals.copy(),
            self.accessible.copy(),
            self.cursors.copy(),
            self.progress.copy(),
            self.total_bytes,
            self.file_count,
            self.projection.aliases.copy() if self.projection is not None else None,
        )
        committed = []
        try:
            yield lambda: committed.append(True)
        finally:
            if not committed:
                new_count = max(0, len(self.entries) - len(before[0]))
                charged = max(self.total_bytes, before[6])
                (
                    self.entries,
                    self.bodies,
                    self.originals,
                    self.accessible,
                    self.cursors,
                    self.progress,
                    _,
                    self.file_count,
                    aliases,
                ) = before
                self.total_bytes = charged
                self.file_count += new_count
                if self.projection is not None:
                    self.projection.aliases = aliases

    def add(
        self,
        sid,
        kind,
        locator,
        body=None,
        *,
        raw=None,
        trace_file=None,
        offset=None,
        length=None,
        seen=False,
    ):
        original = body
        if self.projection is not None:
            if raw is not None:
                original = (
                    raw.decode("utf-8")
                    if sid == "artifact:instructions.txt"
                    else parse_json(raw)
                )
            original_raw = raw if raw is not None else _encoded(body)
            self.originals[sid] = {
                "id": sid,
                "kind": kind,
                "locator": locator,
                "sha256": hashlib.sha256(original_raw).hexdigest(),
            }
            if trace_file is not None:
                self.originals[sid]["target"] = {
                    "path": str(trace_file),
                    "offset": offset,
                    "length": length,
                }
            locator, body = self.projection.source(sid, kind, locator, original)
            raw = (
                body.encode("utf-8")
                if sid == "artifact:instructions.txt"
                else _encoded(body)
            )
        if trace_file is not None:
            # Original event bodies hash canonical JSON, not the JSONL newline.
            sha = (
                hashlib.sha256(raw).hexdigest()
                if self.projection is not None
                else digest(body)
            )
            target = (Path(trace_file), offset, length)
            # Event bytes are already inventoried; index metadata is additional.
            added = len(_encoded([str(trace_file), offset, length]))
        else:
            raw = raw if raw is not None else _encoded(body)
            sha, target, added = hashlib.sha256(raw).hexdigest(), raw, len(raw)
        entry = {"id": sid, "kind": kind, "locator": locator, "sha256": sha}
        if sid in self.entries:
            check(
                self.entries[sid] == entry,
                "source_conflict",
                "Conflicting content under one evidence ID",
            )
        else:
            size_check(
                len(self.entries) + self.file_count + 1,
                self.limits["judge_max_source_count"],
                "source_count_limit",
            )
            size_check(
                self.total_bytes + added + len(_encoded(entry)),
                self.limits["judge_max_source_bytes"],
                "source_limit",
            )
            self.total_bytes += added + len(_encoded(entry))
            self.entries[sid], self.bodies[sid] = entry, target
        if seen:
            self.accessible.add(sid)
        return sid

    def source_bytes(self, sid):
        target = self.bodies[sid]
        if isinstance(target, bytes):
            return target
        path, offset, length = target
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        body = parse_json(raw)
        original_sha = (
            self.originals[sid]["sha256"]
            if self.projection is not None
            else self.entries[sid]["sha256"]
        )
        check(
            digest(body) == original_sha,
            "changed_input",
            "Original trace source changed during judgment",
        )
        if self.projection is not None:
            original = self.originals[sid]
            _, body = self.projection.source(
                sid, original["kind"], original["locator"], body
            )
        raw = _encoded(body)
        check(
            hashlib.sha256(raw).hexdigest() == self.entries[sid]["sha256"],
            "changed_input",
            "Anonymous source changed during judgment",
        )
        return raw

    def partial_citation_feedback(
        self, value, *, delivered, rubric_version=RUBRIC_VERSION
    ):
        """Explain incomplete reads without reading or granting any source."""
        try:
            lists = [
                value["evidence"],
                *(value["criteria"][name]["evidence"] for name in CRITERIA),
                value["reference_difference"]["evidence"],
            ]
        except (KeyError, TypeError):
            return None
        if any(
            not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids)
            for ids in lists
        ):
            return None
        missing = list(
            dict.fromkeys(
                sid for ids in lists for sid in ids if sid not in self.accessible
            )
        )
        if not missing or any(
            sid not in self.entries or self.progress.get(sid, 0) <= 0 for sid in missing
        ):
            return None
        # A copy is used only to prove that incomplete reads are the sole defect.
        # The real accessible set, cursors and read progress stay unchanged.
        try:
            validate_model_judgment(
                value,
                delivered=delivered,
                sources=self.entries,
                accessible=self.accessible | set(missing),
                assemble_evidence=True,
                rubric_version=rubric_version,
            )
        except JudgeError:
            return None
        return {
            "ok": False,
            "error": "invalid_citation",
            "protocol": "partial-citation-feedback-1",
            "source_ids": missing,
            "message": "Finish reading these sources with judge_read_source. Follow their next_cursor values until next_cursor is null, then submit the judgment again.",
        }

    def submission_citation_feedback(
        self,
        value,
        *,
        delivered,
        rubric_version=RUBRIC_VERSION,
        protocol=CITATION_FEEDBACK_VERSION,
    ):
        """Describe rejected IDs without changing a submission or granting reads."""
        if protocol == "partial-citation-feedback-1":
            return self.partial_citation_feedback(
                value, delivered=delivered, rubric_version=rubric_version
            )
        check(
            protocol == "citation-feedback-2",
            "invalid_judgment",
            "Unknown citation correction protocol",
        )
        try:
            lists = [
                value["evidence"],
                *(value["criteria"][name]["evidence"] for name in CRITERIA),
                value["reference_difference"]["evidence"],
            ]
        except (KeyError, TypeError):
            return None
        if any(
            not isinstance(ids, list) or any(not isinstance(sid, str) for sid in ids)
            for ids in lists
        ):
            return None
        missing = list(
            dict.fromkeys(
                sid
                for ids in lists
                for sid in ids
                if sid not in self.entries or sid not in self.accessible
            )
        )
        if not missing:
            return None
        # Temporary validation proves citation availability is the only defect.
        # These placeholder entries never enter the real registry or read state.
        possible = {
            **self.entries,
            **{sid: {"kind": "trace"} for sid in missing if sid not in self.entries},
        }
        try:
            validate_model_judgment(
                value,
                delivered=delivered,
                sources=possible,
                accessible=self.accessible | set(missing),
                assemble_evidence=True,
                rubric_version=rubric_version,
            )
        except JudgeError:
            return None
        issues = []
        for sid in missing:
            if sid not in self.entries:
                issues.append({"source_id": sid, "state": "unknown"})
            elif self.progress.get(sid, 0) <= 0:
                issues.append({"source_id": sid, "state": "unread", "cursor": None})
            else:
                expected = (sid, self.entries[sid]["sha256"], self.progress[sid])
                cursor = next(
                    (
                        key
                        for key in reversed(self.cursors)
                        if self.cursors[key] == expected
                    ),
                    None,
                )
                if cursor is None:
                    return None
                issues.append({"source_id": sid, "state": "partial", "cursor": cursor})
        return {
            "ok": False,
            "error": "invalid_citation",
            "protocol": "citation-feedback-2",
            "issues": issues,
            "message": load_text("judge/citation_feedback.txt"),
        }

    def read(self, args, fragment_reader=None):
        reads = source_read_requests(args)
        if reads is None:
            return {
                "ok": False,
                "error": "invalid_arguments",
                "message": "Invalid source read arguments",
            }
        if isinstance(args["source_id"], list):
            if any(read["source_id"] not in self.entries for read in reads):
                return {
                    "ok": False,
                    "error": "unknown_source",
                    "message": "Source ID is not registered",
                }
            # No source becomes citable unless the whole batch was returned.
            previous = self.cursors.copy(), self.progress.copy(), self.accessible.copy()
            try:
                result = {
                    "ok": True,
                    "results": [self.read(read, fragment_reader) for read in reads],
                }
                size_check(
                    len(_encoded(result)),
                    self.limits["judge_max_result_bytes"],
                    "result_limit",
                )
                return result
            except BaseException:
                self.cursors, self.progress, self.accessible = previous
                raise
        sid, cursor = args["source_id"], args["cursor"]
        if sid not in self.entries:
            return {
                "ok": False,
                "error": "unknown_source",
                "message": "Source ID is not registered",
            }
        sha = self.entries[sid]["sha256"]
        offset = 0
        if cursor is not None:
            value = self.cursors.get(cursor)
            if value is None or value[:2] != (sid, sha):
                return {
                    "ok": False,
                    "error": "stale_cursor",
                    "message": "Cursor does not match this immutable source",
                }
            offset = value[2]
        if (
            self.projection is None
            and fragment_reader is not None
            and not isinstance(self.bodies[sid], bytes)
        ):
            fragment = fragment_reader(sid, offset, args["max_chars"])
            part, end, total_chars = (
                fragment["text"],
                fragment["end"],
                fragment["total"],
            )
        else:
            source = self.source_bytes(sid).decode("utf-8")
            end, total_chars = min(len(source), offset + args["max_chars"]), len(source)
            part = source[offset:end]
        next_cursor = None
        if end < total_chars:
            next_cursor = secrets.token_urlsafe(24)
            self.cursors[next_cursor] = (sid, sha, end)
        result = {
            "ok": True,
            "source_id": sid,
            "sha256": sha,
            "text": part,
            "next_cursor": next_cursor,
        }
        size_check(
            len(_encoded(result)), self.limits["judge_max_result_bytes"], "result_limit"
        )
        # A long source becomes citable only after every fragment was delivered.
        progress = self.progress.get(sid, 0)
        if offset <= progress:
            self.progress[sid] = max(progress, end)
            if self.progress[sid] == total_chars:
                self.accessible.add(sid)
        return result

    def snapshot_message(self, message, *, seen=True):
        record = {
            key: message.get(key)
            for key in ("channel_id", "user_id", "ts", "thread_ts", "text")
        }
        sid = "slack:" + record["channel_id"] + ":" + record["ts"]
        return self.add(
            sid,
            "snapshot",
            {
                "service": "slack",
                "record": {"channel_id": record["channel_id"], "ts": record["ts"]},
            },
            record,
            seen=seen,
        )

    def native(self, result, call_id, seq, name=None, *, original_result=None):
        sid = self.add(
            "read:" + call_id,
            "read_result",
            {"judge_trace_seq": seq, "call_id": call_id},
            result,
            seen=True,
        )
        if self.projection is not None and original_result is not None:
            self.originals[sid].update(
                sha256=digest(original_result),
                restore=private_restore_patch(original_result, result),
                tool=name,
            )
        ids = [sid]
        for source_id, kind, locator, body in services.evidence_entries(name, result):
            ids.append(self.add(source_id, kind, locator, body, seen=True))
        for pointer, body in services.trace_facts(name, result):
            if (
                not self.current_protocol
                and name in services.hubspot_service.READS
                and "results" in result
                and not pointer.startswith("/results/")
            ):
                continue
            fact_id = self.add(
                "read:" + call_id + "#" + pointer,
                "read_result",
                {"judge_trace_seq": seq, "call_id": call_id, "json_pointer": pointer},
                body,
                seen=True,
            )
            if self.projection is not None and original_result is not None:
                original = original_result
                for token in pointer.split("/")[1:]:
                    token = token.replace("~1", "/").replace("~0", "~")
                    original = (
                        original[int(token)]
                        if isinstance(original, list)
                        else original[token]
                    )
                self.originals[fact_id].update(
                    sha256=digest(original),
                    restore=private_restore_patch(original, body),
                    original_read_id=sid,
                )
            ids.append(fact_id)
        return [self.entries[i] for i in dict.fromkeys(ids)]


def native_definitions(scope, result_limit):
    return services.definitions(scope, result_limit)


def native_source_view(entries):
    """Keep every citation ID and pointer without repeating trusted metadata."""
    return [
        (
            [entry["id"], entry["locator"]["json_pointer"]]
            if entry["kind"] == "read_result" and "json_pointer" in entry["locator"]
            else entry
        )
        for entry in entries
    ]


def read_json(path):
    return parse_json(Path(path).read_bytes())


def validate_run(run, config, task_ref, task_config_ref, root, run_files):
    expected = {
        "schema_version",
        "run_id",
        "attempt_id",
        "unit_key",
        "task_ref",
        "config_ref",
        "model_ref",
        "outcome",
        "stop_reason",
        "message_refs",
        "uncertain_delivery",
        "resolved_limits",
        "hashes",
        "provider_responses",
        "timings",
        "trace_path",
        "final_state_path",
        "error",
    }
    extensions = {"baseline_ref", "baseline_receipt_path"}
    current = {"execution_protocol"}
    check(
        isinstance(run, dict)
        and set(run) in (expected | current, expected | extensions | current)
        and type(run["schema_version"]) is int
        and run["schema_version"] == 1,
        "invalid_run",
        "Expected the complete run schema 1 structure, optionally with merged baseline fields",
    )
    if "baseline_ref" in run:
        ref = run["baseline_ref"]
        check(
            isinstance(ref, dict)
            and ref.get("mode") in {"local_declared", "verified_local"},
            "invalid_run",
            "Unknown original baseline mode",
        )
        required = {"mode", "build_ref"} | (
            {"manifest_sha256"} if ref["mode"] == "verified_local" else set()
        )
        if (
            ref["mode"] == "verified_local"
            and config["storage"].get("baseline", {}).get("source") is not None
        ):
            required.add("source")
        check(
            set(ref) == required and ref["build_ref"] == config["build_ref"],
            "pin_mismatch",
            "Original baseline source/pin disagrees",
        )
        if ref["mode"] == "verified_local":
            text(ref["manifest_sha256"], "original baseline bucket")
            check(
                ref.get("source")
                == config["storage"].get("baseline", {}).get("source"),
                "pin_mismatch",
                "Original baseline checkpoint source differs",
            )
            check(
                isinstance(config["storage"].get("baseline"), dict)
                and ref["manifest_sha256"]
                == config["storage"]["baseline"].get("manifest_sha256"),
                "pin_mismatch",
                "Original baseline bucket disagrees with config",
            )
        else:
            check(
                config["storage"].get("baseline") is None
                and run["baseline_receipt_path"] is None,
                "invalid_run",
                "Local-declared runs cannot claim a verified receipt",
            )
        check(
            run["baseline_receipt_path"] in (None, "baseline-receipt.json"),
            "invalid_run",
            "Unknown baseline receipt path",
        )
    for key in ("run_id", "attempt_id", "unit_key", "outcome", "stop_reason"):
        text(run[key], key)
    _relative(run["unit_key"], unit=True)
    check(
        run["task_ref"] == task_ref
        and type(run["task_ref"].get("version")) is int
        and run["config_ref"] == task_config_ref,
        "pin_mismatch",
        "Task/version/package/config pins do not match the original run",
    )
    expected_limits = agent_limits(config)
    check(
        run["resolved_limits"] == expected_limits
        and all(
            type(run["resolved_limits"][k]) is type(v)
            for k, v in expected_limits.items()
        ),
        "pin_mismatch",
        "Run limits disagree with the original config",
    )
    check(
        run["trace_path"] == "trace.jsonl"
        and run["final_state_path"] == "final_state/slack",
        "invalid_run",
        "Unsupported run source paths",
    )
    paths = {f["path"] for f in run_files}
    check(
        {"run.json", "trace.jsonl", "instructions.txt", "tools.json", "model.json"}
        <= paths,
        "missing_source",
        "Run bundle lacks required observable artifacts",
    )
    if current:
        check(
            "execution-protocol.json" in paths
            and any(
                digest(run["execution_protocol"]) == digest(expected)
                for expected in recorded_protocols(
                    agent_limits(config), run["model_ref"]["settings"]
                )
            )
            and digest(read_json(root / "execution-protocol.json"))
            == digest(run["execution_protocol"]),
            "pin_mismatch",
            "Missing or conflicting agent execution protocol",
        )
    allowed = ({"execution-protocol.json"} if current else set()) | {
        "run.json",
        "trace.jsonl",
        "instructions.txt",
        "tools.json",
        "model.json",
        "final_state/slack/seed.json",
        "final_state/slack/overlay.sqlite",
    }
    if "provider-native.jsonl" in paths:
        # Hashed in the bundle inventory, never added to judge-visible sources.
        allowed.add("provider-native.jsonl")
    if run.get("baseline_receipt_path") is not None:
        allowed.add("baseline-receipt.json")
        check(
            "baseline-receipt.json" in paths,
            "missing_source",
            "Pinned baseline receipt is missing",
        )
    if read_admission.supported(
        run.get("execution_protocol", {}).get("read_result_admission"),
        run["model_ref"]["settings"],
    ):
        import re

        extra = {p for p in paths if re.fullmatch(r"read-results/[0-9]{8}\.json", p)}
        allowed |= extra
        evidence_bytes = sum(
            f["bytes"]
            for f in run_files
            if f["path"] in extra
            or f["path"] in {"trace.jsonl", "provider-native.jsonl"}
        )
        check(
            evidence_bytes <= run["resolved_limits"]["max_trace_bytes"] + Trace.RESERVE,
            "invalid_run",
            "Combined trace and withheld-read evidence exceed the pinned bound",
        )
    check(paths <= allowed, "invalid_run", "Unknown run schema 1 payload file")
    fields(
        run["hashes"],
        ("instruction_sha256", "tool_definitions_sha256", "input_sha256"),
        "run.hashes",
    )
    for name, key in (
        ("instructions.txt", "instruction_sha256"),
        ("tools.json", "tool_definitions_sha256"),
    ):
        check(
            hash_file(root / name)[1] == run["hashes"][key],
            "pin_mismatch",
            "Original instruction/tool artifact hash mismatch",
        )
    instruction = config["agent_instructions"].replace(
        "{recent_activity_window_minutes}",
        str(run["resolved_limits"]["recent_activity_window_minutes"]),
    )
    instruction = services.render_guides(instruction, config["tool_scope"])
    check(
        (root / "instructions.txt").read_text() == instruction,
        "pin_mismatch",
        "Original agent instructions disagree with config",
    )
    fields(run["model_ref"], ("sha256", "settings", "route_id"), "run.model_ref")
    check(
        hash_file(root / "model.json")[1] == run["model_ref"]["sha256"]
        and load_model(root / "model.json") == run["model_ref"]["settings"]
        and type(run["model_ref"]["settings"].get("schema_version")) is int
        and type(run["model_ref"]["settings"].get("max_output_tokens")) is int,
        "pin_mismatch",
        "Original model artifact hash/settings mismatch",
    )
    text(run["model_ref"]["route_id"], "run.model_ref.route_id")
    check(
        isinstance(run["message_refs"], list)
        and isinstance(run["uncertain_delivery"], list)
        and isinstance(run["provider_responses"], list),
        "invalid_run",
        "Invalid run arrays",
    )
    timing_fields = (
        "budget_seconds_used",
        "tool_wait_seconds",
        "setup_seconds",
        "wall_seconds",
    )
    if "provider_admission" in run.get("execution_protocol", {}):
        timing_fields += ("quota_wait_seconds", "backoff_seconds")
    fields(run["timings"], timing_fields, "run.timings")
    for value in run["timings"].values():
        check(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid_run",
            "Invalid run timing",
        )
    for p in run["provider_responses"]:
        fields(
            p, ("request_id", "response_id", "model", "version"), "provider identity"
        )
        for v in p.values():
            text(v, "provider identity")
    check(
        run["outcome"] in {"completed", "timeout", "technical_error"},
        "invalid_run",
        "Unsupported run outcome",
    )
    check(
        safe_code(run["stop_reason"], "") == run["stop_reason"],
        "invalid_run",
        "Stop reason must be a stable recorded code",
    )
    check(
        (run["error"] is None) == (run["outcome"] == "completed"),
        "invalid_run",
        "Terminal failure must have its recorded diagnostic",
    )
    if run["error"] is not None:
        check(
            isinstance(run["error"], dict)
            and {"phase", "code", "reason", "request_id", "call_id", "tool"}
            <= run["error"].keys()
            and not run["error"].keys()
            - {"phase", "code", "reason", "request_id", "call_id", "tool", "cause"},
            "invalid_run",
            "Invalid run diagnostic",
        )
        check(
            run["error"]["code"] == run["stop_reason"]
            and run["error"]["phase"] in {"setup", "model", "tool"},
            "invalid_run",
            "Recorded failure code/phase disagrees with the terminal outcome",
        )
    marker_path = root / "_complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        # Validate the completed bundle without opening a transport.
        marker = validate_completion_manifest(marker, run["unit_key"])
        identity = {
            k: run[k]
            for k in ("run_id", "attempt_id", "task_ref", "config_ref", "model_ref")
        }
        if "baseline_ref" in run:
            identity["baseline_ref"] = run["baseline_ref"]
        if current:
            identity["execution_protocol"] = run["execution_protocol"]
        check(
            marker["files"] == run_files and marker["metadata"] == identity,
            "pin_mismatch",
            "Original completion marker does not match the supplied run files/identity",
        )


def index_trace(root, run, sources, *, describe_sources=False):
    """Validate the full observable lifecycle while indexing byte offsets."""
    sources.current_protocol = True
    path = root / "trace.jsonl"
    context = None
    pending, successful, seen_calls = {}, [], set()
    protocol = run.get("execution_protocol")
    check(
        isinstance(protocol, dict)
        and "max_read_calls_per_response" in run["resolved_limits"]
        and any(
            digest(protocol) == digest(expected)
            for expected in recorded_protocols(
                run["resolved_limits"], run["model_ref"]["settings"]
            )
        )
        and (root / "execution-protocol.json").is_file()
        and digest(read_json(root / "execution-protocol.json")) == digest(protocol),
        "pin_mismatch",
        "Missing or conflicting recorded execution protocol",
    )
    group_calls, group_index, declined, invalid_ids = [], 0, None, False
    active_tool = None
    withheld_paths = set()
    admission_profile = protocol.get("read_result_admission") if protocol else None
    try:
        replay_budget = (
            read_admission.ReplayBudget(
                root,
                run["model_ref"]["settings"],
                admission_profile,
                run["resolved_limits"],
                provider_admission=protocol.get("provider_admission"),
                route_id=run["model_ref"]["route_id"],
            )
            if admission_profile
            else None
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise JudgeError(
            "invalid_trace", "Invalid recorded read-admission policy or native trace"
        ) from exc
    history, tools = None, read_json(root / "tools.json")
    previous_elapsed = previous_charge = 0.0
    request_start = request_deadline = None
    request_charge = 0.0
    active_request, last, error, finished = None, None, None, False
    timeout_verified = False
    providers = []
    with path.open("rb") as stream:
        seq = 0
        while True:
            offset = stream.tell()
            raw = stream.readline(sources.limits["judge_max_source_file_bytes"] + 1)
            if not raw:
                break
            size_check(
                len(raw),
                sources.limits["judge_max_source_file_bytes"],
                "source_file_limit",
            )
            try:
                event = parse_json(raw)
            except (ValueError, TypeError, UnicodeError):
                raise JudgeError(
                    "invalid_trace", "Original trace is not valid finite JSON"
                ) from None
            seq += 1
            check(
                isinstance(event, dict)
                and {
                    "seq",
                    "kind",
                    "utc_time",
                    "elapsed_seconds",
                    "charged_model_seconds",
                    "payload",
                }
                <= event.keys()
                and type(event["seq"]) is int
                and event["seq"] == seq
                and last != "stop",
                "invalid_trace",
                "Trace sequence or terminal boundary is inconsistent",
            )
            kind, payload = event["kind"], event["payload"]
            check(
                isinstance(payload, dict)
                and ((error is None and not finished) or kind == "stop"),
                "invalid_trace",
                "Events continued after the recorded terminal operation",
            )
            elapsed, charge = event["elapsed_seconds"], event["charged_model_seconds"]
            check(
                all(
                    type(v) in (int, float) and math.isfinite(v) and v >= 0
                    for v in (elapsed, charge)
                )
                and elapsed >= previous_elapsed
                and charge >= previous_charge,
                "invalid_trace",
                "Trace timing is inconsistent",
            )
            if kind not in {"model_response", "error"}:
                check(
                    charge == previous_charge,
                    "invalid_trace",
                    "Model time changed outside a model response/failure",
                )
            locator = {"seq": seq}
            if describe_sources:
                locator.update(event_kind=kind, record_bytes=len(raw))
                call = payload if kind == "tool_request" else None
                if kind == "tool_result":
                    pending_call = active_tool or pending.get(event.get("call_id"))
                    call = pending_call[0] if isinstance(pending_call, tuple) else None
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    locator["tool_name"] = call["name"]
            sources.add(
                f"trace:{seq}",
                "trace",
                locator,
                event,
                trace_file=path,
                offset=offset,
                length=len(raw),
            )
            if kind == "context":
                check(
                    seq == 1 and context is None,
                    "invalid_trace",
                    "Context must appear once at the start",
                )
                fields(
                    payload, ("instruction", "tools", "initial_input"), "trace context"
                )
                context = payload["initial_input"]
                check(
                    payload["tools"] == tools
                    and payload["instruction"].encode()
                    == (root / "instructions.txt").read_bytes()
                    and digest(context) == run["hashes"]["input_sha256"],
                    "context_mismatch",
                    "Recorded context/instruction/tool hashes disagree",
                )
                history = [
                    {"role": "developer", "content": payload["instruction"]},
                    {"role": "user", "content": _encoded(context).decode()},
                ]
            elif kind == "model_request":
                check(
                    history is not None and request_start is None and not finished,
                    "invalid_trace",
                    "Invalid model request lifecycle",
                )
                if protocol:
                    check(
                        not pending
                        and active_tool is None
                        and group_index == len(group_calls)
                        and not invalid_ids,
                        "invalid_trace",
                        "Previous tool group was not fully resolved",
                    )
                expected_history = (
                    history + [runtime_status(protocol, charge)]
                    if protocol
                    else history
                )
                check(
                    payload.get("input") == expected_history
                    and payload.get("tools") == tools,
                    "invalid_trace",
                    "Model history/tool artifacts disagree with the full trace",
                )
                check(
                    payload.get("model") == run["model_ref"]["settings"]["model_id"]
                    and payload.get("store") is False
                    and payload.get("parallel_tool_calls")
                    is (True if protocol else False)
                    and payload.get("include") == ["reasoning.encrypted_content"]
                    and payload.get("reasoning")
                    == {"effort": run["model_ref"]["settings"]["reasoning_effort"]}
                    and payload.get("max_output_tokens")
                    == run["model_ref"]["settings"]["max_output_tokens"],
                    "invalid_trace",
                    "Original model settings disagree with the request",
                )
                request_start, request_charge = elapsed, charge
                request_deadline = event.get("deadline_seconds")
                check(
                    type(request_deadline) in (int, float)
                    and request_deadline > 0
                    and abs(
                        request_deadline
                        - (run["resolved_limits"]["agent_model_seconds"] - charge)
                    )
                    < 1e-6,
                    "invalid_trace",
                    "Invalid recorded remaining model deadline",
                )
                if replay_budget is not None:
                    try:
                        replay_budget.request(payload)
                    except (ValueError, RuntimeInputError, KeyError, TypeError) as exc:
                        raise JudgeError(
                            "invalid_trace",
                            "Native admission request cannot be reconstructed",
                        ) from exc
                active_request = event.get("request_id")
                check(
                    isinstance(active_request, str) and active_request,
                    "invalid_trace",
                    "Missing request ID",
                )
                pending = {}
            elif kind == "model_response":
                check(
                    request_start is not None
                    and event.get("request_id") == active_request,
                    "invalid_trace",
                    "Response request ID mismatch",
                )
                check(
                    0 <= charge - request_charge <= elapsed - request_start + 1e-6,
                    "invalid_trace",
                    "Charged model time disagrees with request interval",
                )
                request_start = None
                if replay_budget is not None:
                    try:
                        replay_budget.response(active_request, payload)
                    except (ValueError, RuntimeInputError, KeyError, TypeError) as exc:
                        raise JudgeError(
                            "invalid_trace",
                            "Native admission usage cannot be reconstructed",
                        ) from exc
                providers.append(
                    {
                        "request_id": event.get("provider_request_id") or "unavailable",
                        "response_id": payload.get("id", "unavailable"),
                        "model": payload.get("model", "unavailable"),
                        "version": payload.get("model_version", "unavailable"),
                    }
                )
                if (
                    payload.get("status") == "completed"
                    and not payload.get("error")
                    and not (event.get("http_status") and event["http_status"] >= 400)
                ):
                    output = payload.get("output")
                    check(
                        isinstance(output, list),
                        "invalid_trace",
                        "Invalid original provider output",
                    )
                    history.extend(output)
                    if protocol:
                        # Unsupported/incomplete output and deadline responses are terminal
                        # before preflight in the writer. They produce no tool effects.
                        usable = all(
                            isinstance(item, dict)
                            and item.get("type")
                            in {"message", "reasoning", "function_call"}
                            and item.get("status", "completed") == "completed"
                            for item in output
                        )
                        group_calls = (
                            [
                                item
                                for item in output
                                if isinstance(item, dict)
                                and item.get("type") == "function_call"
                            ]
                            if usable
                            and charge < run["resolved_limits"]["agent_model_seconds"]
                            else []
                        )
                        group_index = 0
                        declined, invalid_ids = preflight(
                            group_calls,
                            seen_calls,
                            tools,
                            [t["name"] for t in tools if t["name"] != "finish_task"],
                            protocol,
                        )
                        seen_calls.update(
                            c["call_id"]
                            for c in group_calls
                            if isinstance(c.get("call_id"), str)
                        )
                    else:
                        for item in output:
                            if item.get("type") == "function_call":
                                cid = item.get("call_id")
                                check(
                                    isinstance(cid, str)
                                    and cid
                                    and cid not in seen_calls,
                                    "invalid_trace",
                                    "Duplicate/missing original call ID",
                                )
                                seen_calls.add(cid)
                                pending[cid] = item
            elif kind == "tool_request":
                cid = event.get("call_id")
                if protocol:
                    check(
                        active_tool is None and group_index < len(group_calls),
                        "invalid_trace",
                        "Tool group request order or identity differs",
                    )
                    expected_call = group_calls[group_index]
                    if "requested_arguments" in event:
                        definition = next(
                            (
                                t
                                for t in tools
                                if t["name"] == expected_call.get("name")
                            ),
                            {},
                        )
                        check(
                            declined is None
                            and expected_call.get("name") == "posthog_list_catalog"
                            and definition.get("parameters", {})
                            .get("properties", {})
                            .get("projection", {})
                            .get("default")
                            == "summary"
                            and parse_json(expected_call["arguments"])
                            == event["requested_arguments"],
                            "invalid_trace",
                            "Unverified agent discovery default",
                        )
                        normalized = services.agent_arguments(
                            expected_call["name"], event["requested_arguments"]
                        )
                        check(
                            normalized != event["requested_arguments"],
                            "invalid_trace",
                            "Unexpected argument normalization",
                        )
                        expected_call = {
                            **expected_call,
                            "arguments": _encoded(normalized).decode(),
                        }
                    check(
                        expected_call == payload
                        and cid == payload.get("call_id")
                        and event.get("request_id") == active_request,
                        "invalid_trace",
                        "Tool group request order or identity differs",
                    )
                    active_tool = (payload, seq)
                else:
                    check(
                        cid in pending
                        and pending[cid] == payload
                        and event.get("request_id") == active_request,
                        "invalid_trace",
                        "Tool call is not in its recorded provider response",
                    )
                    check(
                        not any(c[1] == cid for c in successful),
                        "invalid_trace",
                        "Duplicate delivered call",
                    )
                    pending[cid] = (payload, seq)
            elif kind == "tool_result":
                cid = event.get("call_id")
                if protocol:
                    check(
                        active_tool is not None
                        and cid == active_tool[0].get("call_id")
                        and event.get("request_id") == active_request,
                        "invalid_trace",
                        "Tool result lacks its matching ordered request",
                    )
                    call, request_seq = active_tool
                    if declined is not None:
                        check(
                            payload == declined[group_index],
                            "invalid_trace",
                            "Rejected group result differs from preflight",
                        )
                    active_tool = None
                    group_index += 1
                else:
                    check(
                        cid in pending
                        and isinstance(pending[cid], tuple)
                        and event.get("request_id") == active_request,
                        "invalid_trace",
                        "Tool result lacks its matching request",
                    )
                    call, request_seq = pending.pop(cid)
                if "read_admission" in event:
                    check(
                        protocol
                        and read_admission.supported(
                            protocol.get("read_result_admission"),
                            run["model_ref"]["settings"],
                        )
                        and declined is None
                        and call.get("name") in services.read_names(),
                        "invalid_trace",
                        "Withheld-result proof is not an executed read",
                    )
                    try:
                        name = read_admission.validate(
                            root,
                            event,
                            call,
                            lambda raw: read_admission.candidate(
                                run["model_ref"]["settings"],
                                history,
                                tools,
                                protocol,
                                charge,
                                cid,
                                raw,
                                group_calls[group_index:],
                            ),
                            run["resolved_limits"],
                            replay_budget,
                        )
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        raise JudgeError(
                            "invalid_trace",
                            "Withheld-result evidence failed validation",
                        ) from exc
                    check(
                        name not in withheld_paths,
                        "invalid_trace",
                        "Duplicate withheld-result evidence",
                    )
                    withheld_paths.add(name)
                else:
                    check(
                        payload.get("error") != read_admission.ERROR["error"],
                        "invalid_trace",
                        "Withheld result lacks its raw proof",
                    )
                check(
                    isinstance(payload, dict)
                    and (
                        type(payload.get("ok")) is bool
                        or services.native_success(call.get("name"), payload)
                        or services.native_error(call.get("name"), payload)
                    ),
                    "invalid_trace",
                    "Invalid recorded tool result",
                )
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": cid,
                        "output": _encoded(payload).decode(),
                    }
                )
                if (
                    isinstance(call.get("name"), str)
                    and call["name"] in WRITES
                    and payload["ok"]
                ):
                    successful.append((call, cid, payload, request_seq, seq))
                if call.get("name") == "finish_task" and payload["ok"]:
                    check(
                        parse_json(call["arguments"]) == {}
                        and not pending
                        and (not protocol or len(group_calls) == 1),
                        "invalid_trace",
                        "finish_task was not standalone",
                    )
                    finished = True
            elif kind == "reminder":
                check(
                    history is not None and request_start is None,
                    "invalid_trace",
                    "Reminder outside the conversation",
                )
                if protocol:
                    check(
                        not group_calls
                        and payload
                        == {
                            "role": "user",
                            "content": load_text("agent/finish_reminder.txt"),
                        },
                        "invalid_trace",
                        "Unexpected reminder in a tool group",
                    )
                history.append(payload)
            elif kind == "error":
                check(
                    error is None and payload == run["error"],
                    "invalid_trace",
                    "Run/trace errors disagree",
                )
                error = payload
                if request_start is None:
                    check(
                        charge == previous_charge,
                        "invalid_trace",
                        "Model time changed without an active model request",
                    )
                if request_start is not None:
                    check(
                        error["phase"] == "model"
                        and error["request_id"] == active_request
                        and 0
                        <= charge - request_charge
                        <= elapsed - request_start + 1e-6,
                        "invalid_trace",
                        "Model failure timing/phase mismatch",
                    )
                    timeout_verified = (
                        charge >= run["resolved_limits"]["agent_model_seconds"]
                        and elapsed - request_start >= request_deadline
                    )
                    request_start = None
            elif kind == "stop":
                check(
                    payload
                    == {"outcome": run["outcome"], "stop_reason": run["stop_reason"]},
                    "invalid_trace",
                    "Run and terminal trace disagree",
                )
            else:
                raise JudgeError("invalid_trace", "Unknown original trace event kind")
            last, previous_elapsed, previous_charge = kind, elapsed, charge
    check(
        {p.relative_to(root).as_posix() for p in (root / "read-results").glob("*.json")}
        == withheld_paths,
        "invalid_trace",
        "Missing or unreferenced withheld-result evidence",
    )
    check(
        last == "stop"
        and providers == run["provider_responses"]
        and error == run["error"]
        and abs(previous_charge - run["timings"]["budget_seconds_used"]) < 1e-6,
        "invalid_trace",
        "Run terminal/provider/timing metadata disagrees with trace",
    )
    check(
        not run["uncertain_delivery"],
        "uncertain_delivery",
        "Original attempt has uncertain delivery; no guessing or resending",
    )
    if run["outcome"] == "completed" and run["stop_reason"] == "finish_task":
        check(
            finished
            and error is None
            and (not protocol or group_index == len(group_calls)),
            "invalid_trace",
            "Completed attempt lacks successful standalone finish_task",
        )
        eligible, budget = True, False
    elif run["outcome"] == "timeout" and run["stop_reason"] == "model_budget":
        check(
            error is not None
            and error["phase"] == "model"
            and previous_charge >= run["resolved_limits"]["agent_model_seconds"],
            "invalid_trace",
            "Model-budget stop lacks expired model timing",
        )
        eligible, budget = True, True
    elif (
        run["outcome"] == "timeout"
        and run["stop_reason"] == "model_timeout"
        and timeout_verified
    ):
        eligible, budget = True, True
    else:
        eligible, budget = False, False
    return context, successful, eligible, budget


def recover_messages(root, run, successful, task, sources):
    check(
        len(successful) == len(run["message_refs"]),
        "delivery_mismatch",
        "Successful writes and message_refs differ",
    )
    messages, keys = [], set()
    database = root / "final_state/slack/overlay.sqlite"
    if not database.exists():
        check(
            not successful, "missing_source", "Final SQLite delivery rows are missing"
        )
        return messages
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        db.execute("PRAGMA query_only=ON")
        for index, (ref, recorded) in enumerate(zip(run["message_refs"], successful)):
            call, cid, result, _, _ = recorded
            required = {"call_id", "tool", "channel_id", "ts"}
            check(
                isinstance(ref, dict)
                and required <= ref.keys()
                and not ref.keys() - (required | {"destination"}),
                "delivery_mismatch",
                "Invalid message_ref fields",
            )
            check(
                ref["call_id"] == cid and ref["tool"] == call["name"],
                "delivery_mismatch",
                "Message refs are out of recorded call order",
            )
            receipt = result.get("message", result)
            check(
                receipt.get("channel_id") == ref["channel_id"]
                and receipt.get("ts") == ref["ts"],
                "delivery_mismatch",
                "Successful receipt disagrees with message_ref",
            )
            args = parse_json(call["arguments"])
            row = db.execute(
                "SELECT channel_id,ts,user_id,text,thread_ts,operation FROM messages WHERE channel_id=? AND ts=?",
                (ref["channel_id"], ref["ts"]),
            ).fetchone()
            check(
                row is not None
                and row[2] == "U_AGENT"
                and receipt.get("user_id") == row[2]
                and row[3] == args.get("text")
                and row[5] == "upsert",
                "delivery_mismatch",
                "Final message author/text/operation disagrees with the successful call",
            )
            check(
                (row[0], row[1]) not in keys,
                "delivery_mismatch",
                "Duplicate delivered message",
            )
            keys.add((row[0], row[1]))
            if call["name"] == "slack_send_dm":
                dest = {"kind": "dm", "user_id": args["user_id"]}
                check(
                    row[4] is None
                    and result.get("destination", {}).get("user_id") == args["user_id"]
                    and ref.get("destination") == result.get("destination"),
                    "delivery_mismatch",
                    "DM receipt/recipient disagrees",
                )
                stored = db.execute(
                    "SELECT channel_json FROM dm_conversations WHERE user_id=?",
                    (args["user_id"],),
                ).fetchone()
                channel = parse_json(stored[0]) if stored else {}
                check(
                    channel.get("id") == row[0]
                    and set(channel.get("member_ids", []))
                    == {"U_AGENT", args["user_id"]},
                    "delivery_mismatch",
                    "Final DM channel does not match its recipient",
                )
            else:
                check(
                    args.get("channel_id") == row[0]
                    and args.get("thread_ts") == row[4],
                    "delivery_mismatch",
                    "Channel/thread placement disagrees with the successful call",
                )
                dm_rows = db.execute(
                    "SELECT user_id,channel_json FROM dm_conversations"
                ).fetchall()
                dm_users = [
                    uid for uid, raw in dm_rows if parse_json(raw).get("id") == row[0]
                ]
                check(
                    len(dm_users) <= 1,
                    "delivery_mismatch",
                    "Ambiguous persisted DM conversation",
                )
                dest = (
                    {"kind": "dm", "user_id": dm_users[0]}
                    if dm_users
                    else (
                        {"kind": "channel", "channel_id": row[0]}
                        if row[4] is None
                        else {
                            "kind": "thread",
                            "channel_id": row[0],
                            "thread_ts": row[4],
                        }
                    )
                )
                check(
                    "destination" not in ref,
                    "delivery_mismatch",
                    "Unexpected DM destination in a channel call",
                )
            if "message" in result:
                check(
                    receipt.get("text") == row[3]
                    and receipt.get("thread_ts") == row[4],
                    "delivery_mismatch",
                    "Full message receipt disagrees with final SQLite row",
                )
            message = {
                "evidence_id": f"message:{index}",
                "call_id": cid,
                "tool": call["name"],
                "channel_id": row[0],
                "ts": row[1],
                "user_id": row[2],
                "text": row[3],
                "destination": dest,
            }
            sources.add(
                message["evidence_id"],
                "message",
                {"index": index, "channel_id": row[0], "ts": row[1], "call_id": cid},
                message,
                seen=True,
            )
            messages.append(message)
        delta_keys = {
            (c["source"]["record"].get("channel_id"), c["source"]["record"].get("ts"))
            for c in task["changes"]
            if "ts" in c["source"]["record"]
        }
        all_keys = set(db.execute("SELECT channel_id,ts FROM messages"))
        check(
            all_keys == keys | delta_keys,
            "delivery_mismatch",
            "Final SQLite contains missing or unrecorded messages",
        )
    return messages


def runtime_sha256():
    from .runtime_identity import fingerprint

    return fingerprint()


def judgment_identity(judgment):
    return {
        **(
            {"schema_version": judgment["schema_version"]}
            if judgment.get("schema_version", 1) >= 2
            else {}
        ),
        "judgment_id": judgment["judgment_id"],
        "task_ref": judgment["task_ref"],
        "config_ref": judgment["config_ref"],
        "run_ref": judgment["run_ref"],
        "baseline_ref": judgment["baseline_ref"],
        "judge_model_ref": judgment["judge"]["model_ref"],
        "rubric_version": judgment["judge"]["rubric_version"],
        "prompt_version": judgment["judge"]["prompt_version"],
        "prompt_sha256": judgment["judge"]["prompt_sha256"],
        "tools_sha256": judgment["judge"]["tools_sha256"],
        "limits_sha256": judgment["judge"]["limits_sha256"],
        "runtime_sha256": judgment["judge"]["runtime_sha256"],
        **(
            {
                "judge_model_budget_override": judgment["judge"][
                    "judge_model_budget_override"
                ]
            }
            if "judge_model_budget_override" in judgment["judge"]
            else {}
        ),
        **(
            {"read_result_admission": judgment["judge"]["read_result_admission"]}
            if "read_result_admission" in judgment["judge"]
            else {}
        ),
        **(
            {
                "anonymous_projection_version": judgment["judge"][
                    "anonymous_projection_version"
                ]
            }
            if "anonymous_projection_version" in judgment["judge"]
            else {}
        ),
    }


def anonymous_binding(run_files, package_files, config_path):
    return digest(
        {
            "run_files": run_files,
            "package_files": package_files,
            "config_sha256": hash_file(config_path)[1],
            "version": ANONYMOUS_VERSION,
        }
    )


def prepare_anonymous_answer(*, package, config_path, run_bundle):
    """Offline, model-independent preparation reusable by both judge workers.

    Keep the returned handle private. It contains lazy references to original
    trace bytes, not a copy of company data or a judge's investigation state.
    """
    inputs = [Path(package), Path(run_bundle), Path(config_path)]
    check(
        not any(path.is_symlink() for path in inputs),
        "unsafe_source",
        "Immutable roots cannot be symlinks",
    )
    package, root, config_path = (path.resolve() for path in inputs)
    size_check(
        config_path.stat().st_size,
        DEFAULT_LIMITS["judge_max_source_file_bytes"],
        "source_file_limit",
    )
    config_hash = hash_file(config_path)[1]
    config = load_config(config_path)
    limits = resolve_limits(config)
    verify_package(package, config_path=config_path)
    run_files, package_files = inventory(root, limits, exclude_marker=True), inventory(
        package, limits
    )
    run, task = read_json(root / "run.json"), load_task(package / "task.json")
    task_ref = {
        "task_id": task["meta"]["task_id"],
        "version": task["meta"]["version"],
        "manifest_sha256": hash_file(package / "manifest.json")[1],
        "build_ref": config["build_ref"],
    }
    validate_run(run, config, task_ref, task["meta"]["config_ref"], root, run_files)
    sources = Sources(limits, projection=Projection())
    sources.add("task", "task", {"json_pointer": ""}, task, seen=True)
    for name in ("instructions.txt", "tools.json", "model.json"):
        sources.add(
            "artifact:" + name,
            "artifact",
            {"path": name},
            raw=(root / name).read_bytes(),
            seen=name == "instructions.txt",
        )
    context, successful, eligible, agent_budget = index_trace(
        root, run, sources, describe_sources=True
    )
    delivered = recover_messages(root, run, successful, task, sources)
    check(
        context is not None,
        "agent_terminal_failure",
        "Original attempt has no initial context",
    )
    for message in context["messages"]:
        sources.snapshot_message(message)
    inp = {
        "schema_version": (
            3
            if run.get("baseline_ref", {}).get("mode") == "verified_local"
            else SCHEMA_VERSION
        ),
        "task_ref": task_ref,
        "config_ref": task["meta"]["config_ref"],
        "run_ref": {
            **{
                key: run[key]
                for key in ("run_id", "attempt_id", "unit_key", "schema_version")
            },
            "run_json_sha256": hash_file(root / "run.json")[1],
            "bundle_sha256": digest(run_files),
        },
        "baseline_ref": run.get(
            "baseline_ref", {"mode": "local_declared", "build_ref": config["build_ref"]}
        ),
        "task": task,
        "policy": {
            "snapshot_time": config["snapshot_time"],
            "recent_activity_window_minutes": run["resolved_limits"][
                "recent_activity_window_minutes"
            ],
            "agent_instructions": (root / "instructions.txt").read_text(),
            "allowed_tools": config["tool_scope"],
        },
        "attempt": {
            "outcome": run["outcome"],
            "stop_reason": run["stop_reason"],
            "model_budget_termination": agent_budget,
        },
        "initial_context": context,
        "delivered_messages": delivered,
        "sources": [],
    }
    prepared = freeze(
        inp, sources, anonymous_binding(run_files, package_files, config_path)
    )
    size_check(
        len(prepared.input_bytes),
        limits["judge_initial_input_max_bytes"],
        "initial_input_limit",
    )
    # Validation can be slow. Reject concurrent edits before sharing preparation.
    check(
        run_files == inventory(root, limits, exclude_marker=True)
        and package_files == inventory(package, limits)
        and config_hash == hash_file(config_path)[1],
        "changed_input",
        "Original sources changed while preparing anonymous input",
    )
    return prepared


def retry_publication(bundle, *, unit_key, artifact_store):
    """Upload exactly the saved terminal bytes; never invoke a model or read worker."""
    bundle = Path(bundle)
    judgment = read_json(bundle / "judgment.json")
    check(
        judgment["unit_key"] == unit_key,
        "pin_mismatch",
        "Retry unit differs from the frozen terminal bundle",
    )
    return artifact_store.publish_directory(
        bundle, unit_key, judgment_identity(judgment)
    )


def judge_material(config, model, *, judge_model_budget_override=None):
    """The exact deterministic protocol shared by preparation and execution."""
    limits = resolve_limits(
        config, judge_model_budget_override=judge_model_budget_override
    )
    instruction = (
        PROMPT.read_bytes()
        + b"\n\n"
        + services.discovery_guide(config["tool_scope"]).encode("utf-8")
    )
    tools = [
        {"type": "function", "strict": False, **t}
        for t in native_definitions(
            set(config["tool_scope"]) & services.read_names(),
            limits["judge_max_result_bytes"],
        )
    ] + [READ_SOURCE, SUBMIT]
    parallel_reads = model.get("judge_parallel_reads", True)
    version = (
        PROMPT_VERSION
        + "+guides-1+source-index-1+"
        + CITATION_FEEDBACK_VERSION
        + "+compact-native-sources-1+"
        + ERROR_POLICY_VERSION
        + "+"
        + EVALUATION_VERSION
        + ("+read-groups-1" if parallel_reads else "")
    )
    return dict(
        instruction=instruction,
        tools=tools,
        limits=limits,
        prompt_version=version,
        rubric=RUBRIC_VERSION,
    )


def run_judgment(**kwargs):
    """Run the asynchronous runtime from synchronous Python code."""
    return asyncio.run(run_judgment_async(**kwargs))


async def run_judgment_async(
    *,
    package,
    config_path,
    run_bundle,
    model_path,
    baseline_dir=None,
    baseline_build_ref=None,
    work_dir,
    judgment_id,
    unit_key,
    adapter,
    artifact_store=None,
    offline=False,
    environment_factory=ReadOnlySlackEnvironment,
    linear_baseline_dir=None,
    notion_baseline_dir=None,
    stripe_baseline_dir=None,
    calendar_baseline_dir=None,
    baseline_mode="verified_local",
    baseline_store_factory=None,
    baseline_cache_dir=None,
    service_baseline_dirs=None,
    recoverable=False,
    anonymous_preparation=None,
    judge_model_budget_override=None,
):
    service_baseline_dirs = services.local_roots(
        service_baseline_dirs,
        notion=notion_baseline_dir,
        stripe=stripe_baseline_dir,
        calendar=calendar_baseline_dir,
    )
    notion_baseline_dir = service_baseline_dirs.pop("notion", None)
    """Synchronous offline API. Adapter has the unchanged Responses protocol."""
    return await _run_judgment(
        package=package,
        config_path=config_path,
        run_bundle=run_bundle,
        model_path=model_path,
        baseline_dir=baseline_dir,
        baseline_build_ref=baseline_build_ref,
        work_dir=work_dir,
        judgment_id=judgment_id,
        unit_key=unit_key,
        adapter=adapter,
        artifact_store=artifact_store,
        offline=offline,
        environment_factory=environment_factory,
        linear_baseline_dir=linear_baseline_dir,
        notion_baseline_dir=notion_baseline_dir,
        stripe_baseline_dir=stripe_baseline_dir,
        calendar_baseline_dir=calendar_baseline_dir,
        baseline_mode=baseline_mode,
        baseline_store_factory=baseline_store_factory,
        baseline_cache_dir=baseline_cache_dir,
        service_baseline_dirs=service_baseline_dirs,
        recoverable=recoverable,
        anonymous_preparation=anonymous_preparation,
        judge_model_budget_override=judge_model_budget_override,
    )


async def _run_judgment(
    *,
    package,
    config_path,
    run_bundle,
    model_path,
    baseline_dir,
    baseline_build_ref,
    work_dir,
    judgment_id,
    unit_key,
    adapter,
    artifact_store,
    offline,
    environment_factory,
    linear_baseline_dir=None,
    notion_baseline_dir=None,
    stripe_baseline_dir=None,
    calendar_baseline_dir=None,
    baseline_mode="verified_local",
    baseline_store_factory=None,
    baseline_cache_dir=None,
    service_baseline_dirs=None,
    recoverable=False,
    anonymous_preparation=None,
    judge_model_budget_override=None,
):
    service_baseline_dirs = services.local_roots(
        service_baseline_dirs,
        notion=notion_baseline_dir,
        stripe=stripe_baseline_dir,
        calendar=calendar_baseline_dir,
    )
    notion_baseline_dir = service_baseline_dirs.pop("notion", None)
    started = time.monotonic()
    paths = [Path(p) for p in (package, config_path, run_bundle, model_path)]
    work = Path(work_dir)
    config_path = paths[1]
    size_check(
        config_path.stat().st_size,
        DEFAULT_LIMITS["judge_max_source_file_bytes"],
        "source_file_limit",
    )
    config = load_config(config_path)
    limits = resolve_limits(
        config, judge_model_budget_override=judge_model_budget_override
    )
    if baseline_mode == "local_declared":
        check(
            config["storage"].get("baseline") is None
            and baseline_dir is not None
            and baseline_build_ref is not None,
            "baseline_mode_conflict",
            "local_declared needs local baseline/pin without storage.baseline",
        )
        settings = None
    else:
        settings = baseline_settings(
            config,
            baseline_mode,
            baseline_dir,
            baseline_build_ref,
            linear_baseline_dir,
            notion_baseline_dir=notion_baseline_dir,
            service_baseline_dirs=service_baseline_dirs,
        )
    if baseline_cache_dir is not None:
        check(
            settings is not None,
            "baseline_mode_conflict",
            "baseline_cache_dir requires verified_local",
        )
        settings["cache_dir"] = str(Path(baseline_cache_dir).expanduser().resolve())
    verified = settings is not None
    if verified:
        baseline_build_ref = config["build_ref"]
        paths.append(Path(settings["root"]))
    else:
        paths.append(Path(baseline_dir))
        if linear_baseline_dir is not None:
            paths.append(Path(linear_baseline_dir))
        if notion_baseline_dir is not None:
            paths.append(Path(notion_baseline_dir))
    paths.extend(
        Path(p) for p in (service_baseline_dirs or {}).values() if p is not None
    )
    for path in paths:
        check(
            not path.is_symlink(), "unsafe_source", "Immutable roots cannot be symlinks"
        )
        check(
            not work.resolve().is_relative_to(path.resolve())
            and not path.resolve().is_relative_to(work.resolve()),
            "unsafe_work",
            "work_dir must be outside every immutable input",
        )
    paths = [p.resolve() for p in paths]
    package, config_path, root, model_path = paths[:4]
    baseline_dir = None if verified else paths[4]
    linear_baseline_dir = (
        Path(linear_baseline_dir).resolve() if linear_baseline_dir is not None else None
    )
    notion_baseline_dir = (
        Path(notion_baseline_dir).resolve() if notion_baseline_dir is not None else None
    )
    service_baseline_dirs = {
        k: Path(p).resolve()
        for k, p in (service_baseline_dirs or {}).items()
        if p is not None
    }
    work = work.resolve()
    if verified:
        cache_root = Path(settings["cache_dir"]).resolve()
        for path in [work, *paths]:
            check(
                not path.is_relative_to(cache_root)
                and not cache_root.is_relative_to(path),
                "unsafe_work",
                "Cache must be separate from work and immutable sources",
            )
    run_files = inventory(root, limits, exclude_marker=True)
    baseline_files = [] if verified else inventory(baseline_dir, limits)
    linear_files = (
        inventory(linear_baseline_dir, limits)
        if linear_baseline_dir is not None
        else []
    )
    notion_files = (
        catalog_inventory(notion_baseline_dir, limits)
        if notion_baseline_dir is not None
        else []
    )
    package_files = inventory(package, limits)
    extra_files = [
        {"path": str(p), "bytes": hash_file(p)[0], "sha256": hash_file(p)[1]}
        for p in (config_path, model_path, PROMPT)
    ]
    inventories = [(root, run_files), (package, package_files)]
    if not verified:
        inventories.append((baseline_dir, baseline_files))
    if linear_baseline_dir is not None:
        inventories.append((linear_baseline_dir, linear_files))
    if notion_baseline_dir is not None:
        inventories.append((notion_baseline_dir, notion_files))
    if linear_baseline_dir is not None or notion_baseline_dir is not None:
        # Service-qualified paths prevent collisions between separate roots.
        baseline_files = sorted(
            [{**f, "path": "slack/" + f["path"]} for f in baseline_files]
            + [{**f, "path": "linear/" + f["path"]} for f in linear_files]
            + [{**f, "path": "notion/" + f["path"]} for f in notion_files],
            key=lambda f: f["path"],
        )
    service_files = {}
    if (
        service_baseline_dirs
        and linear_baseline_dir is None
        and notion_baseline_dir is None
    ):
        baseline_files = [{**f, "path": "slack/" + f["path"]} for f in baseline_files]
    for name, native_root in service_baseline_dirs.items():
        files = (
            catalog_inventory(native_root, limits)
            if name == "notion"
            else inventory(native_root, limits)
        )
        service_files[name] = files
        inventories.append((native_root, files))
        baseline_files += [{**f, "path": name + "/" + f["path"]} for f in files]
    baseline_files = sorted(baseline_files, key=lambda f: f["path"])
    marker_hash = (
        hash_file(root / "_complete.json")
        if (root / "_complete.json").exists()
        else None
    )
    total = (
        sum(f["bytes"] for _, files in inventories for f in files)
        + sum(f["bytes"] for f in extra_files)
        + (marker_hash[0] if marker_hash else 0)
    )
    size_check(total, limits["judge_max_source_bytes"], "source_limit")
    size_check(
        sum(len(fs) for _, fs in inventories) + len(extra_files),
        limits["judge_max_source_count"],
        "source_count_limit",
    )
    for f in extra_files:
        size_check(
            f["bytes"], limits["judge_max_source_file_bytes"], "source_file_limit"
        )
    verify_package(package, config_path=config_path)
    task, model, run = (
        load_task(package / "task.json"),
        load_judge_model(model_path),
        read_json(root / "run.json"),
    )
    check(
        baseline_build_ref == config["build_ref"],
        "pin_mismatch",
        "Declared local baseline build_ref disagrees with config",
    )
    for name, value in (("judgment_id", judgment_id), ("unit_key", unit_key)):
        text(value, name)
    _relative(unit_key, unit=True)
    check(
        artifact_store is not None or offline,
        "storage_required",
        "ArtifactStore is required unless offline=True",
    )
    task_ref = {
        "task_id": task["meta"]["task_id"],
        "version": task["meta"]["version"],
        "manifest_sha256": hash_file(package / "manifest.json")[1],
        "build_ref": config["build_ref"],
    }
    validate_run(run, config, task_ref, task["meta"]["config_ref"], root, run_files)
    run_ref = {
        k: run[k] for k in ("run_id", "attempt_id", "unit_key", "schema_version")
    }
    run_ref.update(
        run_json_sha256=hash_file(root / "run.json")[1], bundle_sha256=digest(run_files)
    )
    baseline_ref = (
        {
            **baseline_identity("verified_local", baseline_build_ref, settings),
            "source_receipt_sha256": (
                hash_file(root / "baseline-receipt.json")[1]
                if run.get("baseline_receipt_path")
                else None
            ),
        }
        if verified
        else {
            "build_ref": baseline_build_ref,
            "local_inventory_sha256": digest(baseline_files),
            "provenance": "local_fixture_only",
        }
    )
    schema_version = 3 if verified else SCHEMA_VERSION
    model_ref = {
        "sha256": hash_file(model_path)[1],
        "settings": model,
        "route_id": getattr(adapter, "route_id", "unavailable"),
    }
    parallel_reads = model.get("judge_parallel_reads", True)
    material = judge_material(
        config, model, judge_model_budget_override=judge_model_budget_override
    )
    instruction, tools = material["instruction"], material["tools"]
    exposed_reads = {t["name"] for t in tools[:-2]}
    judge_pins = {
        "model_ref": model_ref,
        "provider_responses": [],
        "rubric_version": RUBRIC_VERSION,
        "anonymous_projection_version": ANONYMOUS_VERSION,
        "prompt_version": material["prompt_version"],
        "prompt_sha256": hashlib.sha256(instruction).hexdigest(),
        "tools_sha256": digest(tools),
        "input_sha256": digest(None),
        "limits_sha256": digest(limits),
        "runtime_sha256": runtime_sha256(),
    }
    override = bound_budget_override(
        judge_model_budget_override,
        anonymous_binding(run_files, package_files, config_path),
    )
    if override is not None:
        judge_pins["judge_model_budget_override"] = override
    delivery_protocol = judge_read_admission.protocol(adapter, model)
    judge_pins["read_result_admission"] = delivery_protocol
    pin_shell = {
        "schema_version": schema_version,
        "judgment_id": judgment_id,
        "task_ref": task_ref,
        "config_ref": task["meta"]["config_ref"],
        "run_ref": run_ref,
        "baseline_ref": baseline_ref,
        "judge": judge_pins,
    }
    identity = judgment_identity(pin_shell)
    if artifact_store is not None:
        marker = artifact_store.read_completion(unit_key)
        if marker is not None:
            check(
                marker["metadata"] == identity,
                "completed_identity_conflict",
                "Completed unit has a different pinned judgment identity",
            )
            return {"skipped": True, "completion": marker}
        if artifact_store.store.list_keys(
            artifact_store.bucket, f"{artifact_store.prefix}{unit_key}/"
        ):
            raise JudgeError(
                "partial_publication",
                "Unit has partial publication; retry the unchanged terminal bundle or choose a fresh unit",
            )
    check(not work.exists(), "unsafe_work", "work_dir must be fresh and absent")
    work.mkdir(parents=True)
    bundle = work / "bundle"
    bundle.mkdir()
    budget = WorkBudget(work, limits["judge_max_work_bytes"])
    # Reserve terminal metadata independently of the ordinary trace cap.
    terminal_reserve = (
        Trace.RESERVE + len(_encoded(run_files)) + len(_encoded(baseline_files)) + 65536
    )
    budget.reserve(terminal_reserve)
    trace = JudgeTrace(
        bundle / "trace.jsonl", limits["judge_max_trace_bytes"], started, budget
    )
    if hasattr(adapter, "serialize_request"):
        budget.write(bundle / "provider-native.jsonl", b"")
    sources = Sources(
        limits, total, sum(len(fs) for _, fs in inventories) + len(extra_files)
    )
    for path, raw in (
        ("instructions.txt", instruction),
        ("tools.json", _encoded(tools)),
        ("model.json", model_path.read_bytes()),
        ("run_files.json", _encoded(run_files)),
        ("baseline_files.json", _encoded(baseline_files)),
    ):
        budget.write(bundle / path, raw)
    judge_input = None
    worker, phase, error = None, "setup", None
    result = error_result("Judgment did not complete")
    model_used = read_used = setup_used = 0.0
    input_written = False
    receipt_bytes = 0

    def retain_receipt():
        nonlocal receipt_bytes
        if not (verified or notion_baseline_dir is not None) or worker is None:
            return
        receipt_path = worker.ipc / "baseline-receipt.json"
        if not receipt_path.exists():
            receipt_path = work / "baseline-receipt.pending"
            if not receipt_path.exists():
                return
        size_check(
            receipt_path.stat().st_size,
            limits["judge_max_source_file_bytes"],
            "source_file_limit",
        )
        raw = receipt_path.read_bytes()
        receipt = parse_json(raw)
        files = [
            {"path": r["path"], "bytes": r["size_bytes"], "sha256": r["sha256"]}
            for r in receipt["files"]
        ]
        previous = {r["path"]: r for r in baseline_files}
        added = [r for r in files if r["path"] not in previous]
        for record in files:
            check(
                record["path"] not in previous or previous[record["path"]] == record,
                "receipt_mismatch",
                "Worker receipt changed for a selected file",
            )
        new_bytes = sum(r["bytes"] for r in added) + len(raw) - receipt_bytes
        size_check(
            sources.total_bytes + new_bytes,
            limits["judge_max_source_bytes"],
            "source_limit",
        )
        size_check(
            sources.file_count + len(sources.entries) + len(added),
            limits["judge_max_source_count"],
            "source_count_limit",
        )
        sources.total_bytes += new_bytes
        sources.file_count += len(added)
        receipt_bytes = len(raw)
        baseline_files[:] = sorted(
            (
                files
                if verified
                else list({r["path"]: r for r in [*baseline_files, *files]}.values())
            ),
            key=lambda r: r["path"],
        )
        budget.write(bundle / "baseline-receipt.json", raw)
        budget.write(bundle / "baseline_files.json", _encoded(baseline_files))
        (work / "baseline-receipt.pending").unlink(missing_ok=True)

    def retain_anonymous_mapping():
        private = parse_json(prepared.audit_bytes)
        private.update(aliases=sources.projection.aliases, sources=sources.originals)
        pending = bundle / "anonymous-private.pending"
        budget.write(pending, _encoded(private))
        pending.chmod(0o600)
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        pending.replace(bundle / "anonymous-private.json")
        directory = os.open(bundle, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    try:
        prepared = anonymous_preparation
        if prepared is None:
            prepared = prepare_anonymous_answer(
                package=package, config_path=config_path, run_bundle=root
            )
        require_prepared(
            prepared, anonymous_binding(run_files, package_files, config_path)
        )
        sources = prepared.sources(
            limits, total, sum(len(fs) for _, fs in inventories) + len(extra_files)
        )
        judge_input = parse_json(prepared.input_bytes)
        delivered = judge_input["delivered_messages"]
        # Context comparison below uses the original, verified initial state.
        with (root / "trace.jsonl").open("rb") as original_trace:
            first = parse_json(
                original_trace.readline(limits["judge_max_source_file_bytes"] + 1)
            )
        recorded_context = first["payload"]["initial_input"]
        eligible = (
            run["outcome"] == "completed"
            or judge_input["attempt"]["model_budget_termination"]
        )
        retain_anonymous_mapping()
        raw_input = _encoded(judge_input)
        size_check(
            len(raw_input),
            limits["judge_initial_input_max_bytes"],
            "initial_input_limit",
        )
        budget.write(bundle / "input.json", raw_input)
        input_written = True
        judge_pins["input_sha256"] = digest(judge_input)
        trace.append("context", judge_input, native_guards=NATIVE_GUARDS)
        unsupported = {
            e["service"]
            for e in task["evidence"] + task["changes"]
            if e["service"] not in services.service_names()
        }
        unsupported.update(
            t for t in config["tool_scope"] if t not in services.read_names() | WRITES
        )
        unsupported.update(
            f["path"]
            for f in package_files
            if f["path"].startswith("delta/") and not services.supports_delta(f["path"])
        )
        check(
            not unsupported,
            "unsupported_service",
            "Local judge limitation: required service evidence, changes or tools outside registered services are unsupported",
        )
        source_ref = run.get("baseline_ref", {"mode": "local_declared"})
        check(
            source_ref["mode"] == baseline_mode,
            "baseline_mode_mismatch",
            "Judge/source attempt baseline modes differ",
        )
        if verified:
            check(
                source_ref
                == baseline_identity("verified_local", baseline_build_ref, settings),
                "pin_mismatch",
                "Judge/source attempt build or company bucket differs",
            )
            check(
                run.get("baseline_receipt_path") == "baseline-receipt.json",
                "missing_source",
                "Verified source attempt needs its original baseline receipt",
            )
        else:
            local_roots = {
                "linear": linear_baseline_dir,
                "notion": notion_baseline_dir,
                **service_baseline_dirs,
            }
            local_files = {
                "linear": linear_files,
                "notion": notion_files,
                **service_files,
            }
            for service in services.SERVICES:
                if service.NAME != services.CONTEXT_SERVICE:
                    service.validate_judge_local(
                        service in services.enabled_services(config["tool_scope"]),
                        local_roots.get(service.NAME),
                        local_files.get(service.NAME, []),
                        check,
                    )
        original_tools = [
            {"type": "function", "strict": False, **services.agent_definition(t)}
            for t in native_definitions(
                config["tool_scope"], run["resolved_limits"]["max_result_bytes"]
            )
        ] + [FINISH]
        check(
            original_tools == read_json(root / "tools.json")
            and digest(original_tools) == run["hashes"]["tool_definitions_sha256"],
            "context_mismatch",
            "Reconstructed original agent tools differ from the recorded definitions/hash",
        )
        if not eligible:
            raise JudgeError(
                "agent_terminal_failure",
                "Original attempt cannot be quality-scored: " + run["stop_reason"],
                cause={
                    "type": "AgentAttempt",
                    "code": run["stop_reason"],
                    "reason": "Recorded terminal failure; inspect the original error trace source",
                },
            )
        elapsed_setup = time.monotonic() - started
        remaining_setup = limits["judge_setup_timeout_seconds"] - elapsed_setup
        check(remaining_setup > 0, "setup_timeout", "Judge setup exceeded its deadline")
        worker = SlackWorker(
            {
                "package": package,
                "config_path": config_path,
                "baseline_dir": baseline_dir,
                "read_only": True,
                "baseline_build_ref": baseline_build_ref,
                "linear_baseline_dir": linear_baseline_dir,
                "notion_baseline_dir": notion_baseline_dir,
                "service_baseline_dirs": service_baseline_dirs,
                "work_dir": work / "environment",
                "initial_context_max_bytes": limits["judge_initial_input_max_bytes"],
                "max_result_bytes": limits["judge_max_result_bytes"],
                "recent_activity_window_minutes": run["resolved_limits"][
                    "recent_activity_window_minutes"
                ],
                "duckdb_memory_limit": limits["judge_duckdb_memory_limit"],
                "duckdb_threads": limits["judge_duckdb_threads"],
                "duckdb_max_temp_directory_size": limits[
                    "judge_duckdb_max_temp_directory_size"
                ],
                "judge_work_root": work,
                "judge_max_work_bytes": limits["judge_max_work_bytes"],
                "baseline_settings": settings,
                "judge_baseline_receipt": (
                    read_json(root / "baseline-receipt.json") if verified else None
                ),
                "judge_baseline_limits": (
                    {
                        "file_bytes": limits["judge_max_source_file_bytes"],
                        "bytes": limits["judge_max_source_bytes"] - sources.total_bytes,
                        "count": limits["judge_max_source_count"]
                        - sources.file_count
                        - len(sources.entries),
                    }
                    if verified or notion_baseline_dir is not None
                    else None
                ),
                "judge_local_files": (
                    {
                        r["path"]: dict(
                            path=r["path"], size_bytes=r["bytes"], sha256=r["sha256"]
                        )
                        for r in baseline_files
                    }
                    if not verified
                    else {}
                ),
                "judge_trace_sources": {
                    sid: [
                        str(target[0]),
                        target[1],
                        target[2],
                        sources.entries[sid]["sha256"],
                    ]
                    for sid, target in sources.bodies.items()
                    if sources.projection is None and not isinstance(target, bytes)
                },
            },
            environment_factory,
            baseline_store_factory,
        )
        worker.keep_baseline_receipt = verified or notion_baseline_dir is not None
        ready = await worker_operation(
            worker, lambda: worker.receive(remaining_setup), budget, remaining_setup
        )
        if verified:
            sources.total_bytes += ready["baseline_metadata_bytes"]
            sources.file_count += 2  # Bounded raw READY and manifest metadata.
        retain_receipt()
        setup_used = time.monotonic() - started
        context = ready["context"]
        check(
            context == recorded_context
            and digest(context) == run["hashes"]["input_sha256"],
            "context_mismatch",
            "Reconstructed initial Slack context differs from the original trace/hash",
        )
        check(
            ready["tools"]
            == [
                {k: v for k, v in t.items() if k not in {"type", "strict"}}
                for t in tools[:-2]
            ],
            "forbidden_write",
            "Read wrapper tool scope differs from the pinned definitions",
        )
        history = [
            {"role": "developer", "content": instruction.decode()},
            {"role": "user", "content": raw_input.decode()},
        ]
        if recoverable:
            from .runtime_identity import write_json

            write_json(
                work / "recovery.json",
                dict(shell={**pin_shell, "unit_key": unit_key}, resolved_limits=limits),
            )
        all_call_ids = set()
        submitted = False
        consistency_corrections = 0
        citation_corrections = 0
        for number in range(1, limits["judge_max_requests"] + 1):
            phase = "model"
            remaining = limits["judge_model_seconds"] - model_used
            check(
                remaining > 0,
                "judge_model_budget",
                "Judge model-response budget exhausted",
            )
            request_id = f"judge-request-{number}"
            request = {
                "model": model["model_id"],
                "input": history,
                "tools": tools,
                "store": False,
                "parallel_tool_calls": parallel_reads,
                "include": ["reasoning.encrypted_content"],
                "reasoning": {"effort": model["reasoning_effort"]},
                "max_output_tokens": model["max_output_tokens"],
            }
            size_check(
                len(_encoded(request)),
                limits["judge_max_model_input_bytes"],
                "model_input_limit",
            )
            if hasattr(adapter, "serialize_request"):
                native_request = adapter.serialize_request(request)
                size_check(
                    len(_encoded(native_request)),
                    limits["judge_max_model_input_bytes"],
                    "model_input_limit",
                )
                trace.append_native(
                    {
                        "kind": "request",
                        "request_id": request_id,
                        "route_id": adapter.route_id,
                        "payload": native_request,
                    }
                )
            budget.reserve(
                len(_encoded(request))
                + limits["judge_max_result_bytes"]
                + terminal_reserve
            )
            trace.append(
                "model_request",
                request,
                request_id=request_id,
                deadline_seconds=remaining,
            )
            provider_timing = {"model_seconds": 0.0}
            try:
                response = await model_response(
                    adapter, request, timeout_seconds=remaining, timing=provider_timing
                )
            finally:
                model_used += provider_timing["model_seconds"]
                trace.model_seconds = model_used
            body = provider_body(response.body, http_status=response.http_status)
            if getattr(response, "native_trace", None) is not None:
                trace.append_native(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        **{
                            k: v
                            for k, v in response.native_trace.items()
                            if k not in {"request", "request_id"}
                        },
                        "provider_request_id": safe_text(
                            response.request_id, limit=256
                        ),
                    }
                )
            # Never silently trim a model response to meet a trace/result cap.
            trace.append(
                "model_response",
                body,
                request_id=request_id,
                provider_request_id=safe_text(response.request_id, limit=256),
                http_status=response.http_status,
            )
            size_check(
                len(_encoded(body)),
                limits["judge_max_result_bytes"],
                "provider_result_limit",
            )
            judge_pins["provider_responses"].append(
                {
                    "request_id": safe_text(response.request_id, limit=256)
                    or "unavailable",
                    "response_id": body.get("id", "unavailable"),
                    "model": body.get("model", "unavailable"),
                    "version": body.get("model_version", "unavailable"),
                }
            )
            check(
                model_used < limits["judge_model_seconds"],
                "judge_model_budget",
                "Judge response reached the model deadline; its calls were not executed",
            )
            if (
                body.get("status") != "completed"
                or body.get("error")
                or (response.http_status and response.http_status >= 400)
                or not isinstance(body.get("output"), list)
            ):
                cause = provider_cause(
                    body,
                    http_status=response.http_status,
                    request_id=response.request_id,
                )
                raise JudgeError("provider_incomplete", cause["reason"], cause=cause)
            calls, ids = [], set()
            for item in body["output"]:
                check(
                    isinstance(item, dict)
                    and item.get("type") in {"message", "reasoning", "function_call"},
                    "unsupported_provider_action",
                    "Unsupported provider action; no calls executed",
                )
                check(
                    item.get("status", "completed") == "completed",
                    "provider_incomplete",
                    "Incomplete output item; no calls executed",
                )
                if item["type"] == "function_call":
                    cid = item.get("call_id")
                    check(
                        isinstance(cid, str)
                        and cid
                        and cid not in all_call_ids
                        and cid not in ids
                        and isinstance(item.get("name"), str),
                        "invalid_provider_call",
                        "Missing/duplicate call ID or invalid tool name",
                    )
                    calls.append(item)
                    ids.add(cid)
            all_call_ids.update(ids)
            history.extend(body["output"])
            if not calls:
                reminder = {"role": "user", "content": REMINDER}
                trace.append("reminder", reminder, request_id=request_id)
                history.append(reminder)
                continue
            mixed = (
                any(c["name"] == "submit_judgment" for c in calls) and len(calls) != 1
            )
            if not mixed:
                check(
                    all(
                        c["name"]
                        in exposed_reads | {"judge_read_source", "submit_judgment"}
                        for c in calls
                    ),
                    "forbidden_action",
                    "Judge attempted a write/control/unsupported action; no calls executed",
                )
            read_decline = (
                judge_read_group_decline(calls, tools)
                if parallel_reads and not mixed
                else None
            )
            recoverable_reads = (
                not mixed
                and read_decline is None
                and all(c["name"] != "submit_judgment" for c in calls)
            )
            if recoverable_reads:
                minimum = judge_read_admission.candidate(
                    model,
                    history,
                    tools,
                    parallel_reads,
                    calls[0]["call_id"],
                    judge_read_admission.ERROR,
                    calls[1:],
                )
                check(
                    read_admission.budget(
                        adapter,
                        minimum,
                        limits["judge_max_model_input_bytes"],
                        quota_limit=delivery_protocol["quota_limit"],
                    )["admissible"],
                    "judge_read_admission_base",
                    "The complete pending read receipts cannot fit the next request; no read bodies were delivered",
                )
            for call_index, call in enumerate(calls):
                withheld = None
                phase = "read" if call["name"] != "submit_judgment" else "submission"
                name, cid = call["name"], call["call_id"]
                trace.ensure(
                    len(_encoded(call)) + limits["judge_max_result_bytes"] + 4096
                )
                budget.reserve(
                    len(_encoded(call))
                    + limits["judge_max_result_bytes"] * 2
                    + terminal_reserve
                )
                trace.append("tool_request", call, request_id=request_id, call_id=cid)
                try:
                    arguments = parse_json(call.get("arguments"))
                    if not isinstance(arguments, dict):
                        raise ValueError("Not an object")
                except (ValueError, TypeError, UnicodeError):
                    arguments = None
                if mixed:
                    output = {
                        "ok": False,
                        "error": "not_executed_mixed_submission",
                        "message": load_text("judge/mixed_submission_reminder.txt"),
                    }
                elif read_decline:
                    output = read_decline
                elif name == "submit_judgment":
                    check(
                        arguments is not None,
                        "invalid_judgment",
                        "Submission is not a valid unique-key finite JSON object",
                    )
                    try:
                        result = validate_model_judgment(
                            arguments,
                            delivered=delivered,
                            sources=sources.entries,
                            accessible=sources.accessible,
                            assemble_evidence=True,
                        )
                    except JudgeError as exc:
                        if (
                            exc.code == "invalid_citation"
                            and not citation_corrections
                            and not consistency_corrections
                        ):
                            output = sources.submission_citation_feedback(
                                arguments,
                                delivered=delivered,
                                protocol=CITATION_FEEDBACK_VERSION,
                            )
                            if output is None:
                                raise
                            citation_corrections += 1
                        elif (
                            exc.code == "inconsistent_scores"
                            and not consistency_corrections
                            and not citation_corrections
                        ):
                            consistency_corrections += 1
                            output = score_consistency_feedback(exc)
                        else:
                            raise
                    else:
                        output, submitted = {
                            "ok": True,
                            "evidence_assembly": evidence_assembly(arguments, result),
                        }, True
                elif arguments is None:
                    output = {
                        "ok": False,
                        "error": "invalid_arguments",
                        "message": "Expected a unique-key finite JSON object",
                    }
                else:
                    read_start = time.monotonic()
                    try:
                        with sources.delivery() as commit_delivery:
                            if name == "judge_read_source":

                                def read_fragment(sid, offset, count):
                                    return worker.call(
                                        "_judge_read_fragment",
                                        {
                                            "source_id": sid,
                                            "offset": offset,
                                            "max_chars": count,
                                        },
                                        limits["judge_read_timeout_seconds"],
                                    )

                                output = await worker_operation(
                                    worker,
                                    lambda: sources.read(arguments, read_fragment),
                                    budget,
                                    limits["judge_read_timeout_seconds"],
                                )
                                native_arguments = arguments
                            else:
                                native_arguments = sources.projection.arguments(
                                    arguments, reverse=True
                                )
                                if verified or notion_baseline_dir is not None:
                                    worker.source_remaining = {
                                        "bytes": limits["judge_max_source_bytes"]
                                        - sources.total_bytes,
                                        "count": limits["judge_max_source_count"]
                                        - sources.file_count
                                        - len(sources.entries),
                                    }
                                output = await worker_operation(
                                    worker,
                                    lambda: worker.call(
                                        name,
                                        native_arguments,
                                        limits["judge_read_timeout_seconds"],
                                    ),
                                    budget,
                                    limits["judge_read_timeout_seconds"],
                                )
                            retain_receipt()
                            size_check(
                                len(_encoded(output)),
                                limits["judge_max_result_bytes"],
                                "result_limit",
                            )
                            original_output = output
                            if name != "judge_read_source":
                                output = sources.projection.result(output)
                            size_check(
                                len(_encoded(output)),
                                limits["judge_max_result_bytes"],
                                "result_limit",
                            )
                            native_output = output
                            native_seq = trace.seq + 1
                            try:
                                entries = (
                                    sources.native(
                                        output,
                                        cid,
                                        native_seq,
                                        name,
                                        original_result=original_output,
                                    )
                                    if name != "judge_read_source"
                                    else []
                                )
                            except BaseException:
                                # Preserve the original observable read on an
                                # indexing bound failure, without exposing IDs.
                                if services.records_native_trace(name):
                                    trace.append(
                                        "native_read",
                                        native_output,
                                        request_id=request_id,
                                        call_id=cid,
                                    )
                                raise
                            if entries:
                                output = {
                                    "result": output,
                                    "sources": native_source_view(entries),
                                }
                            size_check(
                                len(_encoded(output)),
                                limits["judge_max_result_bytes"],
                                "result_limit",
                            )
                            if recoverable_reads:
                                upcoming = judge_read_admission.candidate(
                                    model,
                                    history,
                                    tools,
                                    parallel_reads,
                                    cid,
                                    output,
                                    calls[call_index + 1 :],
                                )
                                decision = read_admission.budget(
                                    adapter,
                                    upcoming,
                                    limits["judge_max_model_input_bytes"],
                                    quota_limit=delivery_protocol["quota_limit"],
                                )
                                if not decision["admissible"]:
                                    actual_call = {
                                        **call,
                                        "arguments": _encoded(
                                            native_arguments
                                        ).decode(),
                                    }
                                    budget.reserve(
                                        len(_encoded(original_output))
                                        + 16384
                                        + terminal_reserve
                                    )
                                    withheld = read_admission.save(
                                        bundle,
                                        trace,
                                        actual_call,
                                        native_arguments,
                                        original_output,
                                        decision,
                                    )
                                    output = dict(judge_read_admission.ERROR)
                            if withheld is None:
                                if services.records_native_trace(name):
                                    trace.append(
                                        "native_read",
                                        native_output,
                                        request_id=request_id,
                                        call_id=cid,
                                    )
                                if name != "judge_read_source":
                                    retain_anonymous_mapping()
                                commit_delivery()
                    finally:
                        read_used += time.monotonic() - read_start
                trace.append(
                    "tool_result",
                    output,
                    request_id=request_id,
                    call_id=cid,
                    **(
                        {"judge_read_admission": withheld}
                        if withheld is not None
                        else {}
                    ),
                )
                if recoverable and submitted:
                    os.fsync(trace.file.fileno())
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": cid,
                        "output": _encoded(output).decode(),
                    }
                )
            if submitted:
                break
        check(
            submitted,
            "request_limit",
            "Judge reached judge_max_requests without submission",
        )
        if verified or notion_baseline_dir is not None:
            phase = "freeze"
            await worker_operation(
                worker,
                lambda: worker.call(
                    "_judge_verify_baseline", {}, limits["judge_read_timeout_seconds"]
                ),
                budget,
                limits["judge_read_timeout_seconds"],
            )
            retain_receipt()
        if result["status"] != "scored":
            error = {
                "phase": "submission",
                "code": "model_" + result["status"],
                "reason": result["reason"],
                "cause": None,
            }
    except (Exception, asyncio.CancelledError) as exc:
        trusted = isinstance(
            exc,
            (JudgeError, AttemptError, WorkerFailure, ContractError, RuntimeInputError),
        )
        cause = getattr(exc, "cause", None) or exception_cause(
            exc, trusted_reason=trusted
        )
        code = getattr(exc, "code", None) or (
            "judge_model_timeout"
            if isinstance(exc, TimeoutError)
            else (
                "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "unexpected_failure"
            )
        )
        reason = (
            safe_text(str(exc))
            if trusted
            else f"{phase} failed: {cause['type']} ({cause['code']}); details omitted"
        )
        error = {"phase": phase, "code": code, "reason": reason, "cause": cause}
        result = error_result(reason)
        trace.append("error", error, terminal=True)
    finally:
        if worker is not None:
            worker.close(graceful=error is None)
            try:
                retain_receipt()
            except Exception as exc:
                cause = exception_cause(exc, trusted_reason=isinstance(exc, JudgeError))
                error = {
                    "phase": "freeze",
                    "code": getattr(exc, "code", "receipt_failure"),
                    "reason": "Could not retain bounded baseline receipt",
                    "cause": cause,
                }
                result = error_result(error["reason"])
                trace.append("error", error, terminal=True)
        for name in ("environment", "worker-ipc"):
            shutil.rmtree(work / name, ignore_errors=True)
        if not setup_used:
            setup_used = time.monotonic() - started - model_used - read_used
    # The worker is stopped before rehashing and before freezing/publication.
    try:
        for source_root, before in inventories:
            current = (
                catalog_inventory(source_root, limits)
                if source_root == notion_baseline_dir
                else inventory(source_root, limits, exclude_marker=source_root == root)
            )
            check(
                current == before,
                "changed_input",
                "Immutable source inventory changed during judgment",
            )
        for entry in extra_files:
            check(
                hash_file(entry["path"]) == (entry["bytes"], entry["sha256"]),
                "changed_input",
                "Immutable config/model/prompt changed during judgment",
            )
        check(
            (
                hash_file(root / "_complete.json")
                if (root / "_complete.json").exists()
                else None
            )
            == marker_hash,
            "changed_input",
            "Original completion marker changed during judgment",
        )
    except Exception as exc:
        cause = exception_cause(
            exc, trusted_reason=isinstance(exc, (JudgeError, ContractError))
        )
        error = {
            "phase": "freeze",
            "code": "changed_input",
            "reason": "Immutable input changed before freezing",
            "cause": cause,
        }
        result = error_result(error["reason"])
        trace.append("error", error, terminal=True)
    if not input_written:
        # Failed setup has no constructed typed evidence input. Null is explicit;
        # no partially constructed evidence is claimed to have reached a model.
        budget.write(bundle / "input.json", _encoded(None))
    trace.append(
        "stop",
        {
            "status": result["status"],
            "code": error["code"] if error else "submit_judgment",
        },
        terminal=True,
    )
    trace.close()
    judgment: Judgment = {
        **result,
        "schema_version": schema_version,
        "judgment_id": judgment_id,
        "unit_key": unit_key,
        "task_ref": task_ref,
        "config_ref": task["meta"]["config_ref"],
        "run_ref": run_ref,
        "baseline_ref": baseline_ref,
        "judge": judge_pins,
        "resolved_limits": limits,
        "evidence": [sources.entries[sid] for sid in result["evidence"]],
        "error": error,
        "timings": {
            "model_seconds": model_used,
            "read_seconds": read_used,
            "setup_seconds": setup_used,
            "wall_seconds": time.monotonic() - started,
        },
        "trace_path": "trace.jsonl",
    }
    if sources.projection is not None:
        retain_anonymous_mapping()
    budget.write(bundle / "judgment.json", _encoded(judgment) + b"\n")
    if artifact_store is not None:
        check(
            judgment_identity(judgment) == identity,
            "pin_mismatch",
            "Terminal payload disagrees with preflight identity",
        )
        try:
            artifact_store.publish_directory(bundle, unit_key, identity)
        except Exception as exc:
            raise PublicationError(bundle, unit_key, _safe_reason(exc)) from None
    return {"skipped": False, "bundle": str(bundle), "judgment": judgment}
