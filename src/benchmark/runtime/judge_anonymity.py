"""Deterministic judge views. Original records and mappings never enter the view."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .judge_contracts import JudgeError, parse_json
from .slack_environment import _encoded

VERSION = "judge-anonymous-v2"


def analytics_report_metadata(value):
    """Keep native report qualifications without transport or author fields."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if key in {"dataLossFromOtherRow", "exact", "subjectToThresholding"}:
            if type(item) is bool:
                result[key] = item
        elif (
            key == "currencyCode"
            and isinstance(item, str)
            and re.fullmatch(r"[A-Z]{3}", item)
        ):
            result[key] = item
        elif key == "grain" and item in ("sessions", "events"):
            result[key] = item
        elif key == "timeZone" and isinstance(item, str):
            try:
                ZoneInfo(item)
            except (ValueError, ZoneInfoNotFoundError):
                continue
            result[key] = item
        elif key == "snapshotTime" and isinstance(item, str):
            try:
                parsed = datetime.fromisoformat(item)
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                result[key] = item
    return result


# These are harness fields, not words to replace in business prose.
TECHNICAL = frozenset(
    {
        "model",
        "model_id",
        "model_ref",
        "model_version",
        "provider",
        "provider_id",
        "provider_responses",
        "provider_request_id",
        "deployment",
        "route_id",
        "reasoning",
        "reasoning_effort",
        "effort",
        "usage",
        "system_fingerprint",
        "encrypted_content",
        "request_id",
        "response_id",
        "attempt_id",
        "run_id",
        "unit_key",
        "runtime_sha256",
        "headers",
        "stack",
        "stacktrace",
        "traceback",
        "model_name",
        "provider_name",
        "author_model",
        "author",
        "run_label",
        "attempt_label",
        "trace",
        "debug",
        "diagnostics",
        "execution_metadata",
    }
)
PATHS = frozenset(
    {"path", "file", "file_path", "source_path", "trace_path", "uri", "document_path"}
)
HANDLE_FIELDS = PATHS | {
    "id",
    "document_id",
    "file_id",
    "source_id",
    "filename",
    "contract_document_path",
}
LOCATOR_FIELDS = frozenset(
    {
        "seq",
        "judge_trace_seq",
        "event_kind",
        "tool_name",
        "index",
        "channel_id",
        "ts",
        "call_id",
        "json_pointer",
        "service",
        "record",
        "kind",
        "operation",
        "http_status",
    }
)
# Stable native error enums are behavior evidence. Free-form diagnostic strings
# are private, including any provider-specific error code.
NATIVE_ERRORS = frozenset(
    {
        "not_found",
        "access_denied",
        "forbidden",
        "forbidden_write",
        "unknown_tool",
        "invalid_arguments",
        "invalid_query",
        "invalid_request",
        "result_too_large",
        "result_limit",
        "query_timeout",
        "query_resource_limit",
        "not_linked",
        "wrong_parent",
        "stale_cursor",
        "invalid_cursor",
        "read_result_admission",
        "channel_not_found",
        "user_not_found",
        "message_not_found",
    }
)


def path_field(key, value):
    return isinstance(value, str) and (
        key in PATHS
        or key == "contract_document_path"
        or key in {"id", "document_id", "file_id", "source_id", "filename"}
        and (
            "/" in value
            or "\\" in value
            or value.endswith((".pdf", ".md", ".jsonl", ".sqlite"))
        )
    )


def private_restore_patch(original, shown, pointer=()):
    """Store only changed technical fields, without copying business rows."""
    if isinstance(original, dict) and isinstance(shown, dict):
        result = []
        for key in sorted(original.keys() | shown.keys()):
            if key not in shown:
                result.append([list(pointer + (key,)), True, original[key]])
            elif key not in original:
                result.append([list(pointer + (key,)), False, None])
            else:
                result.extend(
                    private_restore_patch(original[key], shown[key], pointer + (key,))
                )
        if list(original) != list(shown):
            result.append([list(pointer), "order", list(original)])
        return result
    if (
        isinstance(original, list)
        and isinstance(shown, list)
        and len(original) == len(shown)
    ):
        return [
            item
            for index, (a, b) in enumerate(zip(original, shown))
            for item in private_restore_patch(a, b, pointer + (index,))
        ]
    return [] if original == shown else [[list(pointer), True, original]]


def restore_original(shown, patch):
    result = copy.deepcopy(shown)
    for path, present, value in patch:
        if present == "order":
            target = result
            for key in path:
                target = target[key]
            ordered = {key: target[key] for key in value}
            if not path:
                result = ordered
            else:
                parent = result
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = ordered
            continue
        if not path:
            result = copy.deepcopy(value)
            continue
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        if present:
            parent[path[-1]] = copy.deepcopy(value)
        else:
            del parent[path[-1]]
    return result


class Projection:
    """Apply schema rules to technical envelopes. Business strings stay exact."""

    def __init__(self, aliases=None):
        self.aliases = dict(aliases or {})

    def alias(self, kind, original):
        key = kind + ":" + str(original)
        if key not in self.aliases:
            number = 1 + sum(k.startswith(kind + ":") for k in self.aliases)
            self.aliases[key] = f"{kind}-{number:04d}"
        return self.aliases[key]

    def arguments(self, value, *, reverse=False):
        """Only path handles change in tool arguments, never message text or SQL."""
        if isinstance(value, list):
            return [self.arguments(item, reverse=reverse) for item in value]
        back = {
            v: k.split(":", 1)[1]
            for k, v in self.aliases.items()
            if k.startswith("path:")
        }
        if not isinstance(value, dict):
            return value
        return {
            key: (
                (back.get(item, item) if reverse else self.alias("path", item))
                if (reverse and key in HANDLE_FIELDS and isinstance(item, str))
                or (not reverse and path_field(key, item))
                else self.arguments(item, reverse=reverse)
            )
            for key, item in value.items()
        }

    def metadata(self, value):
        if isinstance(value, list):
            return [self.metadata(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {
            key: (
                self.arguments(item)
                if key == "record"
                else (
                    self.alias("path", item)
                    if path_field(key, item)
                    else (
                        self.alias("call", item)
                        if key == "call_id"
                        else self.metadata(item)
                    )
                )
            )
            for key, item in value.items()
            if key in LOCATOR_FIELDS | PATHS
        }

    def task(self, task):
        result = copy.deepcopy(task)
        result["meta"] = {
            key: value
            for key, value in task["meta"].items()
            if key in {"schema_version", "version", "snapshot_time"}
        }
        result["meta"]["task_id"] = "task-0001"
        result["brief"]["task_id"] = "task-0001"
        for item in result.get("changes", []):
            if "file" in item:
                item["file"] = self.alias("path", item["file"])
        for item in result.get("evidence", []) + result.get("changes", []):
            # The native record selector remains usable. Only a source file
            # locator is a harness path.
            source = item.get("source", {})
            if "record" in source:
                source["record"] = self.arguments(source["record"])
            for key in sorted(PATHS & source.keys()):
                if isinstance(source[key], str):
                    source[key] = self.alias("path", source[key])
        return result

    def result(self, value):
        """Project service envelopes, preserving complete native business records."""
        if not isinstance(value, dict):
            return value
        if value.get("ok") is False or value.get("error") is not None:
            from .read_result_admission import ERROR as read_admission_error

            if value == read_admission_error:
                # This exact harness receipt explains which body the agent did
                # not see. It contains no provider or private diagnostics.
                return dict(value)
            # Free-form diagnostics may contain a deployment, request ID, path
            # or stack. Preserve failure and HTTP status, keep details private.
            return {
                "ok": False,
                "error": (
                    value["error"]
                    if isinstance(value.get("error"), str)
                    and value["error"] in NATIVE_ERRORS
                    else "operation_failed"
                ),
                "message": "The operation failed. Original diagnostics are private.",
                **(
                    {"http_status": value["http_status"]}
                    if type(value.get("http_status")) is int
                    else {}
                ),
            }
        result = {}
        for key, item in value.items():
            if key in TECHNICAL:
                continue
            if path_field(key, item):
                result[key] = self.alias("path", item)
            elif key in {"_meta", "metadata", "provenance", "receipt", "source"}:
                # These are transport envelopes at the result root. Native
                # report qualifications and metadata inside records are evidence.
                result[key] = (
                    analytics_report_metadata(item)
                    if key == "metadata"
                    and value.get("kind") == "analyticsData#runReport"
                    else {}
                )
            elif key == "result" and isinstance(item, dict):
                result[key] = self.result(item)
            elif key == "results" and isinstance(item, list):
                result[key] = [self.business_record(row) for row in item]
            else:
                # text, rows, records, messages and document content are evidence.
                result[key] = self.business_record(item)
        return result

    def business_record(self, value):
        if isinstance(value, list):
            return [self.business_record(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        # A native business row can legitimately have a model or provider field.
        # Only explicit harness metadata and file handles are projected here.
        return {
            key: (
                {}
                if key == "_meta"
                else (
                    self.alias("path", item)
                    if path_field(key, item)
                    else self.business_record(item)
                )
            )
            for key, item in value.items()
        }

    def call(self, call):
        result = {key: call[key] for key in ("type", "name", "status") if key in call}
        if "call_id" in call:
            result["call_id"] = self.alias("call", call["call_id"])
        if "arguments" in call:
            try:
                args = parse_json(call["arguments"])
                result["arguments"] = _encoded(self.arguments(args)).decode()
            except (TypeError, ValueError, UnicodeError):
                result["arguments"] = "Invalid arguments"
        return result

    def trace(self, event):
        kind, body = event["kind"], event["payload"]
        if kind == "context":
            payload = {
                "instruction": body["instruction"],
                "tools": body["tools"],
                "initial_input": self.result(body["initial_input"]),
            }
        elif kind == "model_request":
            # Histories duplicate other sources and contain provider continuation
            # tokens. Every tool action/result has its own complete source.
            payload = {"operation": "request"}
        elif kind == "model_response":
            payload = {"status": body.get("status"), "output": []}
            for item in body.get("output", []):
                if isinstance(item, dict) and item.get("type") == "function_call":
                    payload["output"].append(self.call(item))
                elif isinstance(item, dict) and item.get("type") == "message":
                    payload["output"].append(
                        {
                            key: copy.deepcopy(item[key])
                            for key in ("type", "role", "status", "content")
                            if key in item
                        }
                    )
        elif kind == "tool_request":
            payload = self.call(body)
        elif kind == "tool_result":
            payload = self.result(body)
        elif kind == "error":
            payload = {"phase": body.get("phase"), "failed": True}
        elif kind == "stop":
            payload = {
                "outcome": body["outcome"],
                "stop_reason": self.stop_reason(body["stop_reason"]),
            }
        else:
            payload = {key: body[key] for key in ("role", "content") if key in body}
        return {"seq": event["seq"], "kind": kind, "payload": payload}

    def source(self, sid, kind, locator, body):
        if kind == "trace":
            body = self.trace(body)
        elif sid == "artifact:model.json":
            body = {"configuration": "not included in the judge view"}
        elif kind == "task":
            body = self.task(body)
        elif kind == "message":
            body = {**body, "call_id": self.alias("call", body["call_id"])}
        return (
            self.metadata({k: v for k, v in locator.items() if k != "record_bytes"}),
            body,
        )

    def initial(self, inp, entries):
        return {
            **inp,
            "task_ref": {"task_id": "task-0001", "version": inp["task_ref"]["version"]},
            "config_ref": {"id": "policy-0001"},
            "run_ref": {"answer_id": "answer-0001"},
            "baseline_ref": {"id": "dataset-0001"},
            "task": self.task(inp["task"]),
            "attempt": {
                **inp["attempt"],
                "stop_reason": self.stop_reason(inp["attempt"]["stop_reason"]),
            },
            "initial_context": self.result(inp["initial_context"]),
            "delivered_messages": [
                {**m, "call_id": self.alias("call", m["call_id"])}
                for m in inp["delivered_messages"]
            ],
            "sources": entries,
        }

    @staticmethod
    def stop_reason(value):
        return (
            value
            if value in {"finish_task", "model_budget", "model_timeout"}
            else "operation_failed"
        )


@dataclass(frozen=True)
class PreparedAnswer:
    """Immutable shared data. Paging, selected evidence and judge state are absent.

    Trace targets are references to verified original bytes, not dataset copies.
    This object and its audit bytes are private and must never be given to a model.
    """

    input_bytes: bytes
    index_bytes: bytes
    targets: tuple
    audit_bytes: bytes
    binding: str

    @property
    def reference(self):
        return {
            "version": VERSION,
            "input_sha256": hashlib.sha256(self.input_bytes).hexdigest(),
        }

    def write(self, path):
        """Write a private, immutable preparation manifest. Return its checksum."""
        data = dict(
            version=VERSION,
            input=parse_json(self.input_bytes),
            index=parse_json(self.index_bytes),
            audit=parse_json(self.audit_bytes),
            binding=self.binding,
            targets=[],
        )
        for sid, target in self.targets:
            data["targets"].append(
                [
                    sid,
                    (
                        {"text": target.decode("utf-8")}
                        if isinstance(target, bytes)
                        else {
                            "trace": str(target[0]),
                            "offset": target[1],
                            "length": target[2],
                        }
                    ),
                ]
            )
        raw = _encoded(data)
        path = Path(path)
        # Exclusive creation never overwrites a historical preparation.
        import os

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o400)
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def read(cls, path, *, sha256):
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 268435456:
            raise JudgeError(
                "anonymous_input_conflict", "Unsafe anonymous preparation manifest"
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise JudgeError(
                "anonymous_input_conflict",
                "Anonymous preparation manifest checksum differs",
            )
        data = parse_json(raw)
        if data.get("version") != VERSION:
            raise JudgeError(
                "anonymous_input_conflict", "Unsupported anonymous preparation version"
            )
        targets = tuple(
            (
                sid,
                (
                    item["text"].encode("utf-8")
                    if "text" in item
                    else (Path(item["trace"]), item["offset"], item["length"])
                ),
            )
            for sid, item in data["targets"]
        )
        return cls(
            _encoded(data["input"]),
            _encoded(data["index"]),
            targets,
            _encoded(data["audit"]),
            data["binding"],
        )

    def sources(self, limits, total_bytes=0, file_count=0):
        from .judge_runner import Sources

        audit = parse_json(self.audit_bytes)
        result = Sources(
            limits, total_bytes, file_count, projection=Projection(audit["aliases"])
        )
        data = parse_json(self.index_bytes)
        result.entries = {item["id"]: item for item in data["entries"]}
        result.bodies = dict(self.targets)
        result.originals = audit["sources"]
        result.accessible = set(data["accessible"])
        result.current_protocol = data["current_protocol"]
        result.total_bytes += data["total_bytes"]
        if (
            result.total_bytes > limits["judge_max_source_bytes"]
            or len(result.entries) + file_count > limits["judge_max_source_count"]
        ):
            raise JudgeError(
                "source_limit", "Anonymous preparation exceeds source limits"
            )
        return result


def freeze(inp, sources, binding):
    view = sources.projection.initial(inp, list(sources.entries.values()))
    return PreparedAnswer(
        _encoded(view),
        _encoded(
            dict(
                entries=list(sources.entries.values()),
                accessible=sorted(sources.accessible),
                current_protocol=sources.current_protocol,
                total_bytes=sources.total_bytes,
            )
        ),
        tuple(sources.bodies.items()),
        _encoded(
            dict(
                version=VERSION,
                binding=binding,
                original_refs={
                    key: inp[key]
                    for key in ("task_ref", "config_ref", "run_ref", "baseline_ref")
                },
                aliases=sources.projection.aliases,
                sources=sources.originals,
            )
        ),
        binding,
    )


def require_prepared(prepared, binding):
    if not isinstance(prepared, PreparedAnswer) or prepared.binding != binding:
        raise JudgeError(
            "anonymous_input_conflict",
            "Anonymous preparation does not match these frozen inputs",
        )
