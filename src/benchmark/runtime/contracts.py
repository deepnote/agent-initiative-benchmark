"""Version-one file contracts. These checks do not judge scenario quality."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An invalid file or field, ready for a CLI error message."""


def fail(file: str | Path, field: str, message: str) -> None:
    raise ContractError(f"{file}: {field}: {message}")


def object_fields(
    value: Any,
    keys: set[str],
    file: str | Path,
    field: str,
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict:
    if not isinstance(value, dict):
        fail(file, field, "expected an object")
    for key in sorted(keys - value.keys()):
        fail(file, f"{field}.{key}", "required field is missing")
    for key in sorted(value.keys() - keys - optional):
        fail(file, f"{field}.{key}", "unexpected field")
    return value


def string(value: Any, file: str | Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(file, field, "expected a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError:
        fail(file, field, "expected valid Unicode text")
    return value


def positive_int(value: Any, file: str | Path, field: str) -> int:
    if type(value) is not int or value < 1:
        fail(file, field, "expected a positive integer")
    return value


def schema_version(value: Any, file: str | Path, field: str) -> None:
    positive_int(value, file, field)
    if value != 1:
        fail(file, field, "unsupported schema version (supported: 1)")


def sha256(value: Any, file: str | Path, field: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        fail(file, field, "expected 64 lowercase SHA-256 hex characters")


def timestamp(value: Any, file: str | Path, field: str) -> datetime:
    string(value, file, field)
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        fail(file, field, "expected an ISO 8601 timestamp with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(file, field, "invalid ISO 8601 timestamp")
    if result.tzinfo is None or result.utcoffset() is None:
        fail(file, field, "timezone is required")
    return result


def array(value: Any, file: str | Path, field: str) -> list:
    if not isinstance(value, list):
        fail(file, field, "expected an array")
    return value


def native_object(value: Any, file: str | Path, field: str) -> None:
    if not isinstance(value, dict) or not value:
        fail(file, field, "expected a non-empty native service object")


def payload_path(value: Any, file: str | Path, field: str) -> str:
    """Require portable relative paths without normalizing away unsafe parts."""
    string(value, file, field)
    parts = value.split("/")
    if (
        "\\" in value
        or ":" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(p in {"", ".", ".."} for p in parts)
    ):
        fail(file, field, "unsafe relative path")
    if value != "task.json" and (len(parts) < 2 or parts[0] != "delta"):
        fail(file, field, "only task.json and delta/ payload files are allowed")
    return value


def build_ref(value: Any, file: str | Path, field: str) -> None:
    object_fields(value, {"build_id", "manifest_sha256"}, file, field)
    string(value["build_id"], file, f"{field}.build_id")
    sha256(value["manifest_sha256"], file, f"{field}.manifest_sha256")


def config_ref(value: Any, file: str | Path, field: str) -> None:
    object_fields(value, {"uri", "sha256"}, file, field)
    string(value["uri"], file, f"{field}.uri")
    sha256(value["sha256"], file, f"{field}.sha256")


def load_json(path: str | Path) -> Any:
    path = Path(path)

    class ObjectPairs(list):
        pass

    def unpack(value: Any, field: str = "$") -> Any:
        if isinstance(value, ObjectPairs):
            result = {}
            for key, child in value:
                if key in result:
                    fail(path, f"{field}.{key}", "duplicate JSON key")
                result[key] = unpack(child, f"{field}.{key}")
            return result
        if isinstance(value, list):
            return [unpack(child, f"{field}[{i}]") for i, child in enumerate(value)]
        return value

    def invalid_constant(value: str) -> None:
        fail(path, "$", f"invalid JSON constant: {value}")

    try:
        return unpack(
            json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=ObjectPairs,
                parse_constant=invalid_constant,
            )
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        fail(path, "$", str(exc))


def validate_config(value: Any, file: str | Path = "config.json") -> dict:
    object_fields(
        value,
        {
            "schema_version",
            "build_ref",
            "snapshot_time",
            "agent_instructions",
            "tool_scope",
            "judge_settings",
            "limits",
            "storage",
        },
        file,
        "$",
        {"service_settings"},
    )
    schema_version(value["schema_version"], file, "$.schema_version")
    build_ref(value["build_ref"], file, "$.build_ref")
    timestamp(value["snapshot_time"], file, "$.snapshot_time")
    string(value["agent_instructions"], file, "$.agent_instructions")
    scope = array(value["tool_scope"], file, "$.tool_scope")
    for i, tool in enumerate(scope):
        string(tool, file, f"$.tool_scope[{i}]")
        if tool in scope[:i]:
            fail(file, f"$.tool_scope[{i}]", "duplicate tool name")
    for key in (
        "judge_settings",
        "limits",
        "storage",
    ):
        if not isinstance(value[key], dict):
            fail(file, f"$.{key}", "expected an object")
    if "service_settings" in value and not isinstance(value["service_settings"], dict):
        fail(file, "$.service_settings", "expected service objects")
    from .hubspot_environment import settings

    settings(value)
    return value


def validate_brief(value: Any, file: str | Path, field: str = "$") -> dict:
    object_fields(
        value, {"intent", "situation", "ideal_action"}, file, field, {"task_id"}
    )
    if value["intent"] not in ("action", "silence"):
        fail(file, f"{field}.intent", "expected action or silence")
    string(value["situation"], file, f"{field}.situation")
    if value["intent"] == "silence":
        if value["ideal_action"] is not None:
            fail(file, f"{field}.ideal_action", "must be null for silence")
    else:
        string(value["ideal_action"], file, f"{field}.ideal_action")
    if "task_id" in value:
        string(value["task_id"], file, f"{field}.task_id")
    return value


def validate_tasks(value: Any, file: str | Path = "tasks.json") -> list:
    array(value, file, "$")
    seen = set()
    for i, brief in enumerate(value):
        validate_brief(brief, file, f"$[{i}]")
        if "task_id" in brief:
            if brief["task_id"] in seen:
                fail(file, f"$[{i}].task_id", "duplicate supplied task ID")
            seen.add(brief["task_id"])
    return value


def source(value: Any, file: str | Path, field: str, *, query: bool = True) -> None:
    if not isinstance(value, dict) or len(value) != 1:
        fail(file, field, "expected exactly one record or query object")
    kind = next(iter(value))
    if kind not in ({"record", "query"} if query else {"record"}):
        fail(file, field, "expected record" + (" or query" if query else ""))
    native_object(value[kind], file, f"{field}.{kind}")


def validate_task(value: Any, file: str | Path = "task.json") -> dict:
    object_fields(
        value,
        {"meta", "brief", "scenario", "reference", "evidence", "changes", "notes"},
        file,
        "$",
    )
    meta = object_fields(
        value["meta"],
        {
            "task_id",
            "version",
            "schema_version",
            "build_ref",
            "snapshot_time",
            "config_ref",
        },
        file,
        "$.meta",
    )
    string(meta["task_id"], file, "$.meta.task_id")
    positive_int(meta["version"], file, "$.meta.version")
    schema_version(meta["schema_version"], file, "$.meta.schema_version")
    build_ref(meta["build_ref"], file, "$.meta.build_ref")
    config_ref(meta["config_ref"], file, "$.meta.config_ref")
    timestamp(meta["snapshot_time"], file, "$.meta.snapshot_time")
    brief = validate_brief(value["brief"], file, "$.brief")
    if "task_id" in brief and brief["task_id"] != meta["task_id"]:
        fail(file, "$.brief.task_id", "must match meta.task_id")
    string(value["scenario"], file, "$.scenario")
    string(value["notes"], file, "$.notes")
    ref = object_fields(
        value["reference"],
        {"intent", "reason", "example_messages", "important_mistakes"},
        file,
        "$.reference",
    )
    if ref["intent"] != brief["intent"]:
        fail(file, "$.reference.intent", "must match brief.intent")
    string(ref["reason"], file, "$.reference.reason")
    messages = array(ref["example_messages"], file, "$.reference.example_messages")
    if ref["intent"] == "silence" and messages:
        fail(file, "$.reference.example_messages", "must be empty for silence")
    for i, message in enumerate(messages):
        field = f"$.reference.example_messages[{i}]"
        object_fields(message, {"text", "destination"}, file, field)
        string(message["text"], file, f"{field}.text")
        dest = message["destination"]
        if not isinstance(dest, dict) or dest.get("kind") not in (
            "channel",
            "thread",
            "dm",
        ):
            fail(file, f"{field}.destination.kind", "expected channel, thread, or dm")
        keys = {
            "channel": {"channel_id"},
            "thread": {"channel_id", "thread_ts"},
            "dm": {"user_id"},
        }[dest["kind"]]
        object_fields(dest, keys | {"kind"}, file, f"{field}.destination")
        for key in keys:
            string(dest[key], file, f"{field}.destination.{key}")
    for i, mistake in enumerate(
        array(ref["important_mistakes"], file, "$.reference.important_mistakes")
    ):
        string(mistake, file, f"$.reference.important_mistakes[{i}]")
    for i, evidence in enumerate(array(value["evidence"], file, "$.evidence")):
        field = f"$.evidence[{i}]"
        object_fields(evidence, {"service", "source", "why"}, file, field, {"origin"})
        string(evidence["service"], file, f"{field}.service")
        source(evidence["source"], file, f"{field}.source")
        string(evidence["why"], file, f"{field}.why")
        if "origin" in evidence and evidence["origin"] not in ("baseline", "delta"):
            fail(file, f"{field}.origin", "expected baseline or delta")
    for i, change in enumerate(array(value["changes"], file, "$.changes")):
        field = f"$.changes[{i}]"
        object_fields(
            change, {"service", "operation", "source", "file", "reason"}, file, field
        )
        string(change["service"], file, f"{field}.service")
        if change["operation"] not in ("add", "replace"):
            fail(file, f"{field}.operation", "expected add or replace")
        source(change["source"], file, f"{field}.source", query=False)
        payload_path(change["file"], file, f"{field}.file")
        if not change["file"].startswith("delta/"):
            fail(file, f"{field}.file", "must name a delta/ file")
        string(change["reason"], file, f"{field}.reason")
    return value


def load_config(path: str | Path) -> dict:
    return validate_config(load_json(path), path)


def load_tasks(path: str | Path) -> list:
    return validate_tasks(load_json(path), path)


def load_task(path: str | Path) -> dict:
    return validate_task(load_json(path), path)
