"""One isolated Slack attempt. Private task fields never enter tool results."""

from __future__ import annotations

from benchmark.prompts import load_text

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import build_ref, load_config, load_json, load_task, timestamp
from .packages import verify_package
from benchmark.tools.slack.slack import (
    SlackConfig,
    SlackToolError,
    SlackTools,
    OVERLAY_MESSAGE_COLUMNS,
    OVERLAY_REACTION_COLUMNS,
    build_slack_overlay_from_seed,
    _micros_to_timestamp,
)


class SlackEnvironmentError(ValueError):
    """Invalid native data or an unusable initial context, with owner causes."""

    def __init__(
        self,
        message,
        *,
        code="invalid_slack_environment",
        record=None,
        source=None,
        validation_reason=None,
    ):
        super().__init__(message)
        self.code, self.record, self.source = code, record, source
        self.validation_reason = validation_reason


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


# This is the sole dispatch allowlist, also used for argument checking/schema.
_PAGE = {"page_size": "integer", "start_cursor": "cursor"}
_NATIVE_ID = re.compile(r"[A-Z][A-Za-z0-9_]{0,127}")
_SLACK_TS = r"(0|[1-9][0-9]{0,11})\.[0-9]{6}"
_TOOLS = {
    "slack_list_channels": (
        {},
        {**_PAGE, "types": "types", "exclude_archived": "boolean"},
        load_text("tools/slack/slack_list_channels.txt"),
    ),
    "slack_search_messages": (
        {"query": "string"},
        {**_PAGE, "sort": "string", "sort_dir": "string"},
        load_text("tools/slack/slack_search_messages.txt"),
    ),
    "slack_get_thread": (
        {"channel_id": "string", "thread_ts": "string"},
        _PAGE,
        load_text("tools/slack/slack_get_thread.txt"),
    ),
    "slack_post_message": (
        {"channel_id": "string", "text": "string"},
        {},
        load_text("tools/slack/slack_post_message.txt"),
    ),
    "slack_reply_to_thread": (
        {"channel_id": "string", "thread_ts": "string", "text": "string"},
        {},
        load_text("tools/slack/slack_reply_to_thread.txt"),
    ),
    "slack_list_users": ({}, _PAGE, load_text("tools/slack/slack_list_users.txt")),
    "slack_send_dm": (
        {"user_id": "string", "text": "string"},
        {},
        load_text("tools/slack/slack_send_dm.txt"),
    ),
}


def _schema(kind: str) -> dict:
    if kind == "cursor":
        return {"type": ["string", "null"]}
    if kind == "types":
        return {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        }
    result = {"type": kind}
    if kind == "integer":
        result["minimum"] = 1
    return result


def tool_definitions(scope, max_result_bytes, max_page_size=100):
    return [
        {
            "name": name,
            "description": description
            + load_text("tools/slack/result_cap.txt").format(
                max_result_bytes=max_result_bytes
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": list(required),
                "properties": {
                    key: _parameter(key, kind, name, max_page_size)
                    for key, kind in {**required, **optional}.items()
                },
            },
        }
        for name, (required, optional, description) in _TOOLS.items()
        if name in scope
    ]


def _parameter(key: str, kind: str, name: str, max_page_size: int) -> dict:
    schema = _schema(kind)
    descriptions = {
        "start_cursor": load_text("tools/slack/parameters/start_cursor.txt"),
        "channel_id": load_text("tools/slack/parameters/channel_id.txt"),
        "user_id": load_text("tools/slack/parameters/user_id.txt"),
        "thread_ts": load_text("tools/slack/parameters/thread_ts.txt"),
        "text": load_text("tools/slack/parameters/text.txt"),
        "query": load_text("tools/slack/parameters/query.txt"),
        "types": load_text("tools/slack/parameters/types.txt"),
        "exclude_archived": load_text("tools/slack/parameters/exclude_archived.txt"),
        "sort": load_text("tools/slack/parameters/sort.txt"),
        "sort_dir": load_text("tools/slack/parameters/sort_dir.txt"),
    }
    if key != "page_size":
        schema["description"] = descriptions[key]
    if key == "page_size":
        default = min(max_page_size, 20 if name == "slack_search_messages" else 100)
        schema["description"] = load_text(
            "tools/slack/parameters/page_size.txt"
        ).format(max_page_size=max_page_size, default=default)
        schema.update(maximum=max_page_size, default=default)
    if key in {"sort", "sort_dir"}:
        schema["enum"] = ["score", "timestamp"] if key == "sort" else ["desc", "asc"]
    if key == "thread_ts":
        schema["pattern"] = _SLACK_TS
    return schema


class SlackEnvironment:
    """Verify a frozen package, compile its small delta, expose bounded tools.

    baseline_build_ref is the caller's declaration, not proof of baseline bytes.
    work_dir must not exist. No baseline or package files are copied or written.
    """

    def __init__(
        self,
        package: str | Path,
        config_path: str | Path,
        baseline_dir: str | Path,
        baseline_build_ref: dict,
        work_dir: str | Path,
        *,
        initial_context_max_bytes: int,
        max_result_bytes: int = 1_048_576,
        recent_activity_window_minutes: int = 10,
        max_delta_bytes: int = 8_388_608,
        max_page_size: int = 100,
        duckdb_memory_limit: str = "256MiB",
        duckdb_threads: int = 2,
        duckdb_max_temp_directory_size: str = "4GiB",
        max_related_rows: int = 10_000,
        agent_user_id: str = "U_AGENT",
    ):
        self._tools = None
        self._closed = False
        for name, value in {
            "initial_context_max_bytes": initial_context_max_bytes,
            "max_result_bytes": max_result_bytes,
            "recent_activity_window_minutes": recent_activity_window_minutes,
            "max_delta_bytes": max_delta_bytes,
            "max_page_size": max_page_size,
            "duckdb_threads": duckdb_threads,
            "max_related_rows": max_related_rows,
        }.items():
            if type(value) is not int or value < 1:
                raise SlackEnvironmentError(f"{name}: expected a positive integer")
        if max_result_bytes < 512:
            raise SlackEnvironmentError(
                "max_result_bytes: must be at least 512 for delivery receipts"
            )
        if not isinstance(agent_user_id, str) or not _NATIVE_ID.fullmatch(
            agent_user_id
        ):
            raise SlackEnvironmentError("agent_user_id: invalid native ID")
        package, baseline_dir, work_dir = map(Path, (package, baseline_dir, work_dir))
        package_input = package
        package, baseline_dir, work_dir = (
            p.resolve() for p in (package, baseline_dir, work_dir)
        )
        for immutable in (package, baseline_dir, Path(config_path).resolve()):
            if work_dir.is_relative_to(immutable) or immutable.is_relative_to(work_dir):
                raise SlackEnvironmentError(
                    "work_dir must be separate from immutable inputs"
                )
        if work_dir.exists():
            raise SlackEnvironmentError("work_dir must be a fresh, absent directory")
        manifest = verify_package(package_input, config_path=config_path)
        config = load_config(config_path)
        build_ref(baseline_build_ref, "baseline_build_ref", "$")
        if baseline_build_ref != config["build_ref"]:
            raise SlackEnvironmentError(
                "baseline_build_ref does not match the pinned configuration"
            )
        task = load_task(package / "task.json")
        self._scope = frozenset(config["tool_scope"]) & _TOOLS.keys()
        self._snapshot_time = config["snapshot_time"]
        instant = timestamp(
            self._snapshot_time, config_path, "$.snapshot_time"
        ).astimezone(timezone.utc)
        elapsed = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
        self._snapshot_micros = (
            elapsed.days * 86400 + elapsed.seconds
        ) * 1_000_000 + elapsed.microseconds
        if self._snapshot_micros < 0:
            raise SlackEnvironmentError(
                "snapshot_time must be on or after the Unix epoch"
            )
        self._window_minutes = recent_activity_window_minutes
        self._context_limit = initial_context_max_bytes
        self._result_limit = max_result_bytes
        self._page_size = max_page_size
        delta = self._read_delta(package, manifest, task, max_delta_bytes)
        work_dir.mkdir(parents=True, exist_ok=False)
        self._work_dir = work_dir
        overlay = work_dir / "slack"
        overlay.mkdir()
        seed = overlay / "seed.json"
        seed.write_bytes(_encoded(delta))
        build_slack_overlay_from_seed(seed, overlay / "overlay.sqlite")
        try:
            self._tools = SlackTools(
                SlackConfig(
                    baseline_dir=baseline_dir,
                    overlay_dir=overlay,
                    agent_user_id=agent_user_id,
                    benchmark_mode=True,
                    max_page_size=max_page_size,
                    scratch_dir=work_dir / "scratch",
                    duckdb_memory_limit=duckdb_memory_limit,
                    duckdb_threads=duckdb_threads,
                    duckdb_max_temp_directory_size=duckdb_max_temp_directory_size,
                    max_related_rows=max_related_rows,
                ),
                clock_ns=lambda: (self._snapshot_micros + 1) * 1000,
            )
            self._validate_native(task, delta)
        except BaseException:
            self.close()
            raise

    @property
    def tool_definitions(self) -> list[dict]:
        return tool_definitions(self._scope, self._result_limit, self._page_size)

    def _read_delta(
        self, package: Path, manifest: dict, task: dict, limit: int
    ) -> dict:
        slack_files = [
            f for f in manifest["files"] if f["path"].startswith("delta/slack/")
        ]
        for entry in slack_files:
            if entry["path"] != "delta/slack/seed.json":
                raise SlackEnvironmentError(
                    f"{entry['path']}: unsupported Slack payload"
                )
            if entry["bytes"] > limit:
                raise SlackEnvironmentError(
                    "delta/slack/seed.json: exceeds max_delta_bytes", code="delta_limit"
                )
        seed = package / "delta/slack/seed.json"
        delta = load_json(seed) if slack_files else {}
        if not isinstance(delta, dict) or delta.keys() - {"messages", "reactions"}:
            raise SlackEnvironmentError(
                "delta/slack/seed.json: only messages and reactions are supported"
            )
        for dataset, columns in (
            ("messages", OVERLAY_MESSAGE_COLUMNS),
            ("reactions", OVERLAY_REACTION_COLUMNS),
        ):
            rows = delta.get(dataset, [])
            if not isinstance(rows, list):
                raise SlackEnvironmentError(
                    f"delta/slack/seed.json: {dataset}: expected a list"
                )
            seen = set()
            for i, row in enumerate(rows):
                location = f"delta/slack/seed.json: {dataset}[{i}]"
                if not isinstance(row, dict) or row.keys() - set(columns):
                    raise SlackEnvironmentError(f"{location}: unsupported record keys")
                required = (
                    {"channel_id", "ts", "user_id", "text"}
                    if dataset == "messages"
                    else {"channel_id", "message_ts", "name", "user_id", "created_ts"}
                )
                for key in required:
                    if not isinstance(row.get(key), str) or (
                        key != "text" and not row[key]
                    ):
                        raise SlackEnvironmentError(
                            f"{location}.{key}: required string"
                        )
                if row.get("operation", "upsert") != "upsert":
                    raise SlackEnvironmentError(
                        f"{location}.operation: only upsert is supported"
                    )
                for key in ("channel_id", "user_id"):
                    if not _NATIVE_ID.fullmatch(row[key]):
                        raise SlackEnvironmentError(
                            f"{location}.{key}: invalid native ID"
                        )
                for key in (
                    ("ts", "thread_ts", "edited_ts")
                    if dataset == "messages"
                    else ("message_ts", "created_ts")
                ):
                    if row.get(key) is not None and not self._valid_ts(row[key]):
                        raise SlackEnvironmentError(
                            f"{location}.{key}: expected seconds.microseconds (six digits)"
                        )
                if dataset == "messages":
                    for key in ("subtype", "client_msg_id", "event_id"):
                        if row.get(key) is not None and not isinstance(row[key], str):
                            raise SlackEnvironmentError(
                                f"{location}.{key}: expected string or null"
                            )
                    blocks = row.get("blocks_json", [])
                    if isinstance(blocks, str):
                        try:

                            def reject_constant(value):
                                raise ValueError(f"invalid JSON constant: {value}")

                            blocks = json.loads(blocks, parse_constant=reject_constant)
                        except ValueError as exc:
                            raise SlackEnvironmentError(
                                f"{location}.blocks_json: invalid JSON"
                            ) from exc
                    if not isinstance(blocks, list):
                        raise SlackEnvironmentError(
                            f"{location}.blocks_json: expected a JSON array"
                        )
                    if "created_at" in row:
                        timestamp(
                            row["created_at"],
                            "delta/slack/seed.json",
                            f"{dataset}[{i}].created_at",
                        )
                key = self._identity(row)
                if key in seen:
                    raise SlackEnvironmentError(f"{location}: duplicate record {key}")
                seen.add(key)
        payload = {self._identity(row) for rows in delta.values() for row in rows}
        inventory = set()
        for i, change in enumerate(task["changes"]):
            if change["service"] != "slack":
                continue
            location = f"task.json: changes[{i}]"
            record = change["source"]["record"]
            expected_keys = (
                {"channel_id", "ts"}
                if "ts" in record
                else {"channel_id", "message_ts", "name", "user_id"}
            )
            if (
                record.keys() != expected_keys
                or change["file"] != "delta/slack/seed.json"
            ):
                raise SlackEnvironmentError(
                    f"{location}: unsupported Slack inventory record or file"
                )
            if any(
                not isinstance(value, str) or not value for value in record.values()
            ):
                raise SlackEnvironmentError(
                    f"{location}: native identity fields must be non-empty strings"
                )
            key = self._identity(record)
            if key in inventory:
                raise SlackEnvironmentError(
                    f"{location}: duplicate inventory record {key}"
                )
            inventory.add(key)
        if payload != inventory:
            raise SlackEnvironmentError(
                "delta/slack/seed.json: inventory does not match message/reaction records"
            )
        return delta

    @staticmethod
    def _identity(row: dict) -> tuple:
        if "ts" in row:
            return ("messages", row["channel_id"], row["ts"])
        return (
            "reactions",
            row["channel_id"],
            row["message_ts"],
            row["name"],
            row["user_id"],
        )

    @staticmethod
    def _valid_ts(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(_SLACK_TS, value) is not None

    def _validate_native(self, task: dict, delta: dict) -> None:
        tools = self._tools
        for i, change in enumerate(task["changes"]):
            if change["service"] != "slack":
                continue
            key = self._identity(change["source"]["record"])
            dataset, *values = key
            fields = (
                ("channel_id", "ts")
                if dataset == "messages"
                else ("channel_id", "message_ts", "name", "user_id")
            )
            exists = (
                tools._duck.execute(
                    f"SELECT 1 FROM baseline_{dataset} WHERE "
                    + " AND ".join(f"{f} = ?" for f in fields)
                    + " LIMIT 1",
                    values,
                ).fetchone()
                is not None
            )
            if exists != (change["operation"] == "replace"):
                raise SlackEnvironmentError(
                    f"delta/slack/seed.json: task.json changes[{i}]: "
                    f"{change['operation']} identity conflicts with baseline {key}"
                )
            channel = tools._channels.get(values[0])
            if channel and (channel["is_im"] or channel["is_mpim"]):
                raise SlackEnvironmentError(
                    f"delta/slack/seed.json: {key}: employee DM changes are unsupported"
                )

        def check(dataset: str, predicate: str, reason: str) -> None:
            # One diagnostic record only, even for a large baseline.
            fields = (
                "m.channel_id, m.ts"
                if dataset == "messages"
                else "m.channel_id, m.message_ts, m.name, m.user_id"
            )
            row = tools._duck.execute(
                f"SELECT {fields} FROM effective_{dataset} m "
                f"WHERE {predicate} LIMIT 1"
            ).fetchone()
            if row:
                origin = (
                    "delta/slack/seed.json"
                    if (dataset, *row)
                    in {self._identity(r) for r in delta.get(dataset, [])}
                    else f"baseline/{dataset}"
                )
                raise SlackEnvironmentError(
                    f"{origin}: record {row}: {reason}",
                    code=(
                        "state_after_snapshot"
                        if reason == "state is after snapshot"
                        else "invalid_native_record"
                    ),
                    record=dict(zip(fields.replace("m.", "").split(", "), row)),
                    source=origin,
                    validation_reason=reason,
                )

        def invalid_ts(field: str, optional: bool = False) -> str:
            test = (
                f"(NOT regexp_full_match(m.{field}, '{_SLACK_TS}') OR m.{field} IS NULL "
                f"OR TRY_CAST(m.{field} AS DECIMAL(20,6)) >= 253402300800)"
            )
            return f"(m.{field} IS NOT NULL AND {test})" if optional else test

        for dataset in ("messages", "reactions"):
            keys = (
                "channel_id, ts"
                if dataset == "messages"
                else "channel_id, message_ts, name, user_id"
            )
            duplicate = tools._duck.execute(
                f"SELECT {keys} FROM effective_{dataset} "
                f"GROUP BY {keys} HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate:
                raise SlackEnvironmentError(
                    f"baseline/{dataset}: duplicate native identity {duplicate}"
                )
            fields = (
                ("ts", "thread_ts", "edited_ts")
                if dataset == "messages"
                else ("message_ts", "created_ts")
            )
            for field in fields:
                check(
                    dataset,
                    invalid_ts(field, field in {"thread_ts", "edited_ts"}),
                    f"invalid {field}",
                )
            check(
                dataset,
                "NOT EXISTS (SELECT 1 FROM baseline_channels c WHERE c.id = m.channel_id) "
                "OR NOT EXISTS (SELECT 1 FROM baseline_users u WHERE u.id = m.user_id)",
                "unknown channel or author",
            )
            dated_fields = (
                ("ts", "edited_ts") if dataset == "messages" else ("created_ts",)
            )
            for field in dated_fields:
                check(
                    dataset,
                    f"CAST(m.{field} AS DECIMAL(20,6)) > "
                    f"CAST('{_micros_to_timestamp(self._snapshot_micros)}' AS DECIMAL(20,6))",
                    "state is after snapshot",
                )
        check(
            "messages",
            "m.text IS NULL OR m.blocks_json IS NULL OR NOT json_valid(m.blocks_json)",
            "invalid message text or blocks",
        )
        check(
            "messages",
            "json_type(m.blocks_json) != 'ARRAY'",
            "message blocks must be a JSON array",
        )
        check(
            "messages",
            "m.edited_ts IS NOT NULL AND CAST(m.edited_ts AS DECIMAL(20,6)) < CAST(m.ts AS DECIMAL(20,6))",
            "edit predates message",
        )
        check(
            "messages",
            "m.thread_ts IS NOT NULL AND NOT EXISTS (SELECT 1 FROM effective_messages r "
            "WHERE r.channel_id = m.channel_id AND r.ts = m.thread_ts AND r.thread_ts IS NULL "
            "AND CAST(r.ts AS DECIMAL(20,6)) < CAST(m.ts AS DECIMAL(20,6)))",
            "invalid thread root/link",
        )
        check(
            "reactions",
            "m.name IS NULL OR m.name = '' OR NOT EXISTS (SELECT 1 FROM effective_messages r "
            "WHERE r.channel_id = m.channel_id AND r.ts = m.message_ts "
            "AND CAST(r.ts AS DECIMAL(20,6)) <= CAST(m.created_ts AS DECIMAL(20,6)))",
            "invalid reaction link/time",
        )
        for dataset, catalog in (
            ("users", tools._users),
            ("channels", tools._channels),
        ):
            count = tools._duck.execute(
                f"SELECT COUNT(*) FROM baseline_{dataset}"
            ).fetchone()[0]
            if count != len(catalog) or any(
                not _NATIVE_ID.fullmatch(key or "") for key in catalog
            ):
                raise SlackEnvironmentError(
                    f"baseline/{dataset}.parquet: duplicate or invalid native IDs"
                )

    def initial_context(self) -> dict:
        self._ensure_open()
        context = {"snapshot_time": self._snapshot_time, "messages": []}
        size = len(_encoded(context))
        if size > self._context_limit:
            raise SlackEnvironmentError(
                "initial_context exceeds initial_context_max_bytes",
                code="context_limit",
            )
        cursor = self._tools._duck.execute(
            "SELECT m.channel_id, m.user_id, m.ts, m.thread_ts, m.text "
            "FROM effective_messages m JOIN baseline_channels c ON c.id = m.channel_id "
            "WHERE NOT coalesce(c.is_im, false) AND NOT coalesce(c.is_mpim, false) "
            "AND CAST(m.ts AS DECIMAL(20,6)) BETWEEN CAST(? AS DECIMAL(20,6)) AND CAST(? AS DECIMAL(20,6)) "
            "ORDER BY CAST(m.ts AS DECIMAL(20,6)), m.channel_id, m.user_id",
            [
                _micros_to_timestamp(
                    max(0, self._snapshot_micros - self._window_minutes * 60_000_000)
                ),
                _micros_to_timestamp(self._snapshot_micros),
            ],
        )
        while row := cursor.fetchone():
            message = dict(
                zip(("channel_id", "user_id", "ts", "thread_ts", "text"), row)
            )
            user = self._tools._users[message["user_id"]]
            channel = self._tools._channels[message["channel_id"]]
            message["author"] = {
                key: user[key]
                for key in ("real_name", "display_name", "title")
                if user.get(key)
            }
            message["channel"] = {
                key: channel[key]
                for key in ("name", "topic", "purpose")
                if channel.get(key)
            }
            size += len(_encoded(message)) + (1 if context["messages"] else 0)
            if size > self._context_limit:
                raise SlackEnvironmentError(
                    "initial_context exceeds initial_context_max_bytes; increase the explicit limit",
                    code="context_limit",
                )
            context["messages"].append(message)
        return context

    def call(self, name: str, arguments: dict) -> dict:
        self._ensure_open()
        if not isinstance(name, str) or name not in self._scope:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": "Unknown or disallowed tool",
            }
        try:
            required, optional, _ = _TOOLS[name]
            if not isinstance(arguments, dict) or arguments.keys() - (
                required.keys() | optional.keys()
            ):
                raise SlackToolError("invalid_arguments", "Unexpected tool arguments")
            if required.keys() - arguments.keys():
                raise SlackToolError(
                    "invalid_arguments", "Required tool arguments are missing"
                )
            for key, value in arguments.items():
                kind = {**required, **optional}[key]
                valid = (
                    (kind == "integer" and type(value) is int and value > 0)
                    or (kind == "boolean" and type(value) is bool)
                    or (kind == "string" and isinstance(value, str))
                    or (kind == "cursor" and (value is None or isinstance(value, str)))
                    or (
                        kind == "types"
                        and (
                            isinstance(value, str)
                            or isinstance(value, list)
                            and all(isinstance(v, str) for v in value)
                        )
                    )
                )
                if not valid:
                    raise SlackToolError(
                        "invalid_arguments", f"Invalid argument type for {key}"
                    )
                if isinstance(value, str) or isinstance(value, list):
                    try:
                        _encoded(value)
                    except UnicodeError:
                        raise SlackToolError(
                            "invalid_arguments", f"Invalid Unicode text for {key}"
                        ) from None
            result = getattr(self._tools, name)(**arguments)
            if len(_encoded(result)) > self._result_limit:
                # Do not hide a successful write behind a size error.
                if name in {
                    "slack_post_message",
                    "slack_reply_to_thread",
                    "slack_send_dm",
                }:
                    message = result["message"]
                    return {
                        "ok": True,
                        "object": "delivery",
                        "channel_id": message["channel_id"],
                        "ts": message["ts"],
                        "user_id": message["user_id"],
                        **(
                            {
                                "destination": {
                                    "kind": "dm",
                                    "user_id": result["destination"]["user_id"],
                                }
                            }
                            if "destination" in result
                            else {}
                        ),
                    }
                raise SlackToolError(
                    "result_too_large",
                    "Tool result exceeds max_result_bytes; request a smaller page",
                )
            return result
        except SlackToolError as exc:
            # Fixed lookup messages avoid reflecting paths or guessed hidden IDs.
            if exc.code in {"channel_not_found", "user_not_found"}:
                return {
                    "ok": False,
                    "error": exc.code,
                    "message": "Slack destination was not found",
                }
            if exc.code == "invalid_query":
                return {
                    "ok": False,
                    "error": exc.code,
                    "message": "Invalid Slack search query",
                }
            return exc.as_dict()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Slack environment is closed")

    def close(self) -> None:
        if self._tools is not None:
            self._tools.close()
            self._tools = None
        self._closed = True

    def __enter__(self) -> "SlackEnvironment":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
