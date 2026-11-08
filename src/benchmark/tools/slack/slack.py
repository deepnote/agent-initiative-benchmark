"""Local Slack benchmark wrappers backed by Parquet and a SQLite overlay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import sqlite3
import threading
import tempfile
import time
import tomllib
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from benchmark import duckdb_runtime as duckdb


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")

    configured: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        configured.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return configured[0], configured[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "slack"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "slack"


class SlackToolError(ValueError):
    """A predictable error that can be returned to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class SlackConfig:
    """Paths, identity, limits, and query settings for the Slack wrappers."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    agent_user_id: str = "U_AGENT"
    max_page_size: int = 100
    max_message_characters: int = 40_000
    max_search_query_characters: int = 1_000
    sqlite_timeout_seconds: float = 30.0
    duckdb_threads: int = 4
    benchmark_mode: bool = False
    scratch_dir: Path | None = None
    duckdb_memory_limit: str = "1GiB"
    duckdb_max_temp_directory_size: str = "4GiB"
    max_related_rows: int = 10_000

    @classmethod
    def from_env(cls) -> "SlackConfig":
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_SLACK_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_SLACK_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            agent_user_id=os.getenv("BENCHMARK_SLACK_AGENT_USER_ID", cls.agent_user_id),
            max_page_size=int(os.getenv("SLACK_MAX_PAGE_SIZE", cls.max_page_size)),
            max_message_characters=int(
                os.getenv("SLACK_MAX_MESSAGE_CHARACTERS", cls.max_message_characters)
            ),
            max_search_query_characters=int(
                os.getenv(
                    "SLACK_MAX_SEARCH_QUERY_CHARACTERS",
                    cls.max_search_query_characters,
                )
            ),
            sqlite_timeout_seconds=float(
                os.getenv("SLACK_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
            duckdb_threads=int(os.getenv("SLACK_DUCKDB_THREADS", cls.duckdb_threads)),
        )

    def validate(self) -> None:
        for name in (
            "max_page_size",
            "max_message_characters",
            "max_search_query_characters",
            "duckdb_threads",
            "max_related_rows",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")


PARQUET_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "workspaces": (
        ("id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("domain", "VARCHAR"),
        ("created_ts", "VARCHAR"),
    ),
    "users": (
        ("id", "VARCHAR"),
        ("workspace_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("real_name", "VARCHAR"),
        ("display_name", "VARCHAR"),
        ("title", "VARCHAR"),
        ("email", "VARCHAR"),
        ("timezone", "VARCHAR"),
        ("is_bot", "BOOLEAN"),
        ("is_admin", "BOOLEAN"),
        ("is_owner", "BOOLEAN"),
        ("is_deleted", "BOOLEAN"),
    ),
    "channels": (
        ("id", "VARCHAR"),
        ("workspace_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("is_private", "BOOLEAN"),
        ("is_im", "BOOLEAN"),
        ("is_mpim", "BOOLEAN"),
        ("is_archived", "BOOLEAN"),
        ("topic", "VARCHAR"),
        ("purpose", "VARCHAR"),
        ("creator_id", "VARCHAR"),
        ("created_ts", "VARCHAR"),
        ("member_ids_json", "VARCHAR"),
    ),
    "messages": (
        ("channel_id", "VARCHAR"),
        ("ts", "VARCHAR"),
        ("thread_ts", "VARCHAR"),
        ("user_id", "VARCHAR"),
        ("text", "VARCHAR"),
        ("blocks_json", "VARCHAR"),
        ("subtype", "VARCHAR"),
        ("edited_ts", "VARCHAR"),
        ("client_msg_id", "VARCHAR"),
        ("event_id", "VARCHAR"),
    ),
    "threads": (
        ("channel_id", "VARCHAR"),
        ("thread_ts", "VARCHAR"),
        ("reply_count", "BIGINT"),
        ("participant_ids_json", "VARCHAR"),
        ("latest_reply_ts", "VARCHAR"),
    ),
    "reactions": (
        ("channel_id", "VARCHAR"),
        ("message_ts", "VARCHAR"),
        ("name", "VARCHAR"),
        ("user_id", "VARCHAR"),
        ("created_ts", "VARCHAR"),
    ),
    "files": (
        ("id", "VARCHAR"),
        ("channel_id", "VARCHAR"),
        ("message_ts", "VARCHAR"),
        ("user_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("title", "VARCHAR"),
        ("mimetype", "VARCHAR"),
        ("size", "BIGINT"),
        ("url_private", "VARCHAR"),
        ("permalink", "VARCHAR"),
    ),
}


OVERLAY_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    channel_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    thread_ts TEXT,
    user_id TEXT,
    text TEXT NOT NULL DEFAULT '',
    blocks_json TEXT NOT NULL DEFAULT '[]',
    subtype TEXT,
    edited_ts TEXT,
    client_msg_id TEXT,
    event_id TEXT,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (channel_id, ts)
);

CREATE INDEX IF NOT EXISTS messages_thread_idx
    ON messages(channel_id, thread_ts, ts);
CREATE INDEX IF NOT EXISTS messages_user_idx
    ON messages(user_id, ts);

CREATE TABLE IF NOT EXISTS reactions (
    channel_id TEXT NOT NULL,
    message_ts TEXT NOT NULL,
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete')),
    PRIMARY KEY (channel_id, message_ts, name, user_id)
);

CREATE INDEX IF NOT EXISTS reactions_message_idx
    ON reactions(channel_id, message_ts);
"""


OVERLAY_MESSAGE_COLUMNS = (
    "channel_id",
    "ts",
    "thread_ts",
    "user_id",
    "text",
    "blocks_json",
    "subtype",
    "edited_ts",
    "client_msg_id",
    "event_id",
    "operation",
    "created_at",
)

OVERLAY_REACTION_COLUMNS = (
    "channel_id",
    "message_ts",
    "name",
    "user_id",
    "created_ts",
    "operation",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _timestamp_to_datetime(value: str) -> datetime:
    try:
        seconds, micros = value.split(".", 1)
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=int(seconds),
            microseconds=int(micros.ljust(6, "0")[:6]),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SlackToolError(
            "invalid_ts", f"Invalid Slack timestamp: {value}"
        ) from error


def _timestamp_to_micros(value: str) -> int:
    try:
        seconds, micros = value.split(".", 1)
        return int(seconds) * 1_000_000 + int(micros.ljust(6, "0")[:6])
    except (TypeError, ValueError) as error:
        raise SlackToolError(
            "invalid_ts", f"Invalid Slack timestamp: {value}"
        ) from error


def _micros_to_timestamp(value: int) -> str:
    seconds, micros = divmod(value, 1_000_000)
    return f"{seconds}.{micros:06d}"


def _timestamp_iso(value: str) -> str:
    return _timestamp_to_datetime(value).isoformat(timespec="microseconds")


def _validate_seed(seed: Mapping[str, Any]) -> None:
    for dataset in ("workspaces", "users", "channels", "messages"):
        if not isinstance(seed.get(dataset), list):
            raise ValueError(f"Slack seed needs a {dataset} list")

    workspaces = {str(item["id"]) for item in seed["workspaces"]}
    users = {str(item["id"]): item for item in seed["users"]}
    channels = {str(item["id"]): item for item in seed["channels"]}
    messages = {
        (str(item["channel_id"]), str(item["ts"])): item for item in seed["messages"]
    }
    if len(workspaces) != len(seed["workspaces"]):
        raise ValueError("Slack workspace IDs must be unique")
    if len(users) != len(seed["users"]):
        raise ValueError("Slack user IDs must be unique")
    if len(channels) != len(seed["channels"]):
        raise ValueError("Slack channel IDs must be unique")
    if len(messages) != len(seed["messages"]):
        raise ValueError("Slack message channel and timestamp pairs must be unique")

    for user in users.values():
        if str(user["workspace_id"]) not in workspaces:
            raise ValueError(f"User {user['id']} has an unknown workspace")
    for channel in channels.values():
        if str(channel["workspace_id"]) not in workspaces:
            raise ValueError(f"Channel {channel['id']} has an unknown workspace")
        for user_id in channel.get("member_ids_json", []):
            if str(user_id) not in users:
                raise ValueError(
                    f"Channel {channel['id']} has unknown member {user_id}"
                )
    for message in messages.values():
        key = (str(message["channel_id"]), str(message["ts"]))
        if key[0] not in channels:
            raise ValueError(f"Message {key} has an unknown channel")
        if str(message["user_id"]) not in users:
            raise ValueError(f"Message {key} has an unknown user")
        _timestamp_to_datetime(key[1])
        if message.get("thread_ts"):
            root_key = (key[0], str(message["thread_ts"]))
            if root_key not in messages:
                raise ValueError(f"Reply {key} has no root message {root_key}")
            if key[1] <= root_key[1]:
                raise ValueError(f"Reply {key} must be newer than its root")

    for reaction in seed.get("reactions", []):
        key = (str(reaction["channel_id"]), str(reaction["message_ts"]))
        if key not in messages:
            raise ValueError(f"Reaction points to unknown message {key}")
        if str(reaction["user_id"]) not in users:
            raise ValueError(f"Reaction on {key} has an unknown user")
    for file in seed.get("files", []):
        key = (str(file["channel_id"]), str(file["message_ts"]))
        if key not in messages:
            raise ValueError(f"File {file['id']} points to unknown message {key}")


def build_slack_overlay_from_seed(
    seed_path: Path, overlay_path: Path
) -> dict[str, Any]:
    """Create a writable SQLite overlay from a small JSON delta."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise ValueError("Slack overlay seed must be a JSON object")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_path.exists():
        overlay_path.unlink()
    connection = sqlite3.connect(overlay_path)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("format_version", "1"),
        )
        for row in seed.get("messages", []):
            values = {
                "channel_id": row["channel_id"],
                "ts": row["ts"],
                "thread_ts": row.get("thread_ts"),
                "user_id": row.get("user_id"),
                "text": row.get("text", ""),
                "blocks_json": row.get("blocks_json", []),
                "subtype": row.get("subtype"),
                "edited_ts": row.get("edited_ts"),
                "client_msg_id": row.get("client_msg_id"),
                "event_id": row.get("event_id"),
                "operation": row.get("operation", "upsert"),
                "created_at": row.get("created_at", _timestamp_iso(str(row["ts"]))),
            }
            if not isinstance(values["blocks_json"], str):
                values["blocks_json"] = _json(values["blocks_json"])
            connection.execute(
                f"INSERT INTO messages({','.join(OVERLAY_MESSAGE_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in OVERLAY_MESSAGE_COLUMNS)})",
                tuple(values[column] for column in OVERLAY_MESSAGE_COLUMNS),
            )
        for row in seed.get("reactions", []):
            values = {column: row.get(column) for column in OVERLAY_REACTION_COLUMNS}
            values["operation"] = values["operation"] or "upsert"
            connection.execute(
                f"INSERT INTO reactions({','.join(OVERLAY_REACTION_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in OVERLAY_REACTION_COLUMNS)})",
                tuple(values[column] for column in OVERLAY_REACTION_COLUMNS),
            )
        connection.commit()
    finally:
        connection.close()
    return {
        "messages": len(seed.get("messages", [])),
        "reactions": len(seed.get("reactions", [])),
    }


class SlackTools:
    """Read effective Slack state and write only to the task overlay."""

    def __init__(
        self,
        config: SlackConfig | None = None,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ):
        self.config = config or SlackConfig.from_env()
        self.config.validate()
        self._clock_ns = clock_ns
        self._id_factory = id_factory
        self._lock = threading.RLock()
        self._validate_baseline_files()

        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        self._overlay = sqlite3.connect(
            overlay_path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self._overlay.row_factory = sqlite3.Row
        self._overlay.execute("PRAGMA foreign_keys = ON")
        self._overlay.execute("PRAGMA busy_timeout = 30000")
        self._overlay.executescript(OVERLAY_SQL)

        self._duck = None
        self._scratch = None
        try:
            self._initialize_queries()
        except BaseException:
            self.close()
            raise

    def _initialize_queries(self) -> None:
        if self.config.scratch_dir is None:
            self._duck = duckdb.connect(database=":memory:")
        else:
            # An explicit per-attempt disk database, without changing global env.
            import duckdb as native_duckdb

            scratch = self.config.scratch_dir
            scratch.mkdir(parents=True, exist_ok=True)
            duckdb._check_scratch_disk(scratch)
            self._scratch = tempfile.TemporaryDirectory(prefix="query-", dir=scratch)
            scratch = Path(self._scratch.name)
            self._duck = native_duckdb.connect(
                str(scratch / "query.duckdb"),
                config={
                    "memory_limit": self.config.duckdb_memory_limit,
                    "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
                    "temp_directory": str(scratch / "spill"),
                },
            )
        self._duck.execute(f"PRAGMA threads={self.config.duckdb_threads}")
        self._create_views()
        self._reload_overlay_tables()
        self._users = self._load_users()
        self._channels = self._load_channels()
        self._dm_channels: set[str] = set()
        if self.config.benchmark_mode:
            self._overlay.execute(
                "CREATE TABLE IF NOT EXISTS dm_conversations "
                "(user_id TEXT PRIMARY KEY, channel_json TEXT NOT NULL)"
            )
            for row in self._overlay.execute(
                "SELECT channel_json FROM dm_conversations"
            ):
                channel = json.loads(row[0])
                self._channels[channel["id"]] = channel
                self._dm_channels.add(channel["id"])
        if self.config.agent_user_id not in self._users:
            raise SlackToolError(
                "invalid_auth",
                f"Agent user {self.config.agent_user_id} does not exist in Slack data",
            )

    def _validate_baseline_files(self) -> None:
        expected = [
            self.config.baseline_dir / f"{dataset}.parquet"
            for dataset in PARQUET_SCHEMAS
            if dataset != "messages"
        ]
        expected.append(self.config.baseline_dir / "messages")
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Slack baseline data: " + ", ".join(missing)
            )

    def _create_views(self) -> None:
        root = self.config.baseline_dir
        for dataset in PARQUET_SCHEMAS:
            if dataset == "messages":
                path = root / "messages" / "**" / "*.parquet"
                options = ", union_by_name=true, hive_partitioning=true"
            else:
                path = root / f"{dataset}.parquet"
                options = ""
            self._duck.execute(
                f"CREATE VIEW baseline_{dataset} AS "
                f"SELECT * FROM read_parquet('{_sql_path(path)}'{options})"
            )

        self._duck.execute(
            """
            CREATE TABLE overlay_messages (
                channel_id VARCHAR,
                ts VARCHAR,
                thread_ts VARCHAR,
                user_id VARCHAR,
                text VARCHAR,
                blocks_json VARCHAR,
                subtype VARCHAR,
                edited_ts VARCHAR,
                client_msg_id VARCHAR,
                event_id VARCHAR,
                operation VARCHAR,
                created_at VARCHAR
            )
            """
        )
        self._duck.execute(
            """
            CREATE TABLE overlay_reactions (
                channel_id VARCHAR,
                message_ts VARCHAR,
                name VARCHAR,
                user_id VARCHAR,
                created_ts VARCHAR,
                operation VARCHAR
            )
            """
        )
        self._duck.execute(
            """
            CREATE VIEW effective_messages AS
            SELECT channel_id, ts, thread_ts, user_id, text, blocks_json, subtype,
                   edited_ts, client_msg_id, event_id, 'overlay' AS source
            FROM overlay_messages
            WHERE operation = 'upsert'
            UNION ALL
            SELECT b.channel_id, b.ts, b.thread_ts, b.user_id, b.text, b.blocks_json,
                   b.subtype, b.edited_ts, b.client_msg_id, b.event_id,
                   'baseline' AS source
            FROM baseline_messages b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_messages o
                WHERE o.channel_id = b.channel_id AND o.ts = b.ts
            )
            """
        )
        self._duck.execute(
            """
            CREATE VIEW effective_reactions AS
            SELECT channel_id, message_ts, name, user_id, created_ts,
                   'overlay' AS source
            FROM overlay_reactions
            WHERE operation = 'upsert'
            UNION ALL
            SELECT b.channel_id, b.message_ts, b.name, b.user_id, b.created_ts,
                   'baseline' AS source
            FROM baseline_reactions b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_reactions o
                WHERE o.channel_id = b.channel_id
                  AND o.message_ts = b.message_ts
                  AND o.name = b.name
                  AND o.user_id = b.user_id
            )
            """
        )

    def _reload_overlay_tables(self) -> None:
        self._duck.execute("DELETE FROM overlay_messages")
        rows = self._overlay.execute(
            f"SELECT {','.join(OVERLAY_MESSAGE_COLUMNS)} FROM messages"
        )
        while batch := rows.fetchmany(256):
            self._duck.executemany(
                f"INSERT INTO overlay_messages VALUES ({','.join('?' for _ in OVERLAY_MESSAGE_COLUMNS)})",
                [
                    tuple(row[column] for column in OVERLAY_MESSAGE_COLUMNS)
                    for row in batch
                ],
            )
        self._duck.execute("DELETE FROM overlay_reactions")
        reactions = self._overlay.execute(
            f"SELECT {','.join(OVERLAY_REACTION_COLUMNS)} FROM reactions"
        )
        while batch := reactions.fetchmany(256):
            self._duck.executemany(
                f"INSERT INTO overlay_reactions VALUES ({','.join('?' for _ in OVERLAY_REACTION_COLUMNS)})",
                [
                    tuple(row[column] for column in OVERLAY_REACTION_COLUMNS)
                    for row in batch
                ],
            )

    def _load_users(self) -> dict[str, dict[str, Any]]:
        rows = self._rows("SELECT * FROM baseline_users")
        return {row["id"]: row for row in rows}

    def _load_channels(self) -> dict[str, dict[str, Any]]:
        rows = self._rows("SELECT * FROM baseline_channels")
        for row in rows:
            row["member_ids"] = _parse_json(row.pop("member_ids_json", "[]"), [])
        return {row["id"]: row for row in rows}

    def close(self) -> None:
        with self._lock:
            try:
                if self._duck is not None:
                    self._duck.close()
                    self._duck = None
            finally:
                self._overlay.close()
                if self._scratch is not None:
                    self._scratch.cleanup()
                    self._scratch = None

    def __enter__(self) -> "SlackTools":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def slack_list_channels(
        self,
        *,
        types: str | Sequence[str] = ("public_channel", "private_channel"),
        exclude_archived: bool = True,
        page_size: int = 100,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List channels visible to the agent, with stable cursor pagination."""

        with self._lock:
            allowed_types = self._channel_types(types)
            channels = [
                self._channel_result(channel)
                for channel in self._channels.values()
                if self._channel_kind(channel) in allowed_types
                and self._can_read_channel(channel)
                and (not exclude_archived or not channel["is_archived"])
            ]
            channels.sort(key=lambda item: (item["name"].lower(), item["id"]))
            limit = self._page_size(page_size)
            scope = self._scope("channels", sorted(allowed_types), exclude_archived)
            offset = self._cursor_offset(start_cursor, scope)
            page = channels[offset : offset + limit]
            next_offset = offset + len(page)
            has_more = next_offset < len(channels)
            return {
                "ok": True,
                "object": "channel_list",
                "channels": page,
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def slack_search_messages(
        self,
        query: str,
        *,
        sort: str = "score",
        sort_dir: str = "desc",
        page_size: int = 20,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search effective messages with common Slack query modifiers."""

        with self._lock:
            parsed = self._parse_search(query)
            if sort not in {"score", "timestamp"}:
                raise SlackToolError(
                    "invalid_arguments", "sort must be score or timestamp"
                )
            if sort_dir not in {"asc", "desc"}:
                raise SlackToolError(
                    "invalid_arguments", "sort_dir must be asc or desc"
                )

            accessible = {
                channel_id
                for channel_id, channel in self._channels.items()
                if self._can_read_channel(channel)
            }
            if parsed["channel_id"]:
                accessible &= {parsed["channel_id"]}
            if not accessible:
                return {
                    "ok": True,
                    "object": "message_search",
                    "query": query,
                    "messages": [],
                    "total": 0,
                    "has_more": False,
                    "next_cursor": None,
                }

            where, params = self._search_where(parsed, accessible)
            score_sql, score_params = self._score_sql(parsed["terms"])
            direction = "ASC" if sort_dir == "asc" else "DESC"
            if sort == "score":
                order = f"score {direction}, CAST(m.ts AS DECIMAL(20,6)) {direction}, m.channel_id ASC"
            else:
                order = f"CAST(m.ts AS DECIMAL(20,6)) {direction}, m.channel_id ASC"
            scope = self._scope("search", query, sort, sort_dir)
            offset = self._cursor_offset(start_cursor, scope)
            limit = self._page_size(page_size)

            total_row = self._duck.execute(
                f"SELECT COUNT(*) FROM effective_messages m WHERE {where}", params
            ).fetchone()
            total = int(total_row[0])
            rows = self._rows(
                f"SELECT m.*, {score_sql} AS score "
                f"FROM effective_messages m WHERE {where} "
                f"ORDER BY {order} LIMIT ? OFFSET ?",
                [*score_params, *params, limit, offset],
            )
            messages = self._hydrate_messages(rows)
            for message, row in zip(messages, rows):
                message["score"] = float(row["score"])
            next_offset = offset + len(messages)
            has_more = next_offset < total
            return {
                "ok": True,
                "object": "message_search",
                "query": query,
                "messages": messages,
                "total": total,
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def slack_get_thread(
        self,
        channel_id: str,
        thread_ts: str,
        *,
        page_size: int = 100,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return one root message and its replies in timestamp order."""

        with self._lock:
            channel = self._require_readable_channel(channel_id)
            root = self._effective_message(channel["id"], thread_ts)
            if not root or root.get("thread_ts"):
                raise SlackToolError(
                    "thread_not_found", "Slack thread root was not found"
                )
            limit = self._page_size(page_size)
            scope = self._scope("thread", channel["id"], thread_ts)
            offset = self._cursor_offset(start_cursor, scope)
            total = int(
                self._duck.execute(
                    "SELECT COUNT(*) FROM effective_messages "
                    "WHERE channel_id = ? AND (ts = ? OR thread_ts = ?)",
                    [channel["id"], thread_ts, thread_ts],
                ).fetchone()[0]
            )
            page = self._rows(
                """
                SELECT * FROM effective_messages
                WHERE channel_id = ? AND (ts = ? OR thread_ts = ?)
                ORDER BY CAST(ts AS DECIMAL(20,6)) ASC
                LIMIT ? OFFSET ?
                """,
                [channel["id"], thread_ts, thread_ts, limit, offset],
            )
            messages = self._hydrate_messages(page)
            next_offset = offset + len(messages)
            has_more = next_offset < total
            return {
                "ok": True,
                "object": "thread",
                "channel": self._channel_result(channel),
                "thread_ts": thread_ts,
                "messages": messages,
                "reply_count": max(0, total - 1),
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def slack_post_message(self, channel_id: str, text: str) -> dict[str, Any]:
        """Post a top-level message as the configured agent user."""

        with self._lock:
            channel = self._require_writable_channel(channel_id)
            return self._write_message(channel, text, thread_ts=None)

    def slack_reply_to_thread(
        self, channel_id: str, thread_ts: str, text: str
    ) -> dict[str, Any]:
        """Post a reply to an existing root using Slack's thread_ts behavior."""

        with self._lock:
            channel = self._require_writable_channel(channel_id)
            root = self._effective_message(channel["id"], thread_ts)
            if not root or root.get("thread_ts"):
                raise SlackToolError(
                    "thread_not_found", "Slack thread root was not found"
                )
            return self._write_message(channel, text, thread_ts=thread_ts)

    def slack_list_users(
        self, *, page_size: int = 100, start_cursor: str | None = None
    ) -> dict[str, Any]:
        """Discover active employees without a prior conversation."""
        with self._lock:
            limit = self._page_size(page_size)
            scope = self._scope("users")
            offset = self._cursor_offset(start_cursor, scope)
            users = sorted(
                (
                    user
                    for user in self._users.values()
                    if not user.get("is_bot")
                    and not user.get("is_deleted")
                    and user["id"] != self.config.agent_user_id
                ),
                key=lambda user: user["id"],
            )
            page = [self._user_result(user) for user in users[offset : offset + limit]]
            has_more = offset + len(page) < len(users)
            return {
                "ok": True,
                "object": "user_list",
                "users": page,
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(offset + len(page), scope) if has_more else None
                ),
            }

    def slack_send_dm(self, user_id: str, text: str) -> dict[str, Any]:
        """Create/reuse a persisted attempt-local agent conversation."""
        with self._lock:
            if not self.config.benchmark_mode:
                raise SlackToolError("not_allowed", "DM setup requires benchmark mode")
            self._validate_message_text(text)
            user = self._users.get(user_id)
            if (
                not user
                or user_id == self.config.agent_user_id
                or user.get("is_bot")
                or user.get("is_deleted")
            ):
                raise SlackToolError(
                    "user_not_found", "Active Slack employee was not found"
                )
            stored = self._overlay.execute(
                "SELECT channel_json FROM dm_conversations WHERE user_id = ?", [user_id]
            ).fetchone()
            if stored:
                channel = json.loads(stored[0])
            else:
                channel_id = "D_" + uuid.uuid4().hex
                while channel_id in self._channels:
                    channel_id = "D_" + uuid.uuid4().hex
                channel = {
                    "id": channel_id,
                    "name": "dm-" + user_id,
                    "workspace_id": user["workspace_id"],
                    "is_private": True,
                    "is_im": True,
                    "is_mpim": False,
                    "is_archived": False,
                    "member_ids": [self.config.agent_user_id, user_id],
                    "recipient_user_id": user_id,
                    "topic": "",
                    "purpose": "",
                    "created_ts": _micros_to_timestamp(self._clock_ns() // 1000),
                }
                self._overlay.execute(
                    "INSERT INTO dm_conversations VALUES (?, ?)",
                    [user_id, _json(channel)],
                )
            self._channels[channel["id"]] = channel
            self._dm_channels.add(channel["id"])
            result = self._write_message(channel, text, thread_ts=None)
            result["destination"] = {
                "kind": "dm",
                "channel_id": channel["id"],
                "user_id": user_id,
            }
            return result

    def _write_message(
        self, channel: Mapping[str, Any], text: str, *, thread_ts: str | None
    ) -> dict[str, Any]:
        clean_text = self._validate_message_text(text)
        self._overlay.execute("BEGIN IMMEDIATE")
        try:
            if self.config.benchmark_mode:
                row = self._duck.execute(
                    "SELECT ts FROM effective_messages "
                    "ORDER BY CAST(ts AS DECIMAL(20,6)) DESC LIMIT 1"
                ).fetchone()
                latest_overlay, latest_baseline = None, row[0] if row else None
            else:
                latest_overlay = self._overlay.execute(
                    "SELECT MAX(ts) FROM messages"
                ).fetchone()[0]
                latest_baseline = self._duck.execute(
                    "SELECT MAX(ts) FROM baseline_messages"
                ).fetchone()[0]
            latest = max(
                (
                    _timestamp_to_micros(value)
                    for value in (latest_overlay, latest_baseline)
                    if value
                ),
                default=0,
            )
            timestamp = _micros_to_timestamp(max(self._clock_ns() // 1_000, latest + 1))
            values = {
                "channel_id": channel["id"],
                "ts": timestamp,
                "thread_ts": thread_ts,
                "user_id": self.config.agent_user_id,
                "text": clean_text,
                "blocks_json": "[]",
                "subtype": None,
                "edited_ts": None,
                "client_msg_id": str(self._id_factory()),
                "event_id": None,
                "operation": "upsert",
                "created_at": _timestamp_iso(timestamp),
            }
            self._overlay.execute(
                f"INSERT INTO messages({','.join(OVERLAY_MESSAGE_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in OVERLAY_MESSAGE_COLUMNS)})",
                tuple(values[column] for column in OVERLAY_MESSAGE_COLUMNS),
            )
            self._overlay.execute("COMMIT")
        except Exception:
            self._overlay.execute("ROLLBACK")
            raise

        self._duck.execute(
            f"INSERT INTO overlay_messages VALUES ({','.join('?' for _ in OVERLAY_MESSAGE_COLUMNS)})",
            [values[column] for column in OVERLAY_MESSAGE_COLUMNS],
        )
        message = self._effective_message(channel["id"], timestamp)
        return {
            "ok": True,
            "object": "message",
            "message": self._hydrate_messages([message])[0],
        }

    def _validate_message_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise SlackToolError("invalid_arguments", "text must be a string")
        if not text.strip():
            raise SlackToolError("no_text", "Slack message text cannot be empty")
        if len(text) > self.config.max_message_characters:
            raise SlackToolError(
                "msg_too_long",
                f"Slack message text exceeds {self.config.max_message_characters} characters",
            )
        return text

    def _parse_search(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise SlackToolError("invalid_arguments", "query cannot be empty")
        if len(query) > self.config.max_search_query_characters:
            raise SlackToolError("invalid_arguments", "query is too long")
        try:
            tokens = shlex.split(query)
        except ValueError as error:
            raise SlackToolError(
                "invalid_query", f"Invalid search query: {error}"
            ) from error

        parsed: dict[str, Any] = {
            "terms": [],
            "user_id": None,
            "channel_id": None,
            "after": None,
            "before": None,
            "has": set(),
        }
        for token in tokens:
            key, separator, raw = token.partition(":")
            if separator and key.lower() in {
                "from",
                "in",
                "after",
                "before",
                "on",
                "has",
            }:
                value = raw.strip()
                if not value:
                    raise SlackToolError("invalid_query", f"{key}: needs a value")
                modifier = key.lower()
                if modifier == "from":
                    parsed["user_id"] = self._resolve_user(value)["id"]
                elif modifier == "in":
                    resolve = (
                        self._require_readable_channel
                        if self.config.benchmark_mode
                        else self._resolve_channel
                    )
                    parsed["channel_id"] = resolve(value)["id"]
                elif modifier in {"after", "before", "on"}:
                    start, end = self._date_bounds(value)
                    if modifier == "after":
                        parsed["after"] = start
                    elif modifier == "before":
                        parsed["before"] = start
                    else:
                        parsed["after"] = start
                        parsed["before"] = end
                else:
                    if value not in {"link", "reaction", "file"}:
                        raise SlackToolError(
                            "invalid_query", "has: supports link, reaction, or file"
                        )
                    parsed["has"].add(value)
            else:
                parsed["terms"].append(token.lower())
        return parsed

    def _date_bounds(self, raw: str) -> tuple[str, str]:
        try:
            parsed = date.fromisoformat(raw)
        except ValueError as error:
            raise SlackToolError(
                "invalid_query", f"Search date must use YYYY-MM-DD: {raw}"
            ) from error
        start = datetime.combine(parsed, datetime_time.min, tzinfo=timezone.utc)
        try:
            end = start + timedelta(days=1)
        except OverflowError as error:
            raise SlackToolError(
                "invalid_query", "Search date is outside supported range"
            ) from error
        return _micros_to_timestamp(
            int(start.timestamp() * 1_000_000)
        ), _micros_to_timestamp(int(end.timestamp() * 1_000_000))

    def _search_where(
        self, parsed: Mapping[str, Any], accessible: set[str]
    ) -> tuple[str, list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        placeholders = ",".join("?" for _ in accessible)
        conditions.append(f"m.channel_id IN ({placeholders})")
        params.extend(sorted(accessible))
        for term in parsed["terms"]:
            conditions.append("LOWER(m.text) LIKE ? ESCAPE '$'")
            escaped = term.replace("$", "$$").replace("%", "$%").replace("_", "$_")
            params.append(f"%{escaped}%")
        if parsed["user_id"]:
            conditions.append("m.user_id = ?")
            params.append(parsed["user_id"])
        if parsed["after"]:
            conditions.append("CAST(m.ts AS DECIMAL(20,6)) >= CAST(? AS DECIMAL(20,6))")
            params.append(parsed["after"])
        if parsed["before"]:
            conditions.append("CAST(m.ts AS DECIMAL(20,6)) < CAST(? AS DECIMAL(20,6))")
            params.append(parsed["before"])
        if "link" in parsed["has"]:
            conditions.append("regexp_matches(m.text, 'https?://')")
        if "reaction" in parsed["has"]:
            conditions.append(
                "EXISTS (SELECT 1 FROM effective_reactions r "
                "WHERE r.channel_id = m.channel_id AND r.message_ts = m.ts)"
            )
        if "file" in parsed["has"]:
            conditions.append(
                "EXISTS (SELECT 1 FROM baseline_files f "
                "WHERE f.channel_id = m.channel_id AND f.message_ts = m.ts)"
            )
        return " AND ".join(conditions), params

    @staticmethod
    def _score_sql(terms: Sequence[str]) -> tuple[str, list[Any]]:
        if not terms:
            return "1.0", []
        parts: list[str] = []
        params: list[Any] = []
        for term in terms:
            parts.append(
                "(LENGTH(LOWER(m.text)) - LENGTH(REPLACE(LOWER(m.text), ?, ''))) "
                "/ GREATEST(LENGTH(?), 1)"
            )
            params.extend([term, term])
        return " + ".join(parts), params

    def _hydrate_messages(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        keys = [(str(row["channel_id"]), str(row["ts"])) for row in rows]
        predicate = " OR ".join("(channel_id = ? AND message_ts = ?)" for _ in keys)
        key_params = [value for key in keys for value in key]
        related_limit = (
            f" LIMIT {self.config.max_related_rows + 1}"
            if self.config.benchmark_mode
            else ""
        )
        reaction_rows = self._rows(
            f"SELECT * FROM effective_reactions WHERE {predicate} "
            "ORDER BY created_ts, name, user_id" + related_limit,
            key_params,
        )
        file_rows = self._rows(
            f"SELECT * FROM baseline_files WHERE {predicate} ORDER BY id"
            + related_limit,
            key_params,
        )
        if (
            self.config.benchmark_mode
            and max(len(reaction_rows), len(file_rows)) > self.config.max_related_rows
        ):
            raise SlackToolError(
                "result_too_large", "Message attachments exceed the result row limit"
            )

        reactions: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for reaction in reaction_rows:
            reactions[(reaction["channel_id"], reaction["message_ts"])][
                reaction["name"]
            ].append(reaction["user_id"])
        files: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for file in file_rows:
            files[(file["channel_id"], file["message_ts"])].append(file)

        hydrated: list[dict[str, Any]] = []
        for row in rows:
            channel = self._channels[row["channel_id"]]
            author = self._user_result(self._users.get(row["user_id"]))
            key = (row["channel_id"], row["ts"])
            message_reactions = [
                {"name": name, "count": len(user_ids), "users": user_ids}
                for name, user_ids in reactions[key].items()
            ]
            hydrated.append(
                {
                    "type": "message",
                    "channel_id": row["channel_id"],
                    "channel_name": channel["name"],
                    "ts": row["ts"],
                    "datetime": _timestamp_iso(row["ts"]),
                    "thread_ts": row.get("thread_ts"),
                    "is_reply": bool(row.get("thread_ts")),
                    "user_id": row.get("user_id"),
                    "author": author,
                    "text": row.get("text", ""),
                    "blocks": _parse_json(row.get("blocks_json"), []),
                    "subtype": row.get("subtype"),
                    "edited_ts": row.get("edited_ts"),
                    "reactions": message_reactions,
                    "files": files[key],
                    "permalink": f"slack://channel/{row['channel_id']}/message/{row['ts']}",
                    **(
                        {}
                        if self.config.benchmark_mode
                        else {"source": row.get("source")}
                    ),
                }
            )
        return hydrated

    def _effective_message(
        self, channel_id: str, timestamp: str
    ) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT * FROM effective_messages WHERE channel_id = ? AND ts = ?",
            [channel_id, timestamp],
        )
        return rows[0] if rows else None

    def _resolve_user(self, value: str) -> dict[str, Any]:
        needle = value.strip().removeprefix("@").strip("<>").lower()
        if needle.startswith("@"):
            needle = needle[1:]
        for user in self._users.values():
            candidates = {
                str(user.get(field) or "").lower()
                for field in ("id", "name", "display_name", "real_name", "email")
            }
            if needle in candidates:
                return user
        raise SlackToolError("user_not_found", f"Slack user was not found: {value}")

    def _resolve_channel(self, value: str) -> dict[str, Any]:
        needle = value.strip().removeprefix("#").lower()
        if needle in {channel_id.lower() for channel_id in self._channels}:
            return next(
                channel
                for channel_id, channel in self._channels.items()
                if channel_id.lower() == needle
            )
        for channel in self._channels.values():
            if channel["name"].lower() == needle:
                return channel
        raise SlackToolError(
            "channel_not_found", f"Slack channel was not found: {value}"
        )

    def _require_readable_channel(self, value: str) -> dict[str, Any]:
        channel = self._resolve_channel(value)
        if not self._can_read_channel(channel):
            raise SlackToolError("channel_not_found", "Slack channel was not found")
        return channel

    def _require_writable_channel(self, value: str) -> dict[str, Any]:
        channel = self._require_readable_channel(value)
        if channel["is_archived"]:
            raise SlackToolError(
                "is_archived", "Cannot post to an archived Slack channel"
            )
        if (
            not self.config.benchmark_mode
            and self.config.agent_user_id not in channel["member_ids"]
        ):
            raise SlackToolError(
                "not_in_channel", "Agent is not a member of this channel"
            )
        return channel

    def _can_read_channel(self, channel: Mapping[str, Any]) -> bool:
        if self.config.benchmark_mode:
            if channel["is_im"] or channel["is_mpim"]:
                return channel["id"] in self._dm_channels
            return True
        if (
            not channel["is_private"]
            and not channel["is_im"]
            and not channel["is_mpim"]
        ):
            return True
        return self.config.agent_user_id in channel["member_ids"]

    @staticmethod
    def _channel_kind(channel: Mapping[str, Any]) -> str:
        if channel["is_im"]:
            return "im"
        if channel["is_mpim"]:
            return "mpim"
        return "private_channel" if channel["is_private"] else "public_channel"

    def _channel_types(self, value: str | Sequence[str]) -> set[str]:
        raw = value.split(",") if isinstance(value, str) else list(value)
        types = {item.strip() for item in raw if item.strip()}
        supported = {"public_channel", "private_channel", "im", "mpim"}
        unknown = types - supported
        if not types or unknown:
            raise SlackToolError(
                "invalid_arguments",
                "types must contain public_channel, private_channel, im, or mpim",
            )
        return types

    def _channel_result(self, channel: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": channel["id"],
            "name": channel["name"],
            "workspace_id": channel["workspace_id"],
            "type": self._channel_kind(channel),
            "is_private": bool(channel["is_private"]),
            "is_archived": bool(channel["is_archived"]),
            "is_member": self.config.agent_user_id in channel["member_ids"],
            "topic": channel.get("topic") or "",
            "purpose": channel.get("purpose") or "",
            "created_ts": channel.get("created_ts"),
            "member_count": len(channel["member_ids"]),
            **(
                {"recipient_user_id": channel["recipient_user_id"]}
                if "recipient_user_id" in channel
                else {}
            ),
        }

    @staticmethod
    def _user_result(user: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if user is None:
            return None
        return {
            "id": user["id"],
            "name": user["name"],
            "real_name": user.get("real_name") or "",
            "display_name": user.get("display_name") or "",
            "title": user.get("title") or "",
            "timezone": user.get("timezone") or "",
            "is_bot": bool(user.get("is_bot")),
            "is_deleted": bool(user.get("is_deleted")),
        }

    def _page_size(self, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise SlackToolError("invalid_arguments", "page_size must be positive")
        return min(value, self.config.max_page_size)

    @staticmethod
    def _scope(*values: Any) -> str:
        return hashlib.sha256(_json(values).encode()).hexdigest()[:16]

    @staticmethod
    def _encode_cursor(offset: int, scope: str) -> str:
        payload = _json({"v": 1, "offset": offset, "scope": scope}).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def _cursor_offset(cursor: str | None, scope: str) -> int:
        if not cursor:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            if payload.get("v") != 1 or payload.get("scope") != scope:
                raise ValueError
            offset = int(payload["offset"])
            if offset < 0:
                raise ValueError
            return offset
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SlackToolError(
                "invalid_cursor", "Slack cursor is invalid for this request"
            ) from error

    def _rows(self, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        cursor = self._duck.execute(query, list(params))
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


_DEFAULT_TOOLS: SlackTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> SlackTools:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_TOOLS is None:
                _DEFAULT_TOOLS = SlackTools()
    return _DEFAULT_TOOLS
