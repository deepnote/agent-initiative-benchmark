"""Local PostHog benchmark wrappers backed by Parquet and a SQLite overlay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Self

from benchmark import duckdb_runtime as duckdb


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
POSTHOG_UUID_NAMESPACE = uuid.UUID("38764461-53e6-4e4f-b183-cf20d6d22f33")
SUPPORTED_INSIGHT_TYPES = {
    "trends",
    "funnels",
    "retention",
    "stickiness",
    "lifecycle",
    "paths",
}


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")

    configured: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        configured.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return configured[0], configured[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "posthog"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "posthog"


class PostHogToolError(ValueError):
    """A predictable error that can be returned to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class PostHogConfig:
    """Paths, access, limits, and query settings for PostHog."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    project_id: str = "PH_PROD"
    default_page_size: int = 100
    max_page_size: int = 10_000
    max_selected_properties: int = 50
    max_property_filters: int = 50
    max_person_scan: int = 20_000
    max_insight_events: int = 100_000
    max_query_seconds: float = 10.0
    sqlite_timeout_seconds: float = 30.0
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "256MB"
    duckdb_max_temp_directory_size: str = "1GB"
    max_output_rows: int = 10_000
    max_result_bytes: int = 16_777_216
    snapshot_time: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_POSTHOG_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_POSTHOG_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            project_id=os.getenv("BENCHMARK_POSTHOG_PROJECT_ID", cls.project_id),
            default_page_size=int(
                os.getenv("POSTHOG_DEFAULT_PAGE_SIZE", cls.default_page_size)
            ),
            max_page_size=int(os.getenv("POSTHOG_MAX_PAGE_SIZE", cls.max_page_size)),
            max_selected_properties=int(
                os.getenv(
                    "POSTHOG_MAX_SELECTED_PROPERTIES",
                    cls.max_selected_properties,
                )
            ),
            max_property_filters=int(
                os.getenv("POSTHOG_MAX_PROPERTY_FILTERS", cls.max_property_filters)
            ),
            max_person_scan=int(
                os.getenv("POSTHOG_MAX_PERSON_SCAN", cls.max_person_scan)
            ),
            max_insight_events=int(
                os.getenv("POSTHOG_MAX_INSIGHT_EVENTS", cls.max_insight_events)
            ),
            max_query_seconds=float(
                os.getenv("POSTHOG_MAX_QUERY_SECONDS", cls.max_query_seconds)
            ),
            sqlite_timeout_seconds=float(
                os.getenv("POSTHOG_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
            duckdb_threads=int(os.getenv("POSTHOG_DUCKDB_THREADS", cls.duckdb_threads)),
        )

    def validate(self) -> None:
        for name in (
            "default_page_size",
            "max_page_size",
            "max_selected_properties",
            "max_property_filters",
            "max_person_scan",
            "max_insight_events",
            "duckdb_threads",
            "max_output_rows",
            "max_result_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.default_page_size > self.max_page_size:
            raise ValueError("default_page_size cannot exceed max_page_size")
        if self.max_query_seconds <= 0 or self.sqlite_timeout_seconds <= 0:
            raise ValueError("query and SQLite timeouts must be positive")
        if not self.project_id.strip():
            raise ValueError("project_id cannot be empty")


EVENT_COLUMNS = (
    "event_uuid",
    "event_name",
    "timestamp",
    "distinct_id",
    "person_id",
    "workspace_id",
    "website_id",
    "session_id",
    "source_event_id",
    "properties_json",
    "person_properties_json",
    "workspace_properties_json",
    "website_properties_json",
    "event_family",
    "event_kind",
    "project_id",
)

EVENT_TYPES = (
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
)

PERSON_COLUMNS = (
    "person_id",
    "valid_from",
    "valid_to",
    "name",
    "email",
    "workspace_id",
    "distinct_ids_json",
    "properties_json",
    "created_at",
    "project_id",
)

PERSON_TYPES = tuple("VARCHAR" for _ in PERSON_COLUMNS)

GROUP_COLUMNS = (
    "id",
    "name",
    "workspace_id",
    "properties_json",
    "created_at",
    "project_id",
)

GROUP_TYPES = tuple("VARCHAR" for _ in GROUP_COLUMNS)


OVERLAY_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_uuid TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    distinct_id TEXT NOT NULL,
    person_id TEXT,
    workspace_id TEXT NOT NULL,
    website_id TEXT,
    session_id TEXT,
    source_event_id TEXT NOT NULL,
    properties_json TEXT NOT NULL,
    person_properties_json TEXT NOT NULL,
    workspace_properties_json TEXT NOT NULL,
    website_properties_json TEXT NOT NULL,
    event_family TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    project_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_time_idx ON events(project_id, timestamp, event_uuid);
CREATE INDEX IF NOT EXISTS events_name_idx ON events(project_id, event_name, timestamp);
CREATE INDEX IF NOT EXISTS events_person_idx ON events(project_id, person_id, timestamp);
CREATE INDEX IF NOT EXISTS events_workspace_idx ON events(project_id, workspace_id, timestamp);

CREATE TABLE IF NOT EXISTS person_changes (
    person_id TEXT PRIMARY KEY,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    distinct_ids_json TEXT NOT NULL,
    properties_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX IF NOT EXISTS person_changes_workspace_idx
    ON person_changes(project_id, workspace_id, person_id);

CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    definition_patch_json TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_flags (
    id TEXT PRIMARY KEY,
    definition_patch_json TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    updated_at TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _iso_datetime(value: str, field: str = "timestamp") -> str:
    if not isinstance(value, str) or not value.strip():
        raise PostHogToolError(
            "invalid_input", f"{field} must be an ISO 8601 timestamp"
        )
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as error:
        raise PostHogToolError(
            "invalid_input", f"{field} must be an ISO 8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class PostHogTools:
    """Read effective PostHog state without writing to either data layer."""

    def __init__(
        self,
        config: PostHogConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        acquire_family: Callable[[str], None] | None = None,
    ):
        self.config = config or PostHogConfig.from_env()
        self.config.validate()
        self._monotonic = monotonic
        self._acquire_family = acquire_family
        self._families = set()
        self._lock = threading.RLock()
        self._validate_files_and_access()
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        self._overlay = sqlite3.connect(
            f"{overlay_path.as_uri()}?mode=ro",
            uri=True,
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
        )
        self._overlay.row_factory = sqlite3.Row
        self._overlay.execute("PRAGMA query_only = ON")
        self._duck = duckdb.connect(
            database=":memory:",
            config={
                "memory_limit": self.config.duckdb_memory_limit,
                "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
            },
        )
        self._duck.execute(f"PRAGMA threads={self.config.duckdb_threads}")
        self._create_views()
        self._load_overlay_tables()
        self._insights = (
            None if acquire_family else self._load_effective_json_objects("insights")
        )
        self._feature_flags = (
            None
            if acquire_family
            else self._load_effective_json_objects("feature_flags")
        )

    def _validate_files_and_access(self) -> None:
        required = [
            self.config.baseline_dir / "events",
            self.config.baseline_dir / "person_history.parquet",
            self.config.baseline_dir / "distinct_ids.parquet",
            self.config.baseline_dir / "workspace_groups.parquet",
            self.config.baseline_dir / "website_groups.parquet",
            self.config.baseline_dir / "insights.json",
            self.config.baseline_dir / "feature_flags.json",
            self.config.baseline_dir / "manifest.json",
            self.config.overlay_dir / "overlay.sqlite",
        ]
        if self._acquire_family:
            required = [
                self.config.baseline_dir / "manifest.json",
                self.config.overlay_dir / "overlay.sqlite",
            ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing PostHog fixture data: " + ", ".join(missing)
            )
        manifest = _read_json(self.config.baseline_dir / "manifest.json")
        self.project = manifest.get("project", {})
        data_project = manifest.get("project", {}).get("id")
        if data_project != self.config.project_id:
            raise PostHogToolError(
                "forbidden",
                f"Project {self.config.project_id} is not available to this PostHog identity",
            )

    @staticmethod
    def _sql_path(path: Path) -> str:
        return str(path.resolve()).replace("'", "''")

    def _create_views(self) -> None:
        root = self.config.baseline_dir
        if self._acquire_family:
            for name, columns in (
                ("events", EVENT_COLUMNS),
                ("person_history", PERSON_COLUMNS),
            ):
                self._duck.execute(
                    "CREATE VIEW baseline_"
                    + name
                    + " AS SELECT "
                    + ",".join('CAST(NULL AS VARCHAR) AS "' + c + '"' for c in columns)
                    + " WHERE false"
                )
        else:
            self._event_views(root)
        self._overlay_views()

    def _event_views(self, root):
        self._duck.execute(
            "CREATE OR REPLACE VIEW baseline_events AS SELECT * FROM read_parquet("
            f"'{self._sql_path(root / 'events' / '**' / '*.parquet')}', "
            "union_by_name=true, hive_partitioning=true)"
        )
        self._duck.execute(
            "CREATE OR REPLACE VIEW baseline_person_history AS SELECT * FROM read_parquet("
            f"'{self._sql_path(root / 'person_history.parquet')}')"
        )

    def _overlay_views(self):
        event_schema = ",".join(
            f'"{column}" {column_type}'
            for column, column_type in zip(EVENT_COLUMNS, EVENT_TYPES, strict=True)
        )
        self._duck.execute(
            f"CREATE TABLE overlay_events ({event_schema}, operation VARCHAR, created_at VARCHAR)"
        )
        person_schema = ",".join(
            f'"{column}" {column_type}'
            for column, column_type in zip(PERSON_COLUMNS, PERSON_TYPES, strict=True)
        )
        self._duck.execute(
            f"CREATE TABLE overlay_person_changes ({person_schema}, operation VARCHAR)"
        )
        self._duck.execute(f"CREATE TABLE overlay_person_versions ({person_schema})")
        self._duck.execute(
            """
            CREATE VIEW effective_person_history AS
            SELECT * FROM overlay_person_versions
            UNION ALL
            SELECT b.* FROM baseline_person_history b WHERE NOT EXISTS (
                SELECT 1 FROM overlay_person_versions o
                WHERE o.person_id=b.person_id AND o.valid_from=b.valid_from)
            """
        )
        self._duck.execute(
            """
            CREATE VIEW effective_events AS
            SELECT event_uuid, event_name, timestamp, distinct_id, person_id,
                   workspace_id, website_id, session_id, source_event_id,
                   properties_json, person_properties_json,
                   workspace_properties_json, website_properties_json,
                   event_family, event_kind, project_id, 'overlay' AS source
            FROM overlay_events WHERE operation = 'upsert'
            UNION ALL
            SELECT b.event_uuid, b.event_name, b.timestamp, b.distinct_id, b.person_id,
                   b.workspace_id, b.website_id, b.session_id, b.source_event_id,
                   b.properties_json, b.person_properties_json,
                   b.workspace_properties_json, b.website_properties_json,
                   b.event_family, b.event_kind, b.project_id, 'baseline' AS source
            FROM baseline_events b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_events o WHERE o.event_uuid = b.event_uuid
            )
            """
        )
        active_filter = ""
        if self.config.snapshot_time:
            instant = _iso_datetime(self.config.snapshot_time)
            active_filter = (
                " WHERE valid_from::TIMESTAMPTZ <= '"
                + instant
                + "'::TIMESTAMPTZ AND (valid_to IS NULL OR valid_to::TIMESTAMPTZ > '"
                + instant
                + "'::TIMESTAMPTZ)"
            )
        self._duck.execute(
            """
            CREATE VIEW current_baseline_persons AS
            SELECT * EXCLUDE (person_rank) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY person_id ORDER BY valid_from DESC, person_id
                ) AS person_rank
                FROM effective_person_history
            """
            + active_filter
            + ") WHERE person_rank = 1"
        )
        self._duck.execute(
            """
            CREATE VIEW effective_persons AS
            SELECT person_id, valid_from, valid_to, name, email, workspace_id,
                   distinct_ids_json, properties_json, created_at, project_id,
                   'overlay' AS source
            FROM overlay_person_changes WHERE operation = 'upsert'
            UNION ALL
            SELECT b.person_id, b.valid_from, b.valid_to, b.name, b.email,
                   b.workspace_id, b.distinct_ids_json, b.properties_json,
                   b.created_at, b.project_id, 'baseline' AS source
            FROM current_baseline_persons b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_person_changes o WHERE o.person_id = b.person_id
            )
            """
        )

    def _load_overlay_tables(self) -> None:
        if self._overlay.execute(
            "SELECT 1 FROM sqlite_master WHERE name='person_versions'"
        ).fetchone():
            for row in self._overlay.execute(
                "SELECT " + ",".join(PERSON_COLUMNS) + " FROM person_versions"
            ):
                self._duck.execute(
                    "INSERT INTO overlay_person_versions VALUES ("
                    + ",".join("?" for _ in PERSON_COLUMNS)
                    + ")",
                    tuple(row),
                )
        event_rows = self._overlay.execute(
            f"SELECT {','.join(EVENT_COLUMNS)}, operation, created_at FROM events "
            "ORDER BY timestamp, event_uuid"
        ).fetchall()
        if event_rows:
            columns = len(EVENT_COLUMNS) + 2
            self._duck.executemany(
                f"INSERT INTO overlay_events VALUES ({','.join('?' for _ in range(columns))})",
                [tuple(row) for row in event_rows],
            )
        person_rows = self._overlay.execute(
            f"SELECT {','.join(PERSON_COLUMNS)}, operation FROM person_changes "
            "ORDER BY person_id"
        ).fetchall()
        if person_rows:
            columns = len(PERSON_COLUMNS) + 1
            self._duck.executemany(
                f"INSERT INTO overlay_person_changes VALUES ({','.join('?' for _ in range(columns))})",
                [tuple(row) for row in person_rows],
            )

    def _load_effective_json_objects(self, name: str) -> dict[str, dict[str, Any]]:
        baseline = {}
        for row in _read_json(self.config.baseline_dir / f"{name}.json"):
            # Native flag definitions may inherit the single source project.
            # An explicit row project still takes precedence, including denial.
            project = row.get(
                "project_id",
                self.project.get("id") if name == "feature_flags" else None,
            )
            if project != self.config.project_id:
                continue
            if row["id"] in baseline:
                raise PostHogToolError(
                    "invalid_source", "Duplicate saved object identity"
                )
            baseline[row["id"]] = row
        rows = self._overlay.execute(
            f"SELECT id, definition_patch_json, operation FROM {name} ORDER BY id"
        ).fetchall()
        for row in rows:
            if row["operation"] == "delete":
                baseline.pop(row["id"], None)
                continue
            patch = _parse_json(row["definition_patch_json"], {})
            if patch.pop("_complete_replacement", False):
                baseline[row["id"]] = {**patch, "source": "overlay"}
                continue
            if row["id"] not in baseline:
                raise PostHogToolError(
                    "invalid_overlay",
                    f"Overlay {name} row has no baseline object: {row['id']}",
                )
            baseline[row["id"]] = {**baseline[row["id"]], **patch, "source": "overlay"}
        for value in baseline.values():
            value.setdefault("source", "baseline")
        return baseline

    def _ensure(self, family):
        if not self._acquire_family or family in self._families:
            return
        self._acquire_family(family)
        if family in ("events", "person_history"):
            path = self.config.baseline_dir / (
                "events/**/*.parquet"
                if family == "events"
                else "person_history.parquet"
            )
            cutoff = ""
            if self.config.snapshot_time:
                column = "timestamp" if family == "events" else "valid_from"
                cutoff = (
                    " WHERE "
                    + column
                    + "::TIMESTAMPTZ <= '"
                    + _iso_datetime(self.config.snapshot_time)
                    + "'::TIMESTAMPTZ"
                )
            self._duck.execute(
                "CREATE OR REPLACE VIEW baseline_"
                + family
                + " AS SELECT * FROM read_parquet('"
                + self._sql_path(path)
                + "',union_by_name=true,hive_partitioning=true)"
                + cutoff
            )
        if family in ("insights", "feature_flags"):
            setattr(self, "_" + family, self._load_effective_json_objects(family))
        self._families.add(family)

    def _bounded_rows(self, query, params=()):
        return self._read_rows(query, params, self.config.max_output_rows)

    def _read_rows(self, query, params=(), max_rows=None):
        cursor = self._duck.execute(query, list(params))
        description = cursor.description
        rows, size = [], 0
        while (raw := cursor.fetchone()) is not None:
            row = self._row(raw, description)
            size += len(_json(row).encode())
            if (
                size > self.config.max_result_bytes
                or max_rows is not None
                and len(rows) >= max_rows
            ):
                raise PostHogToolError(
                    "result_limit", "Exact result exceeds the configured output limit"
                )
            rows.append(row)
        return rows

    def close(self) -> None:
        with self._lock:
            self._duck.close()
            self._overlay.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _check_time(self, started_at: float) -> None:
        if self._monotonic() - started_at > self.config.max_query_seconds:
            raise PostHogToolError(
                "query_timeout", "PostHog query exceeded its time limit"
            )

    def _page_size(self, limit: int | None) -> int:
        value = self.config.default_page_size if limit is None else limit
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PostHogToolError("invalid_input", "limit must be a positive integer")
        if value > self.config.max_page_size:
            raise PostHogToolError(
                "invalid_input", f"limit cannot exceed {self.config.max_page_size}"
            )
        return value

    @staticmethod
    def _scope(*parts: Any) -> str:
        return hashlib.sha256(_json(parts).encode()).hexdigest()[:24]

    @staticmethod
    def _encode_cursor(offset: int, scope: str) -> str:
        raw = _json({"offset": offset, "scope": scope}).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _cursor_offset(cursor: str | None, scope: str) -> int:
        if cursor is None:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            offset = payload["offset"]
            if payload["scope"] != scope or isinstance(offset, bool) or int(offset) < 0:
                raise ValueError
            return int(offset)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise PostHogToolError(
                "invalid_cursor", "Cursor is invalid for this query"
            ) from error

    def _filters(
        self,
        filters: Mapping[str, Any] | None,
        column: str,
    ) -> tuple[list[str], list[Any]]:
        if filters is None:
            return [], []
        if not isinstance(filters, Mapping):
            raise PostHogToolError(
                "invalid_input", "Property filters must be an object"
            )
        if len(filters) > self.config.max_property_filters:
            raise PostHogToolError(
                "invalid_input",
                f"Property filters cannot exceed {self.config.max_property_filters}",
            )
        clauses: list[str] = []
        params: list[Any] = []
        for key, expected in sorted(filters.items()):
            if not isinstance(key, str) or not re.fullmatch(
                r"[$A-Za-z_][$A-Za-z0-9_.-]{0,199}", key
            ):
                raise PostHogToolError(
                    "invalid_input",
                    "Property filter names contain unsupported characters",
                )
            values = expected if isinstance(expected, list) else [expected]
            if isinstance(expected, dict):
                operators = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
                if (
                    not expected
                    or set(expected) - operators.keys()
                    or any(type(v) not in (int, float) for v in expected.values())
                ):
                    raise PostHogToolError(
                        "unsupported_insight", "Unsupported property comparison"
                    )
                for operator, value in sorted(expected.items()):
                    clauses.append(
                        f"TRY_CAST(json_extract_string({column}, ?) AS DOUBLE) {operators[operator]} ?"
                    )
                    params.extend([f'$."{key}"', value])
                continue
            if not values or any(isinstance(value, (dict, list)) for value in values):
                raise PostHogToolError(
                    "invalid_input", "Property filter values must be scalars"
                )
            comparisons = []
            for value in values:
                comparisons.append(
                    f"COALESCE(json_extract({column}, ?), 'null'::JSON) = ?::JSON"
                )
                params.extend([f'$."{key}"', _json(value)])
            clauses.append("(" + " OR ".join(comparisons) + ")")
        return clauses, params

    @staticmethod
    def _row(
        row: Sequence[Any], description: Sequence[Sequence[Any]]
    ) -> dict[str, Any]:
        return {
            column[0]: value for column, value in zip(description, row, strict=True)
        }

    def _rows(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self._read_rows(query, params)

    @staticmethod
    def _render_event(
        row: Mapping[str, Any], selected: Sequence[str] | None = None
    ) -> dict[str, Any]:
        properties = _parse_json(row.get("properties_json"), {})
        if selected is not None:
            properties = {key: properties[key] for key in selected if key in properties}
        return {
            "event_uuid": row["event_uuid"],
            "event": row["event_name"],
            "timestamp": row["timestamp"],
            "distinct_id": row["distinct_id"],
            "person_id": row.get("person_id"),
            "workspace_id": row["workspace_id"],
            "website_id": row.get("website_id"),
            "session_id": row.get("session_id"),
            "source_event_id": row["source_event_id"],
            "properties": properties,
            "person_properties": _parse_json(row.get("person_properties_json"), {}),
            "workspace_properties": _parse_json(
                row.get("workspace_properties_json"), {}
            ),
            "website_properties": _parse_json(row.get("website_properties_json"), {}),
            "event_family": row["event_family"],
            "event_kind": row["event_kind"],
            "source": row.get("source", "baseline"),
        }

    def posthog_query_events(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        event_names: Sequence[str] | None = None,
        event_properties: Mapping[str, Any] | None = None,
        person_properties: Mapping[str, Any] | None = None,
        workspace_properties: Mapping[str, Any] | None = None,
        website_properties: Mapping[str, Any] | None = None,
        person_id: str | None = None,
        distinct_id: str | None = None,
        workspace_id: str | None = None,
        website_id: str | None = None,
        session_id: str | None = None,
        selected_properties: Sequence[str] | None = None,
        sort_direction: str = "desc",
        limit: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Query effective raw events with bounded, stable cursor pagination."""

        with self._lock:
            started_at = self._monotonic()
            self._ensure("events")
            page_size = self._page_size(limit)
            if sort_direction not in {"asc", "desc"}:
                raise PostHogToolError(
                    "invalid_input", "sort_direction must be asc or desc"
                )
            start = _iso_datetime(start_time, "start_time") if start_time else None
            end = _iso_datetime(end_time, "end_time") if end_time else None
            if start and end and start >= end:
                raise PostHogToolError(
                    "invalid_input", "start_time must be before end_time"
                )
            names = list(event_names or [])
            if any(not isinstance(name, str) or not name for name in names):
                raise PostHogToolError(
                    "invalid_input", "event_names must contain non-empty strings"
                )
            selected = (
                None if selected_properties is None else list(selected_properties)
            )
            if selected is not None:
                if len(selected) > self.config.max_selected_properties or any(
                    not isinstance(name, str) or not name for name in selected
                ):
                    raise PostHogToolError(
                        "invalid_input",
                        f"selected_properties must contain at most {self.config.max_selected_properties} names",
                    )

            clauses = ["project_id = ?"]
            params: list[Any] = [self.config.project_id]
            if start:
                clauses.append("timestamp >= ?")
                params.append(start)
            if end:
                clauses.append("timestamp < ?")
                params.append(end)
            if names:
                clauses.append(f"event_name IN ({','.join('?' for _ in names)})")
                params.extend(names)
            for column, value in (
                ("person_id", person_id),
                ("distinct_id", distinct_id),
                ("workspace_id", workspace_id),
                ("website_id", website_id),
                ("session_id", session_id),
            ):
                if value is not None:
                    if not isinstance(value, str) or not value:
                        raise PostHogToolError(
                            "invalid_input", f"{column} must be a non-empty string"
                        )
                    clauses.append(f"{column} = ?")
                    params.append(value)
            for filters, column in (
                (event_properties, "properties_json"),
                (person_properties, "person_properties_json"),
                (workspace_properties, "workspace_properties_json"),
                (website_properties, "website_properties_json"),
            ):
                extra_clauses, extra_params = self._filters(filters, column)
                clauses.extend(extra_clauses)
                params.extend(extra_params)
            scope = self._scope(
                "events",
                start,
                end,
                names,
                event_properties,
                person_properties,
                workspace_properties,
                website_properties,
                person_id,
                distinct_id,
                workspace_id,
                website_id,
                session_id,
                selected,
                sort_direction,
            )
            offset = self._cursor_offset(start_cursor, scope)
            where = " AND ".join(clauses)
            total = int(
                self._duck.execute(
                    f"SELECT COUNT(*) FROM effective_events WHERE {where}", params
                ).fetchone()[0]
            )
            direction = sort_direction.upper()
            # Sort only page keys. Carrying every JSON payload through the
            # baseline/overlay merge makes even a one-event page scan expensive.
            page = self._rows(
                f"SELECT event_uuid, timestamp FROM effective_events WHERE {where} "
                f"ORDER BY timestamp {direction}, event_uuid {direction} LIMIT ? OFFSET ?",
                [*params, page_size, offset],
            )
            self._check_time(started_at)
            rows = []
            if page:
                ids = [row["event_uuid"] for row in page]
                rows = self._rows(
                    "SELECT * FROM effective_events WHERE project_id = ? "
                    f"AND event_uuid IN ({','.join('?' for _ in ids)}) "
                    f"ORDER BY timestamp {direction}, event_uuid {direction}",
                    [self.config.project_id, *ids],
                )
            self._check_time(started_at)
            next_offset = offset + len(rows)
            has_more = next_offset < total
            return {
                "ok": True,
                "object": "event_list",
                "events": [self._render_event(row, selected) for row in rows],
                "total": total,
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    @staticmethod
    def _render_person(
        row: Mapping[str, Any], last_activity: str | None
    ) -> dict[str, Any]:
        return {
            "person_id": row["person_id"],
            "name": row["name"],
            "email": row["email"],
            "workspace_id": row["workspace_id"],
            "distinct_ids": _parse_json(row.get("distinct_ids_json"), []),
            "properties": _parse_json(row.get("properties_json"), {}),
            "created_at": row["created_at"],
            "last_activity_at": last_activity,
            "source": row["source"],
        }

    def posthog_query_persons(
        self,
        *,
        person_id: str | None = None,
        distinct_id: str | None = None,
        search: str | None = None,
        property_filters: Mapping[str, Any] | None = None,
        workspace_id: str | None = None,
        limit: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Query current persons, distinct IDs, and recent product activity."""

        with self._lock:
            started_at = self._monotonic()
            self._ensure("person_history")
            self._ensure("events")
            page_size = self._page_size(limit)
            if search is not None and (
                not isinstance(search, str) or len(search) > 500
            ):
                raise PostHogToolError(
                    "invalid_input", "search must be a string of at most 500 characters"
                )
            clauses = ["p.project_id = ?"]
            params: list[Any] = [self.config.project_id]
            if person_id:
                clauses.append("p.person_id = ?")
                params.append(person_id)
            if workspace_id:
                clauses.append("p.workspace_id = ?")
                params.append(workspace_id)
            extra_clauses, extra_params = self._filters(
                property_filters, "p.properties_json"
            )
            clauses.extend(extra_clauses)
            params.extend(extra_params)
            rows = self._rows(
                "SELECT p.*, (SELECT MAX(e.timestamp) FROM effective_events e "
                "WHERE e.person_id = p.person_id AND e.project_id = p.project_id) "
                "AS last_activity_at FROM effective_persons p "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY p.name COLLATE NOCASE, p.person_id "
                "LIMIT ?",
                [*params, self.config.max_person_scan + 1],
            )
            if len(rows) > self.config.max_person_scan:
                raise PostHogToolError(
                    "query_limit_exceeded",
                    "Person query is too broad. Add a workspace or property filter.",
                )
            normalized = (search or "").strip().casefold()
            filtered = []
            for row in rows:
                distinct_ids = _parse_json(row.get("distinct_ids_json"), [])
                if distinct_id and distinct_id not in distinct_ids:
                    continue
                searchable = (
                    f"{row['name']} {row['email']} {' '.join(distinct_ids)}".casefold()
                )
                if normalized and normalized not in searchable:
                    continue
                filtered.append(self._render_person(row, row.get("last_activity_at")))
            scope = self._scope(
                "persons",
                person_id,
                distinct_id,
                search,
                property_filters,
                workspace_id,
            )
            offset = self._cursor_offset(start_cursor, scope)
            page = filtered[offset : offset + page_size]
            next_offset = offset + len(page)
            has_more = next_offset < len(filtered)
            self._check_time(started_at)
            return {
                "ok": True,
                "object": "person_list",
                "persons": page,
                "total": len(filtered),
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def posthog_query_insight(
        self,
        insight_id: str,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        property_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a saved insight and a fresh result over effective events."""

        with self._lock:
            started_at = self._monotonic()
            self._ensure("insights")
            self._ensure("events")
            if not isinstance(insight_id, str) or not insight_id:
                raise PostHogToolError(
                    "invalid_input", "insight_id must be a non-empty string"
                )
            definition = self._insights.get(insight_id)
            if definition is None:
                raise PostHogToolError("not_found", "PostHog insight was not found")
            start = (
                _iso_datetime(date_from, "date_from")
                if date_from
                else definition["date_from"]
            )
            end = (
                _iso_datetime(date_to, "date_to") if date_to else definition["date_to"]
            )
            if start >= end:
                raise PostHogToolError(
                    "invalid_input", "date_from must be before date_to"
                )
            self._filters(property_filters, "properties_json")
            from .analysis import compute

            result = compute(
                self,
                definition,
                start,
                end,
                {**definition.get("filters", {}), **(property_filters or {})},
            )
            cached = (
                date_from is None
                and date_to is None
                and not property_filters
                and definition.get("result") == result
            )
            self._check_time(started_at)
            public_definition = {
                key: value
                for key, value in definition.items()
                if key not in {"result", "project_id"}
            }
            return {
                "ok": True,
                "object": "insight",
                "id": definition["id"],
                "name": definition["name"],
                "type": definition["type"],
                "definition": public_definition,
                "result": result,
                "filters": {
                    **definition.get("filters", {}),
                    **(property_filters or {}),
                },
                "breakdowns": definition.get("breakdowns", []),
                "last_refreshed_at": definition["last_refreshed_at"],
                "cached": cached,
                "saved_result": definition.get("result"),
                "calculation": {
                    "mode": "fresh",
                    "date_from": start,
                    "date_to": end,
                    "counting_unit": (
                        "events"
                        if definition["type"] == "trends"
                        else (
                            "sessions"
                            if definition["type"] == "paths"
                            else "person_id_or_distinct_id"
                        )
                    ),
                    "timezone": self.project.get("timezone", "UTC"),
                    "exact": True,
                },
                "source": definition["source"],
            }

    def posthog_get_event(self, event_uuid):
        from .discovery import rows

        with self._lock:
            records = rows(self, "events", {"event_uuid": event_uuid}, limit=2)
            if not records:
                raise PostHogToolError("not_found", "PostHog event was not found")
            if len(records) != 1:
                raise PostHogToolError("invalid_source", "Duplicate event identity")
            return dict(ok=True, object="event", **self._render_event(records[0]))

    def posthog_list_catalog(self, **arguments):
        from .discovery import catalog

        with self._lock:
            return catalog(self, **arguments)

    def posthog_query_history(self, **arguments):
        from .discovery import history

        with self._lock:
            return history(self, **arguments)

    def posthog_get_feature_flag(self, flag_key_or_id: str) -> dict[str, Any]:
        """Get one effective feature flag by key or ID."""

        with self._lock:
            self._ensure("feature_flags")
            if not isinstance(flag_key_or_id, str) or not flag_key_or_id:
                raise PostHogToolError(
                    "invalid_input", "flag key or ID must be a non-empty string"
                )
            flag = next(
                (
                    value
                    for value in self._feature_flags.values()
                    if value["id"] == flag_key_or_id or value["key"] == flag_key_or_id
                ),
                None,
            )
            if flag is None:
                raise PostHogToolError(
                    "not_found", "PostHog feature flag was not found"
                )
            return {
                "ok": True,
                "object": "feature_flag",
                **{key: value for key, value in flag.items() if key != "project_id"},
            }


_DEFAULT_TOOLS: PostHogTools | None = None
_DEFAULT_TOOLS_LOCK = threading.Lock()


def _default_tools() -> PostHogTools:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        with _DEFAULT_TOOLS_LOCK:
            if _DEFAULT_TOOLS is None:
                _DEFAULT_TOOLS = PostHogTools()
    return _DEFAULT_TOOLS


def posthog_get_feature_flag(flag_key_or_id: str) -> dict[str, Any]:
    return _default_tools().posthog_get_feature_flag(flag_key_or_id)
