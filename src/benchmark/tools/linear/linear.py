"""Local Linear benchmark wrappers backed by two SQLite databases."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Self

import tomllib

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
LINEAR_UUID_NAMESPACE = uuid.UUID("4adabfe2-b2b2-4d25-90e4-6c65ad35931f")
_UNSET = object()


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
DEFAULT_BASELINE_DIR = DATA_DIR / "linear"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "linear"


class LinearToolError(ValueError):
    """A predictable error that can be returned to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class LinearConfig:
    """Paths, identity, limits, and storage settings for Linear."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    agent_user_id: str = "U_AGENT"
    max_page_size: int = 100
    max_title_characters: int = 255
    max_description_characters: int = 100_000
    max_comment_characters: int = 10_000
    max_search_query_characters: int = 1_000
    max_detail_comments: int = 200
    max_estimate: int = 100
    sqlite_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> LinearConfig:
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_LINEAR_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_LINEAR_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            agent_user_id=os.getenv(
                "BENCHMARK_LINEAR_AGENT_USER_ID", cls.agent_user_id
            ),
            max_page_size=int(os.getenv("LINEAR_MAX_PAGE_SIZE", cls.max_page_size)),
            max_title_characters=int(
                os.getenv("LINEAR_MAX_TITLE_CHARACTERS", cls.max_title_characters)
            ),
            max_description_characters=int(
                os.getenv(
                    "LINEAR_MAX_DESCRIPTION_CHARACTERS",
                    cls.max_description_characters,
                )
            ),
            max_comment_characters=int(
                os.getenv("LINEAR_MAX_COMMENT_CHARACTERS", cls.max_comment_characters)
            ),
            max_search_query_characters=int(
                os.getenv(
                    "LINEAR_MAX_SEARCH_QUERY_CHARACTERS",
                    cls.max_search_query_characters,
                )
            ),
            max_detail_comments=int(
                os.getenv("LINEAR_MAX_DETAIL_COMMENTS", cls.max_detail_comments)
            ),
            max_estimate=int(os.getenv("LINEAR_MAX_ESTIMATE", cls.max_estimate)),
            sqlite_timeout_seconds=float(
                os.getenv("LINEAR_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
        )

    def validate(self) -> None:
        for name in (
            "max_page_size",
            "max_title_characters",
            "max_description_characters",
            "max_comment_characters",
            "max_search_query_characters",
            "max_detail_comments",
            "max_estimate",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")


OVERLAY_SQL = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    number INTEGER NOT NULL,
    identifier TEXT NOT NULL UNIQUE COLLATE NOCASE,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    project_id TEXT,
    state_id TEXT NOT NULL,
    assignee_id TEXT,
    creator_id TEXT NOT NULL,
    priority INTEGER NOT NULL,
    estimate INTEGER,
    due_date TEXT,
    archived INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX IF NOT EXISTS issues_team_order_idx
    ON issues(team_id, updated_at DESC, identifier);
CREATE INDEX IF NOT EXISTS issues_project_idx ON issues(project_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS issue_labels (
    issue_id TEXT NOT NULL,
    label_id TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete')),
    PRIMARY KEY (issue_id, label_id)
);
CREATE INDEX IF NOT EXISTS issue_labels_issue_idx ON issue_labels(issue_id, label_id);

CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX IF NOT EXISTS comments_issue_idx ON comments(issue_id, created_at, id);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    related_issue_id TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX IF NOT EXISTS relations_issue_idx ON relations(issue_id, created_at, id);

CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    field TEXT NOT NULL,
    from_value TEXT,
    to_value TEXT,
    created_at TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX IF NOT EXISTS history_issue_idx ON history(issue_id, created_at, id);
"""


BASELINE_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": ("id", "name", "display_name", "email", "title", "active", "created_at"),
    "teams": ("id", "key", "name", "description", "private", "archived", "created_at"),
    "team_members": ("team_id", "user_id", "role"),
    "projects": (
        "id",
        "team_id",
        "name",
        "slug",
        "summary",
        "status",
        "lead_id",
        "start_date",
        "target_date",
        "archived",
        "created_at",
        "updated_at",
    ),
    "workflow_states": ("id", "team_id", "name", "type", "position", "color"),
    "labels": ("id", "team_id", "name", "color", "description"),
    "issues": (
        "id",
        "team_id",
        "number",
        "identifier",
        "title",
        "description",
        "project_id",
        "state_id",
        "assignee_id",
        "creator_id",
        "priority",
        "estimate",
        "due_date",
        "archived",
        "created_at",
        "updated_at",
        "completed_at",
    ),
    "issue_labels": ("issue_id", "label_id"),
    "comments": ("id", "issue_id", "user_id", "body", "created_at", "updated_at"),
    "relations": ("id", "issue_id", "related_issue_id", "type", "created_at"),
    "history": (
        "id",
        "issue_id",
        "actor_id",
        "field",
        "from_value",
        "to_value",
        "created_at",
    ),
}

OVERLAY_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "issues": BASELINE_TABLE_COLUMNS["issues"] + ("operation",),
    "issue_labels": BASELINE_TABLE_COLUMNS["issue_labels"] + ("operation",),
    "comments": BASELINE_TABLE_COLUMNS["comments"] + ("operation",),
    "relations": BASELINE_TABLE_COLUMNS["relations"] + ("operation",),
    "history": BASELINE_TABLE_COLUMNS["history"] + ("operation",),
}

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "users": ("id",),
    "teams": ("id",),
    "team_members": ("team_id", "user_id"),
    "projects": ("id",),
    "workflow_states": ("id",),
    "labels": ("id",),
    "issues": ("id",),
    "issue_labels": ("issue_id", "label_id"),
    "comments": ("id",),
    "relations": ("id",),
    "history": ("id",),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _iso_now(clock_ns: Callable[[], int]) -> str:
    return datetime.fromtimestamp(
        clock_ns() / 1_000_000_000, tz=timezone.utc
    ).isoformat(timespec="microseconds")


def _instant(value: str) -> Fraction:
    """Exact ISO instant, without float or datetime's fractional truncation."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    seconds = (
        parsed.toordinal() * 86_400
        + parsed.hour * 3_600
        + parsed.minute * 60
        + parsed.second
    )
    tail = value[19:]
    fraction = re.match(r"[.,](\d+)", tail)
    result = Fraction(seconds) + (Fraction("0." + fraction[1]) if fraction else 0)
    offset = parsed.utcoffset()
    if offset is not None:
        magnitude = abs(offset)
        zone_start = re.search(r"[+-]", tail)
        zone = tail[zone_start.start() :] if zone_start else ""
        fraction = re.search(r"[.,](\d+)", zone)
        offset_seconds = Fraction(magnitude.days * 86_400 + magnitude.seconds) + (
            Fraction("0." + fraction[1])
            if fraction
            else Fraction(magnitude.microseconds, 1_000_000)
        )
        result -= -offset_seconds if zone.startswith("-") else offset_seconds
    return result


def _compare_instants(left: str, right: str) -> int:
    left_instant, right_instant = _instant(left), _instant(right)
    return (left_instant > right_instant) - (left_instant < right_instant)


def _validate_iso_date(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LinearToolError("invalid_input", f"{field} must be an ISO date or null")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise LinearToolError(
            "invalid_input", f"{field} must be an ISO date"
        ) from error


def _normalized_row(columns: Sequence[str], row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _write_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    column_sql = ",".join(columns)
    connection.executemany(
        f"INSERT INTO {table}({column_sql}) VALUES ({placeholders})",
        [_normalized_row(columns, row) for row in rows],
    )


def build_linear_overlay_from_seed(
    seed_path: Path, overlay_path: Path
) -> dict[str, int]:
    """Create a writable SQLite overlay from a small JSON delta."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise TypeError("Linear overlay seed must be a JSON object")
    for table in OVERLAY_TABLE_COLUMNS:
        if not isinstance(seed.get(table, []), list):
            raise TypeError(f"Linear overlay seed {table} must be a list")

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_path.parent / ".overlay.sqlite.tmp"
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", ("format_version", "1")
        )
        for table, columns in OVERLAY_TABLE_COLUMNS.items():
            rows = sorted(
                seed.get(table, []),
                key=lambda row, table=table: tuple(
                    str(row.get(key, "")) for key in PRIMARY_KEYS[table]
                ),
            )
            for row in rows:
                if row.get("operation", "upsert") not in {"upsert", "delete"}:
                    raise ValueError(f"Linear overlay {table} has an invalid operation")
            normalized = [
                dict(row, operation=row.get("operation", "upsert")) for row in rows
            ]
            _write_rows(connection, table, columns, normalized)
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    if overlay_path.exists():
        overlay_path.unlink()
    temporary.replace(overlay_path)
    return {table: len(seed.get(table, [])) for table in OVERLAY_TABLE_COLUMNS}


ISSUE_COLUMNS = BASELINE_TABLE_COLUMNS["issues"]
ISSUE_COLUMN_SQL = ", ".join(ISSUE_COLUMNS)

EFFECTIVE_ISSUES_CTE = f"""
effective_issues AS (
    SELECT {ISSUE_COLUMN_SQL}, 'overlay' AS source
    FROM main.issues
    WHERE operation = 'upsert'
    UNION ALL
    SELECT {ISSUE_COLUMN_SQL}, 'baseline' AS source
    FROM baseline.issues AS baseline_issue
    WHERE NOT EXISTS (
        SELECT 1 FROM main.issues AS overlay_issue
        WHERE overlay_issue.id = baseline_issue.id
    )
)
"""

EFFECTIVE_LABELS_CTE = """
effective_issue_labels AS (
    SELECT issue_id, label_id, 'overlay' AS source
    FROM main.issue_labels
    WHERE operation = 'upsert'
    UNION ALL
    SELECT issue_id, label_id, 'baseline' AS source
    FROM baseline.issue_labels AS baseline_label
    WHERE NOT EXISTS (
        SELECT 1 FROM main.issue_labels AS overlay_label
        WHERE overlay_label.issue_id = baseline_label.issue_id
          AND overlay_label.label_id = baseline_label.label_id
    )
)
"""


def _effective_id_cte(table: str) -> str:
    columns = ", ".join(BASELINE_TABLE_COLUMNS[table])
    return f"""
effective_{table} AS (
    SELECT {columns}, 'overlay' AS source
    FROM main.{table}
    WHERE operation = 'upsert'
    UNION ALL
    SELECT {columns}, 'baseline' AS source
    FROM baseline.{table} AS baseline_row
    WHERE NOT EXISTS (
        SELECT 1 FROM main.{table} AS overlay_row
        WHERE overlay_row.id = baseline_row.id
    )
)
"""


EFFECTIVE_COMMENTS_CTE = _effective_id_cte("comments")
EFFECTIVE_RELATIONS_CTE = _effective_id_cte("relations")
EFFECTIVE_HISTORY_CTE = _effective_id_cte("history")

PRIORITY_LABELS = {
    0: "No priority",
    1: "Urgent",
    2: "High",
    3: "Medium",
    4: "Low",
}


class LinearTools:
    """Read effective Linear state and write only to the task overlay."""

    def __init__(
        self,
        config: LinearConfig | None = None,
        *,
        clock_ns: Callable[[], int] | None = None,
    ):
        self.config = config or LinearConfig.from_env()
        self.config.validate()
        self._clock_ns = clock_ns or time.time_ns
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

        baseline_path = self.config.baseline_dir / "linear.sqlite"
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing Linear baseline: {baseline_path}")
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Missing Linear overlay: {overlay_path}")

        connection = sqlite3.connect(
            overlay_path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
            uri=True,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.create_collation("LINEAR_INSTANT", _compare_instants)
            connection.execute(
                f"PRAGMA busy_timeout = {int(self.config.sqlite_timeout_seconds * 1000)}"
            )
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                "ATTACH DATABASE ? AS baseline",
                (f"{baseline_path.resolve().as_uri()}?mode=ro&immutable=1",),
            )
            connection.execute("SELECT 1 FROM baseline.metadata LIMIT 1").fetchone()
            connection.execute("SELECT 1 FROM main.metadata LIMIT 1").fetchone()
            if (
                connection.execute(
                    "SELECT 1 FROM baseline.users WHERE id = ?",
                    (self.config.agent_user_id,),
                ).fetchone()
                is None
            ):
                raise ValueError(
                    f"Configured Linear agent does not exist: {self.config.agent_user_id}"
                )
        except Exception:
            connection.close()
            raise
        self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("LinearTools is closed")
        return self._connection

    def linear_search_issues(
        self,
        query: str = "",
        *,
        team_id: str | None = None,
        project_id: str | None = None,
        assignee_id: str | None = None,
        state: str | None = None,
        label_ids: Sequence[str] | None = None,
        include_archived: bool = False,
        page_size: int = 50,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search issues visible to the configured agent."""

        if not isinstance(query, str):
            raise LinearToolError("invalid_input", "query must be a string")
        if len(query) > self.config.max_search_query_characters:
            raise LinearToolError("invalid_input", "query is too long")
        if not isinstance(include_archived, bool):
            raise LinearToolError("invalid_input", "include_archived must be a boolean")
        limit = self._page_size(page_size)
        terms = self._search_terms(query)

        with self._lock:
            team = self._resolve_team(team_id, require_access=True) if team_id else None
            project = self._resolve_project(project_id, None) if project_id else None
            assignee = self._resolve_user(assignee_id) if assignee_id else None
            labels = self._resolve_labels(label_ids, team["id"] if team else None)

            where = [
                "EXISTS (SELECT 1 FROM baseline.team_members access WHERE access.team_id = issue.team_id AND access.user_id = ?)",
            ]
            params: list[Any] = [self.config.agent_user_id]
            if not include_archived:
                where.extend(("issue.archived = 0", "team.archived = 0"))
            if team:
                where.append("issue.team_id = ?")
                params.append(team["id"])
            if project:
                where.append("issue.project_id = ?")
                params.append(project["id"])
            if assignee:
                where.append("issue.assignee_id = ?")
                params.append(assignee["id"])
            if state:
                normalized_state = self._required_string(state, "state")
                where.append(
                    "(state.id = ? OR state.name = ? COLLATE NOCASE OR state.type = ? COLLATE NOCASE)"
                )
                params.extend((normalized_state, normalized_state, normalized_state))
            for term in terms:
                where.append(
                    "LOWER(issue.identifier || ' ' || issue.title || ' ' || issue.description) LIKE ? ESCAPE '$'"
                )
                params.append(f"%{self._like(term.lower())}%")
            for label in labels:
                where.append(
                    "EXISTS (SELECT 1 FROM effective_issue_labels issue_label WHERE issue_label.issue_id = issue.id AND issue_label.label_id = ?)"
                )
                params.append(label["id"])

            scope = self._scope(
                query,
                team["id"] if team else None,
                project["id"] if project else None,
                assignee["id"] if assignee else None,
                state,
                [label["id"] for label in labels],
                include_archived,
                limit,
            )
            offset = self._cursor_offset(start_cursor, scope)
            ctes = f"WITH {EFFECTIVE_ISSUES_CTE}, {EFFECTIVE_LABELS_CTE}"
            from_sql = """
                FROM effective_issues AS issue
                JOIN baseline.teams AS team ON team.id = issue.team_id
                JOIN baseline.workflow_states AS state ON state.id = issue.state_id
            """
            where_sql = " AND ".join(where)
            total = int(
                self.connection.execute(
                    f"{ctes} SELECT COUNT(*) {from_sql} WHERE {where_sql}", params
                ).fetchone()[0]
            )
            rows = self._rows(
                f"""
                {ctes}
                SELECT issue.* {from_sql}
                WHERE {where_sql}
                ORDER BY issue.updated_at COLLATE LINEAR_INSTANT DESC, issue.identifier COLLATE NOCASE
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            )
            issues = [self._issue_result(row, detail=False) for row in rows]
            next_offset = offset + len(issues)
            has_more = next_offset < total
            return {
                "issues": issues,
                "total": total,
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def linear_get_issue(self, issue_id: str) -> dict[str, Any]:
        """Return one visible issue with comments, relations, and history."""

        with self._lock:
            row = self._issue_row(issue_id)
            if row is None:
                raise LinearToolError(
                    "not_found", f"Linear issue not found: {issue_id}"
                )
            return self._issue_result(row, detail=True)

    def linear_create_issue(
        self,
        team_id: str,
        title: str,
        *,
        description: str = "",
        project_id: str | None = None,
        assignee_id: str | None = None,
        state_id: str | None = None,
        priority: int = 0,
        estimate: int | None = None,
        due_date: str | None = None,
        label_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create one issue in a team visible to the configured agent."""

        title = self._text(
            title, "title", self.config.max_title_characters, required=True
        )
        description = self._text(
            description,
            "description",
            self.config.max_description_characters,
            required=False,
        )
        priority = self._priority(priority)
        estimate = self._estimate(estimate)
        due_date = _validate_iso_date(due_date, "due_date")

        with self._lock:
            team = self._resolve_team(team_id, require_access=True)
            if team["archived"]:
                raise LinearToolError(
                    "conflict", "Cannot create an issue in an archived team"
                )
            project = (
                self._resolve_project(project_id, team["id"]) if project_id else None
            )
            if project and project["archived"]:
                raise LinearToolError(
                    "conflict", "Cannot create an issue in an archived project"
                )
            assignee = (
                self._resolve_assignee(assignee_id, team["id"]) if assignee_id else None
            )
            state = (
                self._resolve_state(state_id, team["id"])
                if state_id
                else self._default_state(team["id"])
            )
            labels = self._resolve_labels(label_ids, team["id"])
            now = _iso_now(self._clock_ns)
            issue_id = str(
                uuid.uuid5(LINEAR_UUID_NAMESPACE, f"issue:{team['id']}:{title}:{now}")
            )

            try:
                self.connection.execute("BEGIN IMMEDIATE")
                number = int(
                    self.connection.execute(
                        """
                        SELECT COALESCE(MAX(number), 0) + 1
                        FROM (
                            SELECT number FROM baseline.issues WHERE team_id = ?
                            UNION ALL
                            SELECT number FROM main.issues WHERE team_id = ?
                        )
                        """,
                        (team["id"], team["id"]),
                    ).fetchone()[0]
                )
                identifier = f"{team['key']}-{number}"
                values = {
                    "id": issue_id,
                    "team_id": team["id"],
                    "number": number,
                    "identifier": identifier,
                    "title": title,
                    "description": description,
                    "project_id": project["id"] if project else None,
                    "state_id": state["id"],
                    "assignee_id": assignee["id"] if assignee else None,
                    "creator_id": self.config.agent_user_id,
                    "priority": priority,
                    "estimate": estimate,
                    "due_date": due_date,
                    "archived": 0,
                    "created_at": now,
                    "updated_at": now,
                    "completed_at": now if state["type"] == "completed" else None,
                    "operation": "upsert",
                }
                self._insert_overlay("issues", OVERLAY_TABLE_COLUMNS["issues"], values)
                for label in labels:
                    self._upsert_issue_label(issue_id, label["id"], "upsert")
                self._write_history(issue_id, "created", None, identifier, now)
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise LinearToolError(
                    "conflict", "Linear issue could not be created"
                ) from error
            except Exception:
                self._rollback()
                raise
            row = self._issue_row(issue_id)
            if row is None:
                raise RuntimeError("Created Linear issue is missing")
            return self._issue_result(row, detail=True)

    def linear_update_issue(
        self,
        issue_id: str,
        *,
        title: Any = _UNSET,
        description: Any = _UNSET,
        project_id: Any = _UNSET,
        assignee_id: Any = _UNSET,
        state_id: Any = _UNSET,
        priority: Any = _UNSET,
        estimate: Any = _UNSET,
        due_date: Any = _UNSET,
        label_ids: Any = _UNSET,
    ) -> dict[str, Any]:
        """Update supported fields on one visible issue."""

        requested = {
            "title": title,
            "description": description,
            "project_id": project_id,
            "assignee_id": assignee_id,
            "state_id": state_id,
            "priority": priority,
            "estimate": estimate,
            "due_date": due_date,
            "label_ids": label_ids,
        }
        if all(value is _UNSET for value in requested.values()):
            raise LinearToolError(
                "empty_update", "At least one update field is required"
            )

        with self._lock:
            current = self._issue_row(issue_id)
            if current is None:
                raise LinearToolError(
                    "not_found", f"Linear issue not found: {issue_id}"
                )
            if current["archived"]:
                raise LinearToolError("conflict", "Cannot update an archived issue")
            team_id_value = current["team_id"]

            normalized: dict[str, Any] = {}
            if title is not _UNSET:
                normalized["title"] = self._text(
                    title, "title", self.config.max_title_characters, required=True
                )
            if description is not _UNSET:
                normalized["description"] = self._text(
                    description,
                    "description",
                    self.config.max_description_characters,
                    required=False,
                )
            if project_id is not _UNSET:
                normalized["project_id"] = (
                    self._resolve_project(project_id, team_id_value)["id"]
                    if project_id is not None
                    else None
                )
            if assignee_id is not _UNSET:
                normalized["assignee_id"] = (
                    self._resolve_assignee(assignee_id, team_id_value)["id"]
                    if assignee_id is not None
                    else None
                )
            target_state = None
            if state_id is not _UNSET:
                target_state = self._resolve_state(state_id, team_id_value)
                normalized["state_id"] = target_state["id"]
            if priority is not _UNSET:
                normalized["priority"] = self._priority(priority)
            if estimate is not _UNSET:
                normalized["estimate"] = self._estimate(estimate)
            if due_date is not _UNSET:
                normalized["due_date"] = _validate_iso_date(due_date, "due_date")
            labels = (
                self._resolve_labels(label_ids, team_id_value)
                if label_ids is not _UNSET
                else None
            )

            current_labels = [item["id"] for item in self._issue_labels(current["id"])]
            target_labels = (
                [item["id"] for item in labels]
                if labels is not None
                else current_labels
            )
            changes = {
                field: value
                for field, value in normalized.items()
                if current[field] != value
            }
            labels_changed = current_labels != target_labels
            if not changes and not labels_changed:
                raise LinearToolError(
                    "empty_update", "The update does not change the issue"
                )

            now = _iso_now(self._clock_ns)
            replacement = {column: current[column] for column in ISSUE_COLUMNS}
            replacement.update(changes)
            replacement["updated_at"] = now
            if target_state is not None and "state_id" in changes:
                replacement["completed_at"] = (
                    now if target_state["type"] == "completed" else None
                )
            replacement["operation"] = "upsert"

            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self._insert_overlay(
                    "issues", OVERLAY_TABLE_COLUMNS["issues"], replacement, replace=True
                )
                for field, new_value in changes.items():
                    self._write_history(
                        current["id"], field, current[field], new_value, now
                    )
                if labels_changed:
                    old = set(current_labels)
                    new = set(target_labels)
                    for label_id in sorted(old - new):
                        self._upsert_issue_label(current["id"], label_id, "delete")
                    for label_id in sorted(new):
                        self._upsert_issue_label(current["id"], label_id, "upsert")
                    self._write_history(
                        current["id"],
                        "labels",
                        _json(current_labels),
                        _json(target_labels),
                        now,
                    )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise LinearToolError(
                    "conflict", "Linear issue could not be updated"
                ) from error
            except Exception:
                self._rollback()
                raise

            updated = self._issue_row(current["id"])
            if updated is None:
                raise RuntimeError("Updated Linear issue is missing")
            return self._issue_result(updated, detail=True)

    def linear_create_comment(self, issue_id: str, body: str) -> dict[str, Any]:
        """Add a comment to one visible, active issue."""

        body = self._text(
            body, "body", self.config.max_comment_characters, required=True
        )
        with self._lock:
            issue = self._issue_row(issue_id)
            if issue is None:
                raise LinearToolError(
                    "not_found", f"Linear issue not found: {issue_id}"
                )
            if issue["archived"]:
                raise LinearToolError("conflict", "Cannot comment on an archived issue")
            now = _iso_now(self._clock_ns)
            comment_id = str(
                uuid.uuid5(LINEAR_UUID_NAMESPACE, f"comment:{issue['id']}:{body}:{now}")
            )
            values = {
                "id": comment_id,
                "issue_id": issue["id"],
                "user_id": self.config.agent_user_id,
                "body": body,
                "created_at": now,
                "updated_at": now,
                "operation": "upsert",
            }
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self._insert_overlay(
                    "comments", OVERLAY_TABLE_COLUMNS["comments"], values
                )
                self._write_history(issue["id"], "comment", None, comment_id, now)
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise LinearToolError(
                    "conflict", "Linear comment could not be created"
                ) from error
            except Exception:
                self._rollback()
                raise
            return self._comment_result(values, "overlay")

    def _issue_row(self, value: str) -> dict[str, Any] | None:
        normalized = self._required_string(value, "issue_id")
        rows = self._rows(
            f"""
            WITH {EFFECTIVE_ISSUES_CTE}
            SELECT issue.*
            FROM effective_issues AS issue
            WHERE (issue.id = ? OR issue.identifier = ? COLLATE NOCASE)
              AND EXISTS (
                  SELECT 1 FROM baseline.team_members access
                  WHERE access.team_id = issue.team_id AND access.user_id = ?
              )
            LIMIT 1
            """,
            (normalized, normalized, self.config.agent_user_id),
        )
        return rows[0] if rows else None

    def _issue_result(
        self, issue: Mapping[str, Any], *, detail: bool
    ) -> dict[str, Any]:
        team = self._one(
            "SELECT * FROM baseline.teams WHERE id = ?", (issue["team_id"],)
        )
        state = self._one(
            "SELECT * FROM baseline.workflow_states WHERE id = ?", (issue["state_id"],)
        )
        project = (
            self._one(
                "SELECT * FROM baseline.projects WHERE id = ?", (issue["project_id"],)
            )
            if issue.get("project_id")
            else None
        )
        if project is not None and not self._has_team_access(project["team_id"]):
            project = None
        assignee = (
            self._one(
                "SELECT * FROM baseline.users WHERE id = ?", (issue["assignee_id"],)
            )
            if issue.get("assignee_id")
            else None
        )
        creator = self._one(
            "SELECT * FROM baseline.users WHERE id = ?", (issue["creator_id"],)
        )
        result: dict[str, Any] = {
            "id": issue["id"],
            "identifier": issue["identifier"],
            "number": int(issue["number"]),
            "title": issue["title"],
            "description": issue["description"],
            "team": self._team_result(team),
            "project": self._project_result(project),
            "state": self._state_result(state),
            "assignee": self._user_result(assignee),
            "creator": self._user_result(creator),
            "priority": int(issue["priority"]),
            "priority_label": PRIORITY_LABELS[int(issue["priority"])],
            "estimate": issue["estimate"],
            "due_date": issue["due_date"],
            "archived": bool(issue["archived"]),
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "completed_at": issue["completed_at"],
            "labels": self._issue_labels(issue["id"]),
            "source": issue["source"],
        }
        if detail:
            comments = self._issue_comments(issue["id"])
            result.update(
                {
                    "comments": comments[: self.config.max_detail_comments],
                    "comments_truncated": len(comments)
                    > self.config.max_detail_comments,
                    "relations": self._issue_relations(issue["id"]),
                    "history": self._issue_history(issue["id"]),
                }
            )
        return result

    def _issue_labels(self, issue_id: str) -> list[dict[str, Any]]:
        return self._rows(
            f"""
            WITH {EFFECTIVE_LABELS_CTE}
            SELECT label.id, label.team_id, label.name, label.color, label.description
            FROM effective_issue_labels AS issue_label
            JOIN baseline.labels AS label ON label.id = issue_label.label_id
            WHERE issue_label.issue_id = ?
            ORDER BY label.name COLLATE NOCASE, label.id
            """,
            (issue_id,),
        )

    def _issue_comments(self, issue_id: str) -> list[dict[str, Any]]:
        rows = self._rows(
            f"""
            WITH {EFFECTIVE_COMMENTS_CTE}
            SELECT comment.*
            FROM effective_comments AS comment
            WHERE comment.issue_id = ?
            ORDER BY comment.created_at COLLATE LINEAR_INSTANT, comment.id
            LIMIT ?
            """,
            (issue_id, self.config.max_detail_comments + 1),
        )
        return [self._comment_result(row, row["source"]) for row in rows]

    def _issue_relations(self, issue_id: str) -> list[dict[str, Any]]:
        rows = self._rows(
            f"""
            WITH {EFFECTIVE_ISSUES_CTE}, {EFFECTIVE_RELATIONS_CTE}
            SELECT relation.*, related.identifier AS related_identifier,
                   related.title AS related_title
            FROM effective_relations AS relation
            JOIN effective_issues AS related ON related.id = relation.related_issue_id
            WHERE relation.issue_id = ?
              AND EXISTS (
                  SELECT 1 FROM baseline.team_members access
                  WHERE access.team_id = related.team_id AND access.user_id = ?
              )
            ORDER BY relation.created_at COLLATE LINEAR_INSTANT, relation.id
            """,
            (issue_id, self.config.agent_user_id),
        )
        return [
            {
                "id": row["id"],
                "type": row["type"],
                "related_issue": {
                    "id": row["related_issue_id"],
                    "identifier": row["related_identifier"],
                    "title": row["related_title"],
                },
                "created_at": row["created_at"],
                "source": row["source"],
            }
            for row in rows
        ]

    def _issue_history(self, issue_id: str) -> list[dict[str, Any]]:
        rows = self._rows(
            f"""
            WITH {EFFECTIVE_HISTORY_CTE}
            SELECT history.*, user.name AS actor_name, user.display_name AS actor_display_name,
                   user.email AS actor_email
            FROM effective_history AS history
            JOIN baseline.users AS user ON user.id = history.actor_id
            WHERE history.issue_id = ?
            ORDER BY history.created_at COLLATE LINEAR_INSTANT, history.id
            """,
            (issue_id,),
        )
        return [
            {
                "id": row["id"],
                "field": row["field"],
                "from_value": row["from_value"],
                "to_value": row["to_value"],
                "actor": {
                    "id": row["actor_id"],
                    "name": row["actor_name"],
                    "display_name": row["actor_display_name"],
                    "email": row["actor_email"],
                },
                "created_at": row["created_at"],
                "source": row["source"],
            }
            for row in rows
        ]

    def _comment_result(self, row: Mapping[str, Any], source: str) -> dict[str, Any]:
        user = self._one("SELECT * FROM baseline.users WHERE id = ?", (row["user_id"],))
        return {
            "id": row["id"],
            "issue_id": row["issue_id"],
            "body": row["body"],
            "user": self._user_result(user),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "source": source,
        }

    def _resolve_team(self, value: Any, *, require_access: bool) -> dict[str, Any]:
        normalized = self._required_string(value, "team_id")
        team = self._one(
            "SELECT * FROM baseline.teams WHERE id = ? OR key = ? COLLATE NOCASE",
            (normalized, normalized),
        )
        if team is None:
            raise LinearToolError("invalid_team", f"Unknown Linear team: {normalized}")
        if require_access and not self._has_team_access(team["id"]):
            raise LinearToolError(
                "forbidden", f"Agent cannot access Linear team: {team['key']}"
            )
        return team

    def _resolve_project(self, value: Any, team_id: str | None) -> dict[str, Any]:
        normalized = self._required_string(value, "project_id")
        project = self._one(
            "SELECT * FROM baseline.projects WHERE id = ? OR slug = ? COLLATE NOCASE",
            (normalized, normalized),
        )
        if project is None:
            raise LinearToolError(
                "invalid_project", f"Unknown Linear project: {normalized}"
            )
        if not self._has_team_access(project["team_id"]):
            raise LinearToolError(
                "forbidden", "Agent cannot access this Linear project"
            )
        if team_id is not None and project["team_id"] != team_id:
            raise LinearToolError(
                "invalid_project", "Project belongs to another Linear team"
            )
        return project

    def _resolve_user(self, value: Any) -> dict[str, Any]:
        normalized = self._required_string(value, "assignee_id")
        rows = self._rows(
            """
            SELECT * FROM baseline.users
            WHERE id = ? OR email = ? COLLATE NOCASE OR name = ? COLLATE NOCASE
               OR display_name = ? COLLATE NOCASE
            ORDER BY id
            """,
            (normalized, normalized, normalized, normalized),
        )
        if not rows:
            raise LinearToolError("invalid_user", f"Unknown Linear user: {normalized}")
        if len(rows) > 1:
            raise LinearToolError(
                "invalid_user", f"Ambiguous Linear user: {normalized}"
            )
        return rows[0]

    def _resolve_assignee(self, value: Any, team_id: str) -> dict[str, Any]:
        user = self._resolve_user(value)
        if not user["active"]:
            raise LinearToolError(
                "invalid_user", "Cannot assign an inactive Linear user"
            )
        member = self._one(
            "SELECT 1 AS found FROM baseline.team_members WHERE team_id = ? AND user_id = ?",
            (team_id, user["id"]),
        )
        if member is None:
            raise LinearToolError(
                "invalid_user", "Assignee is not a member of this Linear team"
            )
        return user

    def _resolve_state(self, value: Any, team_id: str) -> dict[str, Any]:
        normalized = self._required_string(value, "state_id")
        states = self._rows(
            """
            SELECT * FROM baseline.workflow_states
            WHERE team_id = ? AND (id = ? OR name = ? COLLATE NOCASE OR type = ? COLLATE NOCASE)
            ORDER BY position, id
            """,
            (team_id, normalized, normalized, normalized),
        )
        if not states:
            raise LinearToolError(
                "invalid_state", f"Unknown state for this team: {normalized}"
            )
        return states[0]

    def _default_state(self, team_id: str) -> dict[str, Any]:
        state = self._one(
            "SELECT * FROM baseline.workflow_states WHERE team_id = ? AND type = 'backlog' ORDER BY position LIMIT 1",
            (team_id,),
        )
        if state is None:
            state = self._one(
                "SELECT * FROM baseline.workflow_states WHERE team_id = ? ORDER BY position LIMIT 1",
                (team_id,),
            )
        if state is None:
            raise LinearToolError("invalid_state", "Linear team has no workflow states")
        return state

    def _resolve_labels(self, values: Any, team_id: str | None) -> list[dict[str, Any]]:
        if values is None or values is _UNSET:
            return []
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise LinearToolError("invalid_input", "label_ids must be a list")
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._required_string(value, "label_id")
            rows = self._rows(
                "SELECT * FROM baseline.labels WHERE id = ? OR name = ? COLLATE NOCASE ORDER BY id",
                (normalized, normalized),
            )
            if team_id is not None:
                rows = [row for row in rows if row["team_id"] in (None, team_id)]
            else:
                rows = [
                    row
                    for row in rows
                    if row["team_id"] is None or self._has_team_access(row["team_id"])
                ]
            if not rows:
                raise LinearToolError(
                    "invalid_label", f"Unknown Linear label: {normalized}"
                )
            if len(rows) > 1:
                raise LinearToolError(
                    "invalid_label", f"Ambiguous Linear label: {normalized}"
                )
            if rows[0]["id"] not in seen:
                resolved.append(rows[0])
                seen.add(rows[0]["id"])
        return sorted(resolved, key=lambda item: (item["name"].lower(), item["id"]))

    def _has_team_access(self, team_id: str) -> bool:
        return (
            self._one(
                "SELECT 1 AS found FROM baseline.team_members WHERE team_id = ? AND user_id = ?",
                (team_id, self.config.agent_user_id),
            )
            is not None
        )

    def _write_history(
        self,
        issue_id: str,
        field: str,
        from_value: Any,
        to_value: Any,
        created_at: str,
    ) -> None:
        history_id = str(
            uuid.uuid5(
                LINEAR_UUID_NAMESPACE,
                f"history:{issue_id}:{field}:{from_value}:{to_value}:{created_at}",
            )
        )
        values = {
            "id": history_id,
            "issue_id": issue_id,
            "actor_id": self.config.agent_user_id,
            "field": field,
            "from_value": None if from_value is None else str(from_value),
            "to_value": None if to_value is None else str(to_value),
            "created_at": created_at,
            "operation": "upsert",
        }
        self._insert_overlay("history", OVERLAY_TABLE_COLUMNS["history"], values)

    def _upsert_issue_label(self, issue_id: str, label_id: str, operation: str) -> None:
        self.connection.execute(
            """
            INSERT INTO main.issue_labels(issue_id, label_id, operation)
            VALUES (?, ?, ?)
            ON CONFLICT(issue_id, label_id) DO UPDATE SET operation = excluded.operation
            """,
            (issue_id, label_id, operation),
        )

    def _insert_overlay(
        self,
        table: str,
        columns: Sequence[str],
        values: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.connection.execute(
            f"{verb} INTO main.{table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(values.get(column) for column in columns),
        )

    def _rollback(self) -> None:
        if self.connection.in_transaction:
            self.connection.execute("ROLLBACK")

    def _rows(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(query, tuple(params)).fetchall()
        ]

    def _one(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        row = self.connection.execute(query, tuple(params)).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _team_result(team: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if team is None:
            return None
        return {
            "id": team["id"],
            "key": team["key"],
            "name": team["name"],
            "private": bool(team["private"]),
            "archived": bool(team["archived"]),
        }

    @staticmethod
    def _project_result(project: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if project is None:
            return None
        return {
            "id": project["id"],
            "name": project["name"],
            "slug": project["slug"],
            "status": project["status"],
            "target_date": project["target_date"],
            "archived": bool(project["archived"]),
        }

    @staticmethod
    def _state_result(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if state is None:
            return None
        return {
            "id": state["id"],
            "name": state["name"],
            "type": state["type"],
            "color": state["color"],
            "position": state["position"],
        }

    @staticmethod
    def _user_result(user: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if user is None:
            return None
        return {
            "id": user["id"],
            "name": user["name"],
            "display_name": user["display_name"],
            "email": user["email"],
            "active": bool(user["active"]),
        }

    def _page_size(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise LinearToolError(
                "invalid_input", "page_size must be a positive integer"
            )
        if value > self.config.max_page_size:
            raise LinearToolError(
                "invalid_input", f"page_size cannot exceed {self.config.max_page_size}"
            )
        return value

    @staticmethod
    def _required_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise LinearToolError(
                "invalid_input", f"{field} must be a non-empty string"
            )
        return value.strip()

    @staticmethod
    def _text(value: Any, field: str, maximum: int, *, required: bool) -> str:
        if not isinstance(value, str):
            raise LinearToolError("invalid_input", f"{field} must be a string")
        normalized = value.strip()
        if required and not normalized:
            raise LinearToolError("invalid_input", f"{field} cannot be empty")
        if len(normalized) > maximum:
            raise LinearToolError(
                "invalid_input", f"{field} cannot exceed {maximum} characters"
            )
        return normalized

    @staticmethod
    def _priority(value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in PRIORITY_LABELS
        ):
            raise LinearToolError(
                "invalid_input", "priority must be an integer from 0 to 4"
            )
        return value

    def _estimate(self, value: Any) -> int | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= self.config.max_estimate
        ):
            raise LinearToolError(
                "invalid_input",
                f"estimate must be an integer from 0 to {self.config.max_estimate}, or null",
            )
        return value

    @staticmethod
    def _search_terms(query: str) -> list[str]:
        try:
            return [term for term in shlex.split(query) if term]
        except ValueError as error:
            raise LinearToolError(
                "invalid_query", f"Invalid Linear search query: {error}"
            ) from error

    @staticmethod
    def _like(value: str) -> str:
        return value.replace("$", "$$").replace("%", "$%").replace("_", "$_")

    @staticmethod
    def _scope(*values: Any) -> str:
        payload = _json(values)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    @staticmethod
    def _encode_cursor(offset: int, scope: str) -> str:
        payload = _json({"offset": offset, "scope": scope}).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def _cursor_offset(cursor: str | None, scope: str) -> int:
        if cursor is None:
            return 0
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
            offset = payload["offset"]
            if (
                payload["scope"] != scope
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
            ):
                raise ValueError
            return offset
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise LinearToolError(
                "invalid_cursor", "Linear cursor is invalid for this request"
            ) from error


_DEFAULT_TOOLS: LinearTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> LinearTools:
    global _DEFAULT_TOOLS
    with _DEFAULT_LOCK:
        if _DEFAULT_TOOLS is None:
            _DEFAULT_TOOLS = LinearTools()
        return _DEFAULT_TOOLS
