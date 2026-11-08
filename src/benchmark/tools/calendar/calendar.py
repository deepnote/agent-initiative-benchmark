"""Local Google Calendar benchmark wrappers backed by two SQLite databases."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import tomllib

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
CALENDAR_UUID_NAMESPACE = uuid.UUID("535be482-37ce-47ab-a4cc-a55987071c30")
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
DEFAULT_BASELINE_DIR = DATA_DIR / "calendar"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "calendar"


class CalendarToolError(ValueError):
    """A stable error that can be returned to a benchmark agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class CalendarConfig:
    """Paths, identity, limits, and lock settings for Calendar."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    agent_email: str = "agent@pagemantle.com"
    expand_recurring: bool = False
    max_page_size: int = 100
    max_search_query_characters: int = 1_000
    max_search_scan_events: int = 5_000
    max_attendees: int = 200
    max_reminders: int = 10
    max_recurrence_rules: int = 20
    max_recurrence_rule_characters: int = 1_000
    max_summary_characters: int = 1_000
    max_description_characters: int = 50_000
    max_location_characters: int = 2_000
    max_event_duration_days: int = 366
    max_free_busy_calendars: int = 50
    max_free_busy_window_days: int = 93
    max_reminder_minutes: int = 40_320
    sqlite_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> CalendarConfig:
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_CALENDAR_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_CALENDAR_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            agent_email=os.getenv("BENCHMARK_CALENDAR_AGENT_EMAIL", cls.agent_email),
            max_page_size=int(os.getenv("CALENDAR_MAX_PAGE_SIZE", cls.max_page_size)),
            max_search_query_characters=int(
                os.getenv(
                    "CALENDAR_MAX_SEARCH_QUERY_CHARACTERS",
                    cls.max_search_query_characters,
                )
            ),
            max_search_scan_events=int(
                os.getenv("CALENDAR_MAX_SEARCH_SCAN_EVENTS", cls.max_search_scan_events)
            ),
            max_attendees=int(os.getenv("CALENDAR_MAX_ATTENDEES", cls.max_attendees)),
            max_reminders=int(os.getenv("CALENDAR_MAX_REMINDERS", cls.max_reminders)),
            max_recurrence_rules=int(
                os.getenv("CALENDAR_MAX_RECURRENCE_RULES", cls.max_recurrence_rules)
            ),
            max_recurrence_rule_characters=int(
                os.getenv(
                    "CALENDAR_MAX_RECURRENCE_RULE_CHARACTERS",
                    cls.max_recurrence_rule_characters,
                )
            ),
            max_summary_characters=int(
                os.getenv("CALENDAR_MAX_SUMMARY_CHARACTERS", cls.max_summary_characters)
            ),
            max_description_characters=int(
                os.getenv(
                    "CALENDAR_MAX_DESCRIPTION_CHARACTERS",
                    cls.max_description_characters,
                )
            ),
            max_location_characters=int(
                os.getenv(
                    "CALENDAR_MAX_LOCATION_CHARACTERS", cls.max_location_characters
                )
            ),
            max_event_duration_days=int(
                os.getenv(
                    "CALENDAR_MAX_EVENT_DURATION_DAYS", cls.max_event_duration_days
                )
            ),
            max_free_busy_calendars=int(
                os.getenv(
                    "CALENDAR_MAX_FREE_BUSY_CALENDARS", cls.max_free_busy_calendars
                )
            ),
            max_free_busy_window_days=int(
                os.getenv(
                    "CALENDAR_MAX_FREE_BUSY_WINDOW_DAYS",
                    cls.max_free_busy_window_days,
                )
            ),
            max_reminder_minutes=int(
                os.getenv("CALENDAR_MAX_REMINDER_MINUTES", cls.max_reminder_minutes)
            ),
            sqlite_timeout_seconds=float(
                os.getenv("CALENDAR_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
        )

    def validate(self) -> None:
        for name in (
            "max_page_size",
            "max_search_query_characters",
            "max_search_scan_events",
            "max_attendees",
            "max_reminders",
            "max_recurrence_rules",
            "max_recurrence_rule_characters",
            "max_summary_characters",
            "max_description_characters",
            "max_location_characters",
            "max_event_duration_days",
            "max_free_busy_calendars",
            "max_free_busy_window_days",
            "max_reminder_minutes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")
        if not _valid_email(self.agent_email):
            raise ValueError("agent_email must be a valid email address")


OVERLAY_SQL = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    calendar_id TEXT NOT NULL,
    id TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    start_utc INTEGER NOT NULL,
    end_utc INTEGER NOT NULL,
    all_day INTEGER NOT NULL,
    timezone TEXT NOT NULL,
    transparency TEXT NOT NULL,
    visibility TEXT NOT NULL,
    creator_email TEXT NOT NULL,
    organizer_email TEXT NOT NULL,
    recurring_event_id TEXT,
    original_start_time TEXT,
    sequence INTEGER NOT NULL,
    conference_url TEXT,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    etag TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT 'upsert'
        CHECK (operation IN ('upsert', 'delete')),
    PRIMARY KEY (calendar_id, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS events_id_idx ON events(id);
CREATE INDEX IF NOT EXISTS events_calendar_time_idx
    ON events(calendar_id, start_utc, end_utc, id);

CREATE TABLE IF NOT EXISTS attendees (
    calendar_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    email TEXT NOT NULL COLLATE NOCASE,
    display_name TEXT NOT NULL DEFAULT '',
    response_status TEXT NOT NULL,
    optional INTEGER NOT NULL,
    organizer INTEGER NOT NULL,
    self INTEGER NOT NULL,
    resource INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (calendar_id, event_id, email)
);
CREATE INDEX IF NOT EXISTS attendees_email_idx
    ON attendees(email, event_id, calendar_id);

CREATE TABLE IF NOT EXISTS recurrence_rules (
    calendar_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    rule TEXT NOT NULL,
    PRIMARY KEY (calendar_id, event_id, position)
);

CREATE TABLE IF NOT EXISTS reminders (
    calendar_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    method TEXT NOT NULL,
    minutes INTEGER NOT NULL,
    PRIMARY KEY (calendar_id, event_id, position)
);
"""


BASELINE_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": ("email", "user_id", "display_name", "timezone", "active"),
    "calendars": (
        "id",
        "summary",
        "description",
        "timezone",
        "owner_email",
        "primary_calendar",
        "hidden",
        "selected",
        "color_id",
    ),
    "calendar_acl": ("calendar_id", "user_email", "role"),
    "events": (
        "calendar_id",
        "id",
        "status",
        "summary",
        "description",
        "location",
        "start_time",
        "end_time",
        "start_utc",
        "end_utc",
        "all_day",
        "timezone",
        "transparency",
        "visibility",
        "creator_email",
        "organizer_email",
        "recurring_event_id",
        "original_start_time",
        "sequence",
        "conference_url",
        "event_type",
        "created_at",
        "updated_at",
        "etag",
    ),
    "attendees": (
        "calendar_id",
        "event_id",
        "position",
        "email",
        "display_name",
        "response_status",
        "optional",
        "organizer",
        "self",
        "resource",
        "comment",
    ),
    "recurrence_rules": ("calendar_id", "event_id", "position", "rule"),
    "reminders": ("calendar_id", "event_id", "position", "method", "minutes"),
}

EVENT_COLUMNS = BASELINE_TABLE_COLUMNS["events"]
OVERLAY_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "events": EVENT_COLUMNS + ("operation",),
    "attendees": BASELINE_TABLE_COLUMNS["attendees"],
    "recurrence_rules": BASELINE_TABLE_COLUMNS["recurrence_rules"],
    "reminders": BASELINE_TABLE_COLUMNS["reminders"],
}
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "users": ("email",),
    "calendars": ("id",),
    "calendar_acl": ("calendar_id", "user_email"),
    "events": ("calendar_id", "id"),
    "attendees": ("calendar_id", "event_id", "email"),
    "recurrence_rules": ("calendar_id", "event_id", "position"),
    "reminders": ("calendar_id", "event_id", "position"),
}

EVENT_COLUMN_SQL = ", ".join(EVENT_COLUMNS)
EFFECTIVE_EVENTS_CTE = f"""
effective_events AS (
    SELECT {EVENT_COLUMN_SQL}, 'overlay' AS source
    FROM main.events
    WHERE operation = 'upsert'
    UNION ALL
    SELECT {EVENT_COLUMN_SQL}, 'baseline' AS source
    FROM baseline.events AS baseline_event
    WHERE NOT EXISTS (
        SELECT 1 FROM main.events AS overlay_event
        WHERE overlay_event.calendar_id = baseline_event.calendar_id
          AND overlay_event.id = baseline_event.id
    )
)
"""

ACCESS_RANKS = {
    "none": 0,
    "freeBusyReader": 1,
    "reader": 2,
    "writer": 3,
    "owner": 4,
}
EVENT_STATUSES = {"confirmed", "tentative", "cancelled"}
TRANSPARENCIES = {"opaque", "transparent"}
VISIBILITIES = {"default", "public", "private", "confidential"}
EVENT_TYPES = {"default", "focusTime", "outOfOffice", "workingLocation", "birthday"}
RESPONSE_STATUSES = {"needsAction", "declined", "tentative", "accepted"}
REMINDER_METHODS = {"email", "popup"}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_email(value: Any) -> bool:
    return isinstance(value, str) and bool(EMAIL_PATTERN.fullmatch(value.strip()))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _now(clock_ns: Callable[[], int]) -> str:
    return datetime.fromtimestamp(
        clock_ns() / 1_000_000_000, tz=timezone.utc
    ).isoformat(timespec="microseconds")


def _timezone(value: Any, field: str = "timezone") -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalendarToolError("invalid_input", f"{field} must be an IANA timezone")
    normalized = value.strip()
    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError as error:
        raise CalendarToolError(
            "invalid_input", f"{field} must be an IANA timezone"
        ) from error
    return normalized


def _event_time(value: Any, event_timezone: str, field: str) -> tuple[str, int, bool]:
    if not isinstance(value, str) or not value.strip():
        raise CalendarToolError(
            "invalid_input", f"{field} must be an ISO date or timezone-aware datetime"
        )
    normalized = value.strip()
    if DATE_PATTERN.fullmatch(normalized):
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError as error:
            raise CalendarToolError("invalid_input", f"{field} is invalid") from error
        local = datetime.combine(
            parsed_date, datetime_time.min, tzinfo=ZoneInfo(event_timezone)
        )
        return parsed_date.isoformat(), int(local.timestamp()), True
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as error:
        raise CalendarToolError(
            "invalid_input", f"{field} must be an ISO date or timezone-aware datetime"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalendarToolError("invalid_input", f"{field} must include a UTC offset")
    return parsed.isoformat(), int(parsed.timestamp()), False


def _instant(value: Any, field: str) -> tuple[str, float]:
    normalized, timestamp, all_day = _event_time(value, "UTC", field)
    if all_day:
        raise CalendarToolError(
            "invalid_input", f"{field} must be a timezone-aware datetime"
        )
    return normalized, datetime.fromisoformat(normalized).timestamp()


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
    connection.executemany(
        f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",
        [_normalized_row(columns, row) for row in rows],
    )


def _validate_seed(seed: Mapping[str, Any]) -> None:
    for table in BASELINE_TABLE_COLUMNS:
        if not isinstance(seed.get(table), list):
            raise TypeError(f"Calendar seed needs a {table} list")

    for table, keys in PRIMARY_KEYS.items():
        values = [tuple(row.get(key) for key in keys) for row in seed[table]]
        if len(values) != len(set(values)):
            raise ValueError(f"Calendar {table} primary keys must be unique")

    user_emails = {row["email"].lower() for row in seed["users"]}
    calendar_ids = {row["id"] for row in seed["calendars"]}
    event_keys = {(row["calendar_id"], row["id"]) for row in seed["events"]}
    event_ids = [row["id"] for row in seed["events"]]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Calendar event IDs must be globally unique")

    for user in seed["users"]:
        if not _valid_email(user.get("email")):
            raise ValueError("Calendar user email is invalid")
        _timezone(user.get("timezone"), "user timezone")
    for calendar in seed["calendars"]:
        if calendar["owner_email"].lower() not in user_emails:
            raise ValueError(f"Calendar {calendar['id']} has an unknown owner")
        _timezone(calendar.get("timezone"), "calendar timezone")
    for access in seed["calendar_acl"]:
        if access["calendar_id"] not in calendar_ids:
            raise ValueError("Calendar ACL points to an unknown calendar")
        if access["role"] not in ACCESS_RANKS:
            raise ValueError("Calendar ACL has an invalid role")
    for event in seed["events"]:
        if event["calendar_id"] not in calendar_ids:
            raise ValueError(f"Event {event['id']} has an unknown calendar")
        event_timezone = _timezone(event.get("timezone"), "event timezone")
        start, start_utc, start_all_day = _event_time(
            event.get("start_time"), event_timezone, "start_time"
        )
        end, end_utc, end_all_day = _event_time(
            event.get("end_time"), event_timezone, "end_time"
        )
        if start_all_day != end_all_day or end_utc <= start_utc:
            raise ValueError(f"Event {event['id']} has an invalid time range")
        if int(event.get("all_day", 0)) != int(start_all_day):
            raise ValueError(f"Event {event['id']} has an inconsistent all_day value")
        if start != event["start_time"] or end != event["end_time"]:
            raise ValueError(f"Event {event['id']} times must be normalized")
        if start_utc != event["start_utc"] or end_utc != event["end_utc"]:
            raise ValueError(f"Event {event['id']} UTC values are inconsistent")
        if event["status"] not in EVENT_STATUSES:
            raise ValueError(f"Event {event['id']} has an invalid status")
        if event["transparency"] not in TRANSPARENCIES:
            raise ValueError(f"Event {event['id']} has invalid transparency")
        if event["visibility"] not in VISIBILITIES:
            raise ValueError(f"Event {event['id']} has invalid visibility")
        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(f"Event {event['id']} has an invalid event type")
        for field in ("creator_email", "organizer_email"):
            if not _valid_email(event.get(field)):
                raise ValueError(f"Event {event['id']} has an invalid {field}")

    for table in ("attendees", "recurrence_rules", "reminders"):
        for row in seed[table]:
            if (row["calendar_id"], row["event_id"]) not in event_keys:
                raise ValueError(f"Calendar {table} points to an unknown event")
    for attendee in seed["attendees"]:
        if not _valid_email(attendee.get("email")):
            raise ValueError("Calendar attendee email is invalid")
        if attendee["response_status"] not in RESPONSE_STATUSES:
            raise ValueError("Calendar attendee response is invalid")
    for reminder in seed["reminders"]:
        if reminder["method"] not in REMINDER_METHODS or reminder["minutes"] < 0:
            raise ValueError("Calendar reminder is invalid")


def build_calendar_overlay_from_seed(
    seed_path: Path, overlay_path: Path
) -> dict[str, int]:
    """Create a writable task overlay from a small JSON delta."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise TypeError("Calendar overlay seed must be a JSON object")
    for table in OVERLAY_TABLE_COLUMNS:
        if not isinstance(seed.get(table, []), list):
            raise TypeError(f"Calendar overlay seed {table} must be a list")

    event_keys = {
        (row["calendar_id"], row["id"]): row.get("operation", "upsert")
        for row in seed.get("events", [])
    }
    for row in seed.get("events", []):
        if row.get("operation", "upsert") not in {"upsert", "delete"}:
            raise ValueError("Calendar overlay event has an invalid operation")
    for table in ("attendees", "recurrence_rules", "reminders"):
        for row in seed.get(table, []):
            key = (row["calendar_id"], row["event_id"])
            if event_keys.get(key) != "upsert":
                raise ValueError(f"Calendar overlay {table} needs an upsert event")

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_path.parent / ".overlay.sqlite.tmp"
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("format_version", "1"),
        )
        for table, columns in OVERLAY_TABLE_COLUMNS.items():
            rows = sorted(
                seed.get(table, []),
                key=lambda row, table=table: tuple(
                    str(row.get(key, "")) for key in PRIMARY_KEYS[table]
                ),
            )
            if table == "events":
                rows = [
                    dict(row, operation=row.get("operation", "upsert")) for row in rows
                ]
            _write_rows(connection, table, columns, rows)
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    if overlay_path.exists():
        overlay_path.unlink()
    temporary.replace(overlay_path)
    return {table: len(seed.get(table, [])) for table in OVERLAY_TABLE_COLUMNS}


class CalendarTools:
    """Read effective Calendar state and write only to the task overlay."""

    def __init__(
        self,
        config: CalendarConfig | None = None,
        *,
        clock_ns: Callable[[], int] | None = None,
    ):
        self.config = config or CalendarConfig.from_env()
        self.config.validate()
        self._clock_ns = clock_ns or time.time_ns
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

        baseline_path = self.config.baseline_dir / "calendar.sqlite"
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing Calendar baseline: {baseline_path}")
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Missing Calendar overlay: {overlay_path}")

        connection = sqlite3.connect(
            overlay_path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
            uri=True,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                f"PRAGMA busy_timeout = {int(self.config.sqlite_timeout_seconds * 1000)}"
            )
            connection.execute("PRAGMA synchronous = FULL")
            baseline_uri = f"{baseline_path.resolve().as_uri()}?mode=ro&immutable=1"
            connection.execute("ATTACH DATABASE ? AS baseline", (baseline_uri,))
            connection.execute("SELECT 1 FROM baseline.metadata LIMIT 1").fetchone()
            connection.execute("SELECT 1 FROM main.metadata LIMIT 1").fetchone()
            if (
                connection.execute(
                    "SELECT 1 FROM baseline.users WHERE email = ? COLLATE NOCASE",
                    (self.config.agent_email,),
                ).fetchone()
                is None
            ):
                raise ValueError(
                    f"Configured Calendar agent does not exist: {self.config.agent_email}"
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
            raise RuntimeError("CalendarTools is closed")
        return self._connection

    def calendar_list_calendars(self, *, page_size=50, start_cursor=None):
        """Discover calendars with at least free/busy access, in stable pages."""
        limit = self._page_size(page_size)
        scope = self._scope("calendar_list_calendars", self.config.agent_email, limit)
        offset = self._cursor_offset(start_cursor, scope)
        with self._lock:
            query = """FROM baseline.calendars c JOIN baseline.calendar_acl a ON a.calendar_id=c.id
                       WHERE a.user_email=? COLLATE NOCASE AND a.role<>'none'"""
            total = self._one(
                "SELECT COUNT(*) AS n " + query, (self.config.agent_email,)
            )["n"]
            rows = self._rows(
                "SELECT c.id, c.summary, c.timezone, a.role AS access_role "
                + query
                + " ORDER BY c.summary COLLATE NOCASE, c.id LIMIT ? OFFSET ?",
                (self.config.agent_email, limit, offset),
            )
            more = offset + len(rows) < total
            return dict(
                calendars=rows,
                total=total,
                has_more=more,
                next_cursor=(
                    self._encode_cursor(offset + len(rows), scope) if more else None
                ),
            )

    def _expanded_window(self, calendars, lower, upper):
        from .recurrence import expand, RecurrenceError

        if lower is None or upper is None:
            raise CalendarToolError(
                "recurrence_window_required",
                "Calendar search requires both time_min and time_max",
            )
        if not calendars:
            return []
        ids = [c["id"] for c in calendars]
        # Masters and all overrides must be loaded even when their stored slot
        # falls outside the query. Rules and children follow the event layer.
        rows = self._rows(
            f"""WITH {EFFECTIVE_EVENTS_CTE}
            SELECT e.*, a.role AS access_role FROM effective_events e
            JOIN baseline.calendar_acl a ON a.calendar_id=e.calendar_id
            WHERE a.user_email=? COLLATE NOCASE AND e.calendar_id IN ({','.join('?' for _ in ids)})
            AND ((e.end_utc+1>? AND e.start_utc<?) OR e.recurring_event_id IS NOT NULL OR
                 EXISTS (SELECT 1 FROM main.recurrence_rules r WHERE e.source='overlay' AND r.calendar_id=e.calendar_id AND r.event_id=e.id) OR
                 EXISTS (SELECT 1 FROM baseline.recurrence_rules r WHERE e.source='baseline' AND r.calendar_id=e.calendar_id AND r.event_id=e.id))
            LIMIT ?""",
            [
                self.config.agent_email,
                *ids,
                lower,
                upper,
                self.config.max_search_scan_events + 1,
            ],
        )
        if len(rows) > self.config.max_search_scan_events:
            raise CalendarToolError(
                "result_too_large",
                "Calendar window exceeds the stored event scan limit",
            )
        try:
            return expand(
                rows,
                self._child_recurrence,
                lower,
                upper,
                cap=self.config.max_search_scan_events,
            )
        except (RecurrenceError, ValueError, OverflowError, KeyError) as error:
            raise CalendarToolError(
                "unsupported_recurrence",
                "Calendar recurrence is unsupported or exceeds its bound",
            ) from error

    def calendar_search_events(
        self,
        query: str = "",
        *,
        calendar_ids: Sequence[str] | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        attendee_email: str | None = None,
        include_cancelled: bool = False,
        page_size: int = 50,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search event details on calendars where the agent has reader access."""

        if not isinstance(query, str):
            raise CalendarToolError("invalid_input", "query must be a string")
        if len(query) > self.config.max_search_query_characters:
            raise CalendarToolError("invalid_input", "query is too long")
        if not isinstance(include_cancelled, bool):
            raise CalendarToolError(
                "invalid_input", "include_cancelled must be a boolean"
            )
        limit = self._page_size(page_size)
        normalized_query = " ".join(query.split()).lower()
        attendee = None
        if attendee_email is not None:
            attendee = self._email(attendee_email, "attendee_email")
        min_value, min_timestamp = (
            _instant(time_min, "time_min") if time_min is not None else (None, None)
        )
        max_value, max_timestamp = (
            _instant(time_max, "time_max") if time_max is not None else (None, None)
        )
        if min_timestamp is not None and max_timestamp is not None:
            self._validate_range(
                min_timestamp,
                max_timestamp,
                self.config.max_event_duration_days,
                "search",
            )

        with self._lock:
            calendars = self._resolve_calendars(calendar_ids, minimum_role="reader")
            where = [
                "acl.user_email = ? COLLATE NOCASE",
                "acl.role IN ('reader', 'writer', 'owner')",
            ]
            params: list[Any] = [self.config.agent_email]
            if calendars:
                where.append(
                    f"event.calendar_id IN ({','.join('?' for _ in calendars)})"
                )
                params.extend(calendar["id"] for calendar in calendars)
            if not include_cancelled:
                where.append("event.status <> 'cancelled'")
            if min_timestamp is not None:
                where.append("event.end_utc > ?")
                params.append(min_timestamp)
            if max_timestamp is not None:
                where.append("event.start_utc < ?")
                params.append(max_timestamp)
            if attendee is not None:
                where.append(
                    """
                    EXISTS (
                        SELECT 1 FROM main.attendees AS overlay_attendee
                        WHERE event.source = 'overlay'
                          AND overlay_attendee.calendar_id = event.calendar_id
                          AND overlay_attendee.event_id = event.id
                          AND overlay_attendee.email = ? COLLATE NOCASE
                        UNION ALL
                        SELECT 1 FROM baseline.attendees AS baseline_attendee
                        WHERE event.source = 'baseline'
                          AND baseline_attendee.calendar_id = event.calendar_id
                          AND baseline_attendee.event_id = event.id
                          AND baseline_attendee.email = ? COLLATE NOCASE
                    )
                    """
                )
                params.extend((attendee, attendee))

            if self.config.expand_recurring:
                rows = self._expanded_window(calendars, min_timestamp, max_timestamp)
                rows = [
                    r for r in rows if include_cancelled or r["status"] != "cancelled"
                ]
                if attendee is not None:
                    rows = [
                        r
                        for r in rows
                        if self._can_see_details(r)
                        and any(
                            a["email"].lower() == attendee
                            for a in self._child_attendees(r)
                        )
                    ]
            else:
                rows = self._rows(
                    f"""
                    WITH {EFFECTIVE_EVENTS_CTE}
                    SELECT event.*, acl.role AS access_role
                    FROM effective_events AS event
                    JOIN baseline.calendar_acl AS acl
                      ON acl.calendar_id = event.calendar_id
                    WHERE {' AND '.join(where)}
                    ORDER BY event.start_utc, event.end_utc, event.calendar_id COLLATE NOCASE, event.id
                    LIMIT ?
                    """,
                    [*params, self.config.max_search_scan_events + 1],
                )
            if len(rows) > self.config.max_search_scan_events:
                raise CalendarToolError(
                    "result_too_large",
                    "Calendar search is too broad. Add a calendar or time filter.",
                )

            filtered: list[dict[str, Any]] = []
            for row in rows:
                can_see_details = self._can_see_details(row)
                if attendee is not None and not can_see_details:
                    continue
                if normalized_query:
                    if not can_see_details:
                        continue
                    searchable = " ".join(
                        (row["summary"], row["description"], row["location"])
                    ).lower()
                    if normalized_query not in searchable:
                        continue
                filtered.append(row)

            scope = self._scope(
                normalized_query,
                [calendar["id"] for calendar in calendars],
                min_value,
                max_value,
                attendee,
                include_cancelled,
                limit,
            )
            offset = self._cursor_offset(start_cursor, scope)
            page = filtered[offset : offset + limit]
            next_offset = offset + len(page)
            has_more = next_offset < len(filtered)
            return {
                "events": [self._event_result(row) for row in page],
                "total": len(filtered),
                "has_more": has_more,
                "next_cursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
            }

    def calendar_get_event(
        self, event_id: str, *, calendar_id: str | None = None
    ) -> dict[str, Any]:
        """Return one event if the agent can read its details calendar."""

        with self._lock:
            row = self._event_row(event_id, calendar_id, minimum_role="reader")
            if row is None:
                raise CalendarToolError(
                    "not_found", f"Calendar event not found: {event_id}"
                )
            return self._event_result(row)

    def calendar_query_free_busy(
        self,
        time_min: str,
        time_max: str,
        *,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return merged busy intervals without exposing event details."""

        normalized_min, min_timestamp = _instant(time_min, "time_min")
        normalized_max, max_timestamp = _instant(time_max, "time_max")
        self._validate_range(
            min_timestamp,
            max_timestamp,
            self.config.max_free_busy_window_days,
            "free_busy",
        )
        with self._lock:
            calendars = self._resolve_calendars(
                calendar_ids, minimum_role="freeBusyReader"
            )
            if len(calendars) > self.config.max_free_busy_calendars:
                raise CalendarToolError(
                    "invalid_input",
                    f"calendar_ids cannot exceed {self.config.max_free_busy_calendars}",
                )
            expanded = (
                self._expanded_window(calendars, min_timestamp, max_timestamp)
                if self.config.expand_recurring
                else None
            )
            result: dict[str, Any] = {}
            for calendar in calendars:
                if self.config.expand_recurring:
                    rows = [
                        r
                        for r in expanded
                        if r["calendar_id"] == calendar["id"]
                        and r["status"] != "cancelled"
                        and r["transparency"] == "opaque"
                    ]
                else:
                    rows = self._rows(
                        f"""
                        WITH {EFFECTIVE_EVENTS_CTE}
                        SELECT start_utc, end_utc
                        FROM effective_events
                        WHERE calendar_id = ?
                          AND status <> 'cancelled'
                          AND transparency = 'opaque'
                          AND end_utc > ?
                          AND start_utc < ?
                        ORDER BY start_utc, end_utc, id
                        """,
                        (calendar["id"], min_timestamp, max_timestamp),
                    )
                intervals = [
                    (
                        max(row["start_utc"], min_timestamp),
                        min(row["end_utc"], max_timestamp),
                    )
                    for row in rows
                ]
                merged: list[list[int]] = []
                for start_utc, end_utc in intervals:
                    if merged and start_utc <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end_utc)
                    else:
                        merged.append([start_utc, end_utc])
                result[calendar["id"]] = {
                    "busy": [
                        {
                            "start": datetime.fromtimestamp(
                                start, timezone.utc
                            ).isoformat(),
                            "end": datetime.fromtimestamp(
                                end, timezone.utc
                            ).isoformat(),
                        }
                        for start, end in merged
                    ]
                }
            return {
                "time_min": normalized_min,
                "time_max": normalized_max,
                "calendars": result,
            }

    def calendar_create_event(
        self,
        calendar_id: str,
        summary: str,
        start: str,
        end: str,
        *,
        description: str = "",
        location: str = "",
        timezone_name: str | None = None,
        attendees: Sequence[str | Mapping[str, Any]] | None = None,
        reminders: Sequence[Mapping[str, Any]] | None = None,
        recurrence: Sequence[str] | None = None,
        transparency: str = "opaque",
        visibility: str = "default",
        status: str = "confirmed",
        conference_url: str | None = None,
        event_type: str = "default",
    ) -> dict[str, Any]:
        """Create one event on a calendar where the agent can write."""

        summary = self._text(
            summary,
            "summary",
            self.config.max_summary_characters,
            required=True,
        )
        description = self._text(
            description,
            "description",
            self.config.max_description_characters,
            required=False,
        )
        location = self._text(
            location,
            "location",
            self.config.max_location_characters,
            required=False,
        )
        status = self._choice(status, "status", EVENT_STATUSES)
        transparency = self._choice(transparency, "transparency", TRANSPARENCIES)
        visibility = self._choice(visibility, "visibility", VISIBILITIES)
        event_type = self._choice(event_type, "event_type", EVENT_TYPES)
        conference_url = self._url(conference_url, "conference_url")

        with self._lock:
            calendar = self._resolve_calendar(calendar_id, minimum_role="writer")
            event_timezone = _timezone(timezone_name or calendar["timezone"])
            times = self._time_pair(start, end, event_timezone)
            normalized_attendees = self._attendees(attendees)
            normalized_reminders = self._reminders(reminders)
            normalized_recurrence = self._recurrence(recurrence)
            now = _now(self._clock_ns)
            event_id = (
                "evt_"
                + uuid.uuid5(
                    CALENDAR_UUID_NAMESPACE,
                    f"event:{calendar['id']}:{summary}:{times['start_time']}:{now}",
                ).hex
            )
            values = {
                "calendar_id": calendar["id"],
                "id": event_id,
                "status": status,
                "summary": summary,
                "description": description,
                "location": location,
                **times,
                "transparency": transparency,
                "visibility": visibility,
                "creator_email": self.config.agent_email,
                "organizer_email": calendar["owner_email"],
                "recurring_event_id": None,
                "original_start_time": None,
                "sequence": 0,
                "conference_url": conference_url,
                "event_type": event_type,
                "created_at": now,
                "updated_at": now,
                "etag": "",
                "operation": "upsert",
            }
            values["etag"] = self._etag(values)
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self._write_event_and_children(
                    values,
                    normalized_attendees,
                    normalized_recurrence,
                    normalized_reminders,
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise CalendarToolError(
                    "conflict", "Calendar event could not be created"
                ) from error
            except Exception:
                self._rollback()
                raise
            created = self._event_row(event_id, calendar["id"], minimum_role="reader")
            if created is None:
                raise RuntimeError("Created Calendar event is missing")
            return self._event_result(created)

    def calendar_update_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        summary: Any = _UNSET,
        description: Any = _UNSET,
        location: Any = _UNSET,
        start: Any = _UNSET,
        end: Any = _UNSET,
        timezone_name: Any = _UNSET,
        attendees: Any = _UNSET,
        reminders: Any = _UNSET,
        recurrence: Any = _UNSET,
        transparency: Any = _UNSET,
        visibility: Any = _UNSET,
        status: Any = _UNSET,
        conference_url: Any = _UNSET,
        event_type: Any = _UNSET,
        deleted: Any = _UNSET,
    ) -> dict[str, Any]:
        """Replace selected event fields or write an event tombstone."""

        requested = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": start,
            "end": end,
            "timezone_name": timezone_name,
            "attendees": attendees,
            "reminders": reminders,
            "recurrence": recurrence,
            "transparency": transparency,
            "visibility": visibility,
            "status": status,
            "conference_url": conference_url,
            "event_type": event_type,
            "deleted": deleted,
        }
        if all(value is _UNSET for value in requested.values()):
            raise CalendarToolError(
                "empty_update", "At least one update field is required"
            )

        with self._lock:
            current = self._event_row(event_id, calendar_id, minimum_role="writer")
            if current is None:
                raise CalendarToolError(
                    "not_found", f"Calendar event not found: {event_id}"
                )
            if deleted is not _UNSET:
                if not isinstance(deleted, bool):
                    raise CalendarToolError(
                        "invalid_input", "deleted must be a boolean"
                    )
                other_values = [
                    value for key, value in requested.items() if key != "deleted"
                ]
                if deleted and any(value is not _UNSET for value in other_values):
                    raise CalendarToolError(
                        "invalid_input", "deleted cannot be combined with other updates"
                    )
                if not deleted:
                    raise CalendarToolError(
                        "empty_update", "deleted=false does not change the event"
                    )
                replacement = {column: current[column] for column in EVENT_COLUMNS}
                replacement["updated_at"] = _now(self._clock_ns)
                replacement["sequence"] = int(current["sequence"]) + 1
                replacement["operation"] = "delete"
                replacement["etag"] = self._etag(replacement)
                try:
                    self.connection.execute("BEGIN IMMEDIATE")
                    self._upsert_overlay_event(replacement)
                    self._delete_overlay_children(current["calendar_id"], current["id"])
                    self.connection.execute("COMMIT")
                except Exception:
                    self._rollback()
                    raise
                return {
                    "id": current["id"],
                    "calendar_id": current["calendar_id"],
                    "deleted": True,
                }

            changes: dict[str, Any] = {}
            if summary is not _UNSET:
                changes["summary"] = self._text(
                    summary,
                    "summary",
                    self.config.max_summary_characters,
                    required=True,
                )
            if description is not _UNSET:
                changes["description"] = self._text(
                    description,
                    "description",
                    self.config.max_description_characters,
                    required=False,
                )
            if location is not _UNSET:
                changes["location"] = self._text(
                    location,
                    "location",
                    self.config.max_location_characters,
                    required=False,
                )
            for field, value, choices in (
                ("status", status, EVENT_STATUSES),
                ("transparency", transparency, TRANSPARENCIES),
                ("visibility", visibility, VISIBILITIES),
                ("event_type", event_type, EVENT_TYPES),
            ):
                if value is not _UNSET:
                    changes[field] = self._choice(value, field, choices)
            if conference_url is not _UNSET:
                changes["conference_url"] = self._url(conference_url, "conference_url")

            event_timezone = (
                _timezone(timezone_name)
                if timezone_name is not _UNSET
                else current["timezone"]
            )
            if start is not _UNSET or end is not _UNSET or timezone_name is not _UNSET:
                changes.update(
                    self._time_pair(
                        current["start_time"] if start is _UNSET else start,
                        current["end_time"] if end is _UNSET else end,
                        event_timezone,
                    )
                )

            current_attendees = self._child_attendees(current)
            current_recurrence = self._child_recurrence(current)
            current_reminders = self._child_reminders(current)
            target_attendees = (
                self._attendees(attendees)
                if attendees is not _UNSET
                else current_attendees
            )
            target_recurrence = (
                self._recurrence(recurrence)
                if recurrence is not _UNSET
                else current_recurrence
            )
            target_reminders = (
                self._reminders(reminders)
                if reminders is not _UNSET
                else current_reminders
            )

            row_changes = {
                field: value
                for field, value in changes.items()
                if current[field] != value
            }
            children_changed = any(
                (
                    target_attendees != current_attendees,
                    target_recurrence != current_recurrence,
                    target_reminders != current_reminders,
                )
            )
            if not row_changes and not children_changed:
                raise CalendarToolError(
                    "empty_update", "The update does not change the event"
                )

            replacement = {column: current[column] for column in EVENT_COLUMNS}
            replacement.update(row_changes)
            replacement["updated_at"] = _now(self._clock_ns)
            replacement["sequence"] = int(current["sequence"]) + 1
            replacement["operation"] = "upsert"
            replacement["etag"] = self._etag(replacement)
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self._write_event_and_children(
                    replacement,
                    target_attendees,
                    target_recurrence,
                    target_reminders,
                )
                self.connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise CalendarToolError(
                    "conflict", "Calendar event could not be updated"
                ) from error
            except Exception:
                self._rollback()
                raise
            updated = self._event_row(
                event_id, current["calendar_id"], minimum_role="reader"
            )
            if updated is None:
                raise RuntimeError("Updated Calendar event is missing")
            return self._event_result(updated)

    def _resolve_calendars(
        self, values: Sequence[str] | None, *, minimum_role: str
    ) -> list[dict[str, Any]]:
        if values is None:
            return self._rows(
                """
                SELECT calendar.*, acl.role AS access_role
                FROM baseline.calendars AS calendar
                JOIN baseline.calendar_acl AS acl ON acl.calendar_id = calendar.id
                WHERE acl.user_email = ? COLLATE NOCASE
                  AND CASE acl.role
                      WHEN 'owner' THEN 4 WHEN 'writer' THEN 3 WHEN 'reader' THEN 2
                      WHEN 'freeBusyReader' THEN 1 ELSE 0 END >= ?
                ORDER BY calendar.primary_calendar DESC, calendar.summary COLLATE NOCASE, calendar.id
                """,
                (self.config.agent_email, ACCESS_RANKS[minimum_role]),
            )
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise CalendarToolError("invalid_input", "calendar_ids must be a list")
        if not values:
            raise CalendarToolError("invalid_input", "calendar_ids cannot be empty")
        if len(values) > self.config.max_free_busy_calendars:
            raise CalendarToolError(
                "invalid_input",
                f"calendar_ids cannot exceed {self.config.max_free_busy_calendars}",
            )
        calendars: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in values:
            calendar = self._resolve_calendar(value, minimum_role=minimum_role)
            key = calendar["id"].lower()
            if key not in seen:
                calendars.append(calendar)
                seen.add(key)
        return calendars

    def _resolve_calendar(self, value: Any, *, minimum_role: str) -> dict[str, Any]:
        calendar_id = self._required_string(value, "calendar_id")
        calendar = self._one(
            """
            SELECT calendar.*, acl.role AS access_role
            FROM baseline.calendars AS calendar
            LEFT JOIN baseline.calendar_acl AS acl
              ON acl.calendar_id = calendar.id AND acl.user_email = ? COLLATE NOCASE
            WHERE calendar.id = ? COLLATE NOCASE
            """,
            (self.config.agent_email, calendar_id),
        )
        if calendar is None:
            raise CalendarToolError(
                "invalid_calendar", f"Unknown calendar: {calendar_id}"
            )
        role = calendar.get("access_role") or "none"
        if ACCESS_RANKS[role] < ACCESS_RANKS[minimum_role]:
            raise CalendarToolError(
                "forbidden", f"Agent cannot {minimum_role} calendar: {calendar['id']}"
            )
        return calendar

    def _event_row(
        self,
        event_id: Any,
        calendar_id: str | None,
        *,
        minimum_role: str,
    ) -> dict[str, Any] | None:
        normalized_id = self._required_string(event_id, "event_id")
        params: list[Any] = [
            self.config.agent_email,
            ACCESS_RANKS[minimum_role],
            normalized_id,
        ]
        calendar_clause = ""
        if calendar_id is not None:
            calendar = self._resolve_calendar(calendar_id, minimum_role=minimum_role)
            calendar_clause = "AND event.calendar_id = ?"
            params.append(calendar["id"])
        return self._one(
            f"""
            WITH {EFFECTIVE_EVENTS_CTE}
            SELECT event.*, acl.role AS access_role
            FROM effective_events AS event
            JOIN baseline.calendar_acl AS acl ON acl.calendar_id = event.calendar_id
            WHERE acl.user_email = ? COLLATE NOCASE
              AND CASE acl.role
                  WHEN 'owner' THEN 4 WHEN 'writer' THEN 3 WHEN 'reader' THEN 2
                  WHEN 'freeBusyReader' THEN 1 ELSE 0 END >= ?
              AND event.id = ?
              {calendar_clause}
            LIMIT 1
            """,
            params,
        )

    def _event_result(self, event: Mapping[str, Any]) -> dict[str, Any]:
        calendar = self._one(
            "SELECT id, summary, timezone FROM baseline.calendars WHERE id = ?",
            (event["calendar_id"],),
        )
        can_see_details = self._can_see_details(event)
        result: dict[str, Any] = {
            "kind": "calendar#event",
            "id": event["id"],
            "calendar_id": event["calendar_id"],
            "calendar_summary": (
                calendar["summary"] if calendar else event["calendar_id"]
            ),
            "status": event["status"],
            "summary": event["summary"] if can_see_details else "Busy",
            "description": event["description"] if can_see_details else "",
            "location": event["location"] if can_see_details else "",
            "start": self._time_result(
                event["start_time"], event["timezone"], bool(event["all_day"])
            ),
            "end": self._time_result(
                event["end_time"], event["timezone"], bool(event["all_day"])
            ),
            "transparency": event["transparency"],
            "visibility": event["visibility"],
            "sequence": int(event["sequence"]),
            "event_type": event["event_type"],
            "created_at": event["created_at"],
            "updated_at": event["updated_at"],
            "etag": event["etag"],
            "source": event["source"],
            "access_role": event["access_role"],
            "details_redacted": not can_see_details,
            "html_link": f"https://calendar.pagemantle.com/event?eid={event['id']}",
        }
        if can_see_details:
            result.update(
                {
                    "creator": self._person(event["creator_email"]),
                    "organizer": self._person(event["organizer_email"]),
                    "attendees": self._child_attendees(event),
                    "recurrence": self._child_recurrence(event),
                    "reminders": self._child_reminders(event),
                    "recurring_event_id": event["recurring_event_id"],
                    "original_start_time": event["original_start_time"],
                    "conference_url": event["conference_url"],
                }
            )
        else:
            result.update(
                {
                    "creator": None,
                    "organizer": None,
                    "attendees": [],
                    "recurrence": [],
                    "reminders": [],
                    "recurring_event_id": None,
                    "original_start_time": None,
                    "conference_url": None,
                }
            )
        if event.get("occurrence"):
            result["expanded_occurrence"] = True
        return result

    def _can_see_details(self, event: Mapping[str, Any]) -> bool:
        if event["visibility"] not in {"private", "confidential"}:
            return True
        if ACCESS_RANKS[event["access_role"]] >= ACCESS_RANKS["writer"]:
            return True
        if self.config.agent_email.lower() in {
            event["creator_email"].lower(),
            event["organizer_email"].lower(),
        }:
            return True
        child_table = (
            "main.attendees" if event["source"] == "overlay" else "baseline.attendees"
        )
        return (
            self._one(
                f"""
            SELECT 1 AS found FROM {child_table}
            WHERE calendar_id = ? AND event_id = ? AND email = ? COLLATE NOCASE
            """,
                (event["calendar_id"], event["id"], self.config.agent_email),
            )
            is not None
        )

    def _child_attendees(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        table = (
            "main.attendees" if event["source"] == "overlay" else "baseline.attendees"
        )
        rows = self._rows(
            f"""
            SELECT * FROM {table}
            WHERE calendar_id = ? AND event_id = ?
            ORDER BY position, email COLLATE NOCASE
            """,
            (event["calendar_id"], event["id"]),
        )
        return [
            {
                "email": row["email"],
                "display_name": row["display_name"],
                "response_status": row["response_status"],
                "optional": bool(row["optional"]),
                "organizer": bool(row["organizer"]),
                "self": bool(row["self"]),
                "resource": bool(row["resource"]),
                "comment": row["comment"],
            }
            for row in rows
        ]

    def _child_recurrence(self, event: Mapping[str, Any]) -> list[str]:
        table = (
            "main.recurrence_rules"
            if event["source"] == "overlay"
            else "baseline.recurrence_rules"
        )
        return [
            row["rule"]
            for row in self._rows(
                f"""
                SELECT rule FROM {table}
                WHERE calendar_id = ? AND event_id = ?
                ORDER BY position
                """,
                (event["calendar_id"], event["id"]),
            )
        ]

    def _child_reminders(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        table = (
            "main.reminders" if event["source"] == "overlay" else "baseline.reminders"
        )
        return [
            {"method": row["method"], "minutes": int(row["minutes"])}
            for row in self._rows(
                f"""
                SELECT method, minutes FROM {table}
                WHERE calendar_id = ? AND event_id = ?
                ORDER BY position
                """,
                (event["calendar_id"], event["id"]),
            )
        ]

    def _write_event_and_children(
        self,
        event: Mapping[str, Any],
        attendees: Sequence[Mapping[str, Any]],
        recurrence: Sequence[str],
        reminders: Sequence[Mapping[str, Any]],
    ) -> None:
        self._upsert_overlay_event(event)
        self._delete_overlay_children(event["calendar_id"], event["id"])
        for position, attendee in enumerate(attendees):
            values = {
                "calendar_id": event["calendar_id"],
                "event_id": event["id"],
                "position": position,
                **attendee,
            }
            self._insert_overlay(
                "attendees", OVERLAY_TABLE_COLUMNS["attendees"], values
            )
        for position, rule in enumerate(recurrence):
            self._insert_overlay(
                "recurrence_rules",
                OVERLAY_TABLE_COLUMNS["recurrence_rules"],
                {
                    "calendar_id": event["calendar_id"],
                    "event_id": event["id"],
                    "position": position,
                    "rule": rule,
                },
            )
        for position, reminder in enumerate(reminders):
            self._insert_overlay(
                "reminders",
                OVERLAY_TABLE_COLUMNS["reminders"],
                {
                    "calendar_id": event["calendar_id"],
                    "event_id": event["id"],
                    "position": position,
                    **reminder,
                },
            )

    def _upsert_overlay_event(self, values: Mapping[str, Any]) -> None:
        columns = OVERLAY_TABLE_COLUMNS["events"]
        self.connection.execute(
            f"INSERT OR REPLACE INTO main.events({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(values.get(column) for column in columns),
        )

    def _delete_overlay_children(self, calendar_id: str, event_id: str) -> None:
        for table in ("attendees", "recurrence_rules", "reminders"):
            self.connection.execute(
                f"DELETE FROM main.{table} WHERE calendar_id = ? AND event_id = ?",
                (calendar_id, event_id),
            )

    def _insert_overlay(
        self, table: str, columns: Sequence[str], values: Mapping[str, Any]
    ) -> None:
        self.connection.execute(
            f"INSERT INTO main.{table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(values.get(column) for column in columns),
        )

    def _time_pair(self, start: Any, end: Any, event_timezone: str) -> dict[str, Any]:
        start_time, start_utc, start_all_day = _event_time(
            start, event_timezone, "start"
        )
        end_time, end_utc, end_all_day = _event_time(end, event_timezone, "end")
        if start_all_day != end_all_day:
            raise CalendarToolError(
                "invalid_time_range", "start and end must both be dates or datetimes"
            )
        self._validate_range(
            start_utc,
            end_utc,
            self.config.max_event_duration_days,
            "event",
        )
        return {
            "start_time": start_time,
            "end_time": end_time,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "all_day": int(start_all_day),
            "timezone": event_timezone,
        }

    @staticmethod
    def _validate_range(start: int, end: int, maximum_days: int, label: str) -> None:
        if end <= start:
            raise CalendarToolError(
                "invalid_time_range", f"{label} end must be after start"
            )
        if end - start > maximum_days * 86_400:
            raise CalendarToolError(
                "invalid_time_range",
                f"{label} range cannot exceed {maximum_days} days",
            )

    def _attendees(
        self, values: Sequence[str | Mapping[str, Any]] | None
    ) -> list[dict[str, Any]]:
        if values is None:
            return []
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise CalendarToolError("invalid_input", "attendees must be a list")
        if len(values) > self.config.max_attendees:
            raise CalendarToolError(
                "invalid_input", f"attendees cannot exceed {self.config.max_attendees}"
            )
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in values:
            if isinstance(item, str):
                item = {"email": item}
            if not isinstance(item, Mapping):
                raise CalendarToolError(
                    "invalid_attendee", "Each attendee must be an email or object"
                )
            email = self._email(item.get("email"), "attendee email")
            key = email.lower()
            if key in seen:
                raise CalendarToolError(
                    "invalid_attendee", f"Duplicate attendee: {email}"
                )
            response = item.get("response_status", "needsAction")
            if response not in RESPONSE_STATUSES:
                raise CalendarToolError(
                    "invalid_attendee", f"Invalid attendee response: {response}"
                )
            display_name = item.get("display_name", "")
            comment = item.get("comment", "")
            if not isinstance(display_name, str) or not isinstance(comment, str):
                raise CalendarToolError(
                    "invalid_attendee",
                    "Attendee display_name and comment must be strings",
                )
            if not display_name.strip():
                user = self._one(
                    "SELECT display_name FROM baseline.users WHERE email = ? COLLATE NOCASE",
                    (email,),
                )
                display_name = user["display_name"] if user else ""
            booleans: dict[str, int] = {}
            for field in ("optional", "organizer", "self", "resource"):
                value = item.get(
                    field, field == "self" and key == self.config.agent_email.lower()
                )
                if not isinstance(value, bool):
                    raise CalendarToolError(
                        "invalid_attendee", f"Attendee {field} must be a boolean"
                    )
                booleans[field] = int(value)
            normalized.append(
                {
                    "email": email,
                    "display_name": display_name.strip(),
                    "response_status": response,
                    **booleans,
                    "comment": comment.strip(),
                }
            )
            seen.add(key)
        return normalized

    def _reminders(
        self, values: Sequence[Mapping[str, Any]] | None
    ) -> list[dict[str, Any]]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise CalendarToolError("invalid_input", "reminders must be a list")
        if len(values) > self.config.max_reminders:
            raise CalendarToolError(
                "invalid_input", f"reminders cannot exceed {self.config.max_reminders}"
            )
        normalized: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, Mapping):
                raise CalendarToolError(
                    "invalid_input", "Each reminder must be an object"
                )
            method = item.get("method")
            minutes = item.get("minutes")
            if method not in REMINDER_METHODS:
                raise CalendarToolError(
                    "invalid_input", "Reminder method must be email or popup"
                )
            if (
                isinstance(minutes, bool)
                or not isinstance(minutes, int)
                or not 0 <= minutes <= self.config.max_reminder_minutes
            ):
                raise CalendarToolError(
                    "invalid_input",
                    f"Reminder minutes must be from 0 to {self.config.max_reminder_minutes}",
                )
            normalized.append({"method": method, "minutes": minutes})
        return normalized

    def _recurrence(self, values: Sequence[str] | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise CalendarToolError("invalid_input", "recurrence must be a list")
        if len(values) > self.config.max_recurrence_rules:
            raise CalendarToolError(
                "invalid_input",
                f"recurrence cannot exceed {self.config.max_recurrence_rules} rules",
            )
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise CalendarToolError(
                    "invalid_recurrence", "Recurrence rules must be non-empty strings"
                )
            rule = value.strip()
            if len(rule) > self.config.max_recurrence_rule_characters:
                raise CalendarToolError(
                    "invalid_recurrence", "Recurrence rule is too long"
                )
            if "\n" in rule or not rule.startswith(("RRULE:", "RDATE:", "EXDATE:")):
                raise CalendarToolError(
                    "invalid_recurrence",
                    "Recurrence rules must start with RRULE:, RDATE:, or EXDATE:",
                )
            normalized.append(rule)
        return normalized

    def _person(self, email: str) -> dict[str, Any]:
        user = self._one(
            "SELECT email, user_id, display_name FROM baseline.users WHERE email = ? COLLATE NOCASE",
            (email,),
        )
        return {
            "email": email,
            "id": user["user_id"] if user else None,
            "display_name": user["display_name"] if user else "",
            "self": email.lower() == self.config.agent_email.lower(),
        }

    @staticmethod
    def _time_result(value: str, event_timezone: str, all_day: bool) -> dict[str, Any]:
        if all_day:
            return {"date": value, "time_zone": event_timezone}
        return {"date_time": value, "time_zone": event_timezone}

    @staticmethod
    def _etag(values: Mapping[str, Any]) -> str:
        payload = _json(
            {key: values.get(key) for key in EVENT_COLUMNS if key != "etag"}
        )
        return '"' + hashlib.sha256(payload.encode()).hexdigest()[:24] + '"'

    def _page_size(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CalendarToolError(
                "invalid_input", "page_size must be a positive integer"
            )
        if value > self.config.max_page_size:
            raise CalendarToolError(
                "invalid_input", f"page_size cannot exceed {self.config.max_page_size}"
            )
        return value

    @staticmethod
    def _required_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CalendarToolError(
                "invalid_input", f"{field} must be a non-empty string"
            )
        return value.strip()

    @staticmethod
    def _text(value: Any, field: str, maximum: int, *, required: bool) -> str:
        if not isinstance(value, str):
            raise CalendarToolError("invalid_input", f"{field} must be a string")
        normalized = value.strip()
        if required and not normalized:
            raise CalendarToolError("invalid_input", f"{field} cannot be empty")
        if len(normalized) > maximum:
            raise CalendarToolError(
                "invalid_input", f"{field} cannot exceed {maximum} characters"
            )
        return normalized

    @staticmethod
    def _choice(value: Any, field: str, choices: set[str]) -> str:
        if not isinstance(value, str) or value not in choices:
            raise CalendarToolError(
                "invalid_input", f"{field} must be one of: {', '.join(sorted(choices))}"
            )
        return value

    @staticmethod
    def _url(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.startswith("https://"):
            raise CalendarToolError(
                "invalid_input", f"{field} must be an https URL or null"
            )
        return value

    @staticmethod
    def _email(value: Any, field: str) -> str:
        if not _valid_email(value):
            raise CalendarToolError("invalid_input", f"{field} must be an email")
        return value.strip().lower()

    @staticmethod
    def _scope(*values: Any) -> str:
        return hashlib.sha256(_json(values).encode()).hexdigest()[:20]

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
            raise CalendarToolError(
                "invalid_cursor", "Calendar cursor is invalid for this request"
            ) from error

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


_DEFAULT_TOOLS: CalendarTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> CalendarTools:
    global _DEFAULT_TOOLS
    with _DEFAULT_LOCK:
        if _DEFAULT_TOOLS is None:
            _DEFAULT_TOOLS = CalendarTools()
        return _DEFAULT_TOOLS
