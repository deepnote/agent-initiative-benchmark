"""Local Google Analytics Data API benchmark wrapper."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

from benchmark import duckdb_runtime as duckdb
import tomllib

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
DEFAULT_REFERENCE_DATE = date(2026, 9, 2)


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
DEFAULT_BASELINE_DIR = DATA_DIR / "analytics"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "analytics"


class AnalyticsToolError(ValueError):
    """A stable error that is safe to return to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class AnalyticsConfig:
    """Paths, identity, limits, and query settings for Analytics reports."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    agent_user_id: str = "U_AGENT"
    reference_date: date = DEFAULT_REFERENCE_DATE
    max_page_size: int = 500
    max_dimensions: int = 9
    max_metrics: int = 10
    max_date_ranges: int = 4
    max_order_bys: int = 10
    max_filter_depth: int = 8
    max_filter_nodes: int = 64
    max_regex_characters: int = 256
    max_group_rows: int = 50_000
    query_timeout_seconds: float = 5.0
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "256MB"
    duckdb_max_temp_directory_size: str = "1GB"
    snapshot_time: str | None = None

    @classmethod
    def from_env(cls) -> AnalyticsConfig:
        raw_reference_date = os.getenv(
            "BENCHMARK_ANALYTICS_REFERENCE_DATE", DEFAULT_REFERENCE_DATE.isoformat()
        )
        try:
            reference_date = date.fromisoformat(raw_reference_date)
        except ValueError as error:
            raise ValueError(
                "BENCHMARK_ANALYTICS_REFERENCE_DATE must use YYYY-MM-DD"
            ) from error
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_ANALYTICS_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_ANALYTICS_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            agent_user_id=os.getenv(
                "BENCHMARK_ANALYTICS_AGENT_USER_ID", cls.agent_user_id
            ),
            reference_date=reference_date,
            max_page_size=int(os.getenv("ANALYTICS_MAX_PAGE_SIZE", cls.max_page_size)),
            max_dimensions=int(
                os.getenv("ANALYTICS_MAX_DIMENSIONS", cls.max_dimensions)
            ),
            max_metrics=int(os.getenv("ANALYTICS_MAX_METRICS", cls.max_metrics)),
            max_date_ranges=int(
                os.getenv("ANALYTICS_MAX_DATE_RANGES", cls.max_date_ranges)
            ),
            max_order_bys=int(os.getenv("ANALYTICS_MAX_ORDER_BYS", cls.max_order_bys)),
            max_filter_depth=int(
                os.getenv("ANALYTICS_MAX_FILTER_DEPTH", cls.max_filter_depth)
            ),
            max_filter_nodes=int(
                os.getenv("ANALYTICS_MAX_FILTER_NODES", cls.max_filter_nodes)
            ),
            max_regex_characters=int(
                os.getenv("ANALYTICS_MAX_REGEX_CHARACTERS", cls.max_regex_characters)
            ),
            max_group_rows=int(
                os.getenv("ANALYTICS_MAX_GROUP_ROWS", cls.max_group_rows)
            ),
            query_timeout_seconds=float(
                os.getenv("ANALYTICS_QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds)
            ),
            duckdb_threads=int(
                os.getenv("ANALYTICS_DUCKDB_THREADS", cls.duckdb_threads)
            ),
        )

    def validate(self) -> None:
        for name in (
            "max_page_size",
            "max_dimensions",
            "max_metrics",
            "max_date_ranges",
            "max_order_bys",
            "max_filter_depth",
            "max_filter_nodes",
            "max_regex_characters",
            "max_group_rows",
            "duckdb_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive")


SESSION_SCHEMA: tuple[tuple[str, str], ...] = (
    ("property_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("user_id", "VARCHAR"),
    ("date", "DATE"),
    ("country", "VARCHAR"),
    ("region", "VARCHAR"),
    ("city", "VARCHAR"),
    ("device_category", "VARCHAR"),
    ("session_default_channel_group", "VARCHAR"),
    ("source", "VARCHAR"),
    ("medium", "VARCHAR"),
    ("campaign_name", "VARCHAR"),
    ("landing_page", "VARCHAR"),
    ("is_new_user", "BOOLEAN"),
    ("is_active_user", "BOOLEAN"),
    ("engaged_session", "BOOLEAN"),
    ("engagement_duration_seconds", "DOUBLE"),
)

EVENT_SCHEMA: tuple[tuple[str, str], ...] = (
    ("property_id", "VARCHAR"),
    ("event_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("user_id", "VARCHAR"),
    ("date", "DATE"),
    ("event_timestamp", "VARCHAR"),
    ("event_name", "VARCHAR"),
    ("page_path", "VARCHAR"),
    ("page_title", "VARCHAR"),
    ("hostname", "VARCHAR"),
    ("event_count", "BIGINT"),
    ("key_event_count", "BIGINT"),
    ("total_revenue", "DOUBLE"),
)

OVERLAY_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    property_id TEXT,
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    date TEXT,
    country TEXT,
    region TEXT,
    city TEXT,
    device_category TEXT,
    session_default_channel_group TEXT,
    source TEXT,
    medium TEXT,
    campaign_name TEXT,
    landing_page TEXT,
    is_new_user INTEGER,
    is_active_user INTEGER,
    engaged_session INTEGER,
    engagement_duration_seconds REAL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);

CREATE INDEX IF NOT EXISTS sessions_date_idx ON sessions(property_id, date);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);

CREATE TABLE IF NOT EXISTS events (
    property_id TEXT,
    event_id TEXT PRIMARY KEY,
    session_id TEXT,
    user_id TEXT,
    date TEXT,
    event_timestamp TEXT,
    event_name TEXT,
    page_path TEXT,
    page_title TEXT,
    hostname TEXT,
    event_count INTEGER,
    key_event_count INTEGER,
    total_revenue REAL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);

CREATE INDEX IF NOT EXISTS events_date_idx ON events(property_id, date);
CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id);
CREATE INDEX IF NOT EXISTS events_name_idx ON events(event_name, date);
"""

SESSION_COLUMNS = tuple(name for name, _ in SESSION_SCHEMA)
EVENT_COLUMNS = tuple(name for name, _ in EVENT_SCHEMA)

DIMENSION_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "apiName": "date",
        "uiName": "Date",
        "description": "Event date as YYYYMMDD.",
        "column": "date_key",
    },
    {
        "apiName": "year",
        "uiName": "Year",
        "description": "Four digit event year.",
        "column": "year",
    },
    {
        "apiName": "month",
        "uiName": "Month",
        "description": "Two digit event month.",
        "column": "month",
    },
    {
        "apiName": "yearMonth",
        "uiName": "Year month",
        "description": "Event year and month as YYYYMM.",
        "column": "year_month",
    },
    {
        "apiName": "country",
        "uiName": "Country",
        "description": "Country recorded for the session.",
        "column": "country",
    },
    {
        "apiName": "region",
        "uiName": "Region",
        "description": "Region recorded for the session.",
        "column": "region",
    },
    {
        "apiName": "city",
        "uiName": "City",
        "description": "City recorded for the session.",
        "column": "city",
    },
    {
        "apiName": "deviceCategory",
        "uiName": "Device category",
        "description": "Desktop, mobile, or tablet.",
        "column": "device_category",
    },
    {
        "apiName": "sessionDefaultChannelGroup",
        "uiName": "Session default channel group",
        "description": "Acquisition channel assigned to the session.",
        "column": "session_default_channel_group",
    },
    {
        "apiName": "sessionSource",
        "uiName": "Session source",
        "description": "Traffic source assigned to the session.",
        "column": "source",
    },
    {
        "apiName": "sessionMedium",
        "uiName": "Session medium",
        "description": "Traffic medium assigned to the session.",
        "column": "medium",
    },
    {
        "apiName": "sessionCampaignName",
        "uiName": "Session campaign",
        "description": "Campaign assigned to the session.",
        "column": "campaign_name",
    },
    {
        "apiName": "landingPage",
        "uiName": "Landing page",
        "description": "First page path in the session.",
        "column": "landing_page",
    },
    {
        "apiName": "pagePath",
        "uiName": "Page path",
        "description": "Path attached to the event.",
        "column": "page_path",
    },
    {
        "apiName": "pageTitle",
        "uiName": "Page title",
        "description": "Title attached to the event.",
        "column": "page_title",
    },
    {
        "apiName": "hostName",
        "uiName": "Host name",
        "description": "Host attached to the event.",
        "column": "hostname",
    },
    {
        "apiName": "eventName",
        "uiName": "Event name",
        "description": "Name of the recorded event.",
        "column": "event_name",
    },
)

METRIC_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "apiName": "activeUsers",
        "uiName": "Active users",
        "description": "Distinct active users.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "totalUsers",
        "uiName": "Total users",
        "description": "Distinct users.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "newUsers",
        "uiName": "New users",
        "description": "Distinct users whose first session is in the group.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "sessions",
        "uiName": "Sessions",
        "description": "Distinct sessions.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "engagedSessions",
        "uiName": "Engaged sessions",
        "description": "Distinct engaged sessions.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "engagementRate",
        "uiName": "Engagement rate",
        "description": "Engaged sessions divided by sessions.",
        "type": "TYPE_FLOAT",
    },
    {
        "apiName": "bounceRate",
        "uiName": "Bounce rate",
        "description": "One minus engagement rate.",
        "type": "TYPE_FLOAT",
    },
    {
        "apiName": "eventCount",
        "uiName": "Event count",
        "description": "Recorded event count.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "screenPageViews",
        "uiName": "Views",
        "description": "Page view event count.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "keyEvents",
        "uiName": "Key events",
        "description": "Recorded key event count.",
        "type": "TYPE_INTEGER",
    },
    {
        "apiName": "totalRevenue",
        "uiName": "Total revenue",
        "description": "Revenue in the property currency.",
        "type": "TYPE_CURRENCY",
    },
)

DIMENSION_SQL = {
    "date": "strftime(r.event_date, '%Y%m%d')",
    "year": "strftime(r.event_date, '%Y')",
    "month": "strftime(r.event_date, '%m')",
    "yearMonth": "strftime(r.event_date, '%Y%m')",
    "country": "r.country",
    "region": "r.region",
    "city": "r.city",
    "deviceCategory": "r.device_category",
    "sessionDefaultChannelGroup": "r.session_default_channel_group",
    "sessionSource": "r.source",
    "sessionMedium": "r.medium",
    "sessionCampaignName": "r.campaign_name",
    "landingPage": "r.landing_page",
    "pagePath": "r.page_path",
    "pageTitle": "r.page_title",
    "hostName": "r.hostname",
    "eventName": "r.event_name",
}

METRIC_SQL = {
    "activeUsers": "COUNT(DISTINCT CASE WHEN r.is_active_user THEN r.user_id END)",
    "totalUsers": "COUNT(DISTINCT r.user_id)",
    "newUsers": "COUNT(DISTINCT CASE WHEN r.is_new_user THEN r.user_id END)",
    "sessions": "COUNT(DISTINCT r.session_id)",
    "engagedSessions": "COUNT(DISTINCT CASE WHEN r.engaged_session THEN r.session_id END)",
    "engagementRate": "CASE WHEN COUNT(DISTINCT r.session_id) = 0 THEN 0 ELSE COUNT(DISTINCT CASE WHEN r.engaged_session THEN r.session_id END)::DOUBLE / COUNT(DISTINCT r.session_id) END",
    "bounceRate": "CASE WHEN COUNT(DISTINCT r.session_id) = 0 THEN 0 ELSE 1.0 - COUNT(DISTINCT CASE WHEN r.engaged_session THEN r.session_id END)::DOUBLE / COUNT(DISTINCT r.session_id) END",
    "eventCount": "COALESCE(SUM(r.event_count), 0)",
    "screenPageViews": "COALESCE(SUM(CASE WHEN r.event_name = 'page_view' THEN r.event_count ELSE 0 END), 0)",
    "keyEvents": "COALESCE(SUM(r.key_event_count), 0)",
    "totalRevenue": "COALESCE(SUM(r.total_revenue), 0)",
}

METRIC_TYPES = {item["apiName"]: item["type"] for item in METRIC_DEFINITIONS}
DATE_AGO_PATTERN = re.compile(r"^(\d+)daysAgo$")

AGGREGATE_SESSION_DIMENSION_SQL = {
    "date": "strftime(r.date, '%Y%m%d')",
    "year": "strftime(r.date, '%Y')",
    "month": "strftime(r.date, '%m')",
    "yearMonth": "strftime(r.date, '%Y%m')",
    "hour": "lpad(CAST(r.hour AS VARCHAR), 2, '0')",
    "dayOfWeek": "r.day_of_week",
    "streamId": "r.stream_id",
    "workspaceId": "r.workspace_id",
    "websiteId": "r.website_id",
    "scenarioId": "r.scenario_id",
    "hostName": "r.host_name",
    "landingPageId": "r.landing_page_id",
    "landingPage": "r.landing_page_path",
    "pageReferrer": "r.page_referrer",
    "websiteTier": "r.website_tier",
    "websiteTrafficClass": "r.website_traffic_class",
    "contentGroup": "r.content_group",
    "firstUserDefaultChannelGroup": "r.first_user_default_channel_group",
    "firstUserSource": "r.first_user_source",
    "firstUserMedium": "r.first_user_medium",
    "firstUserCampaignId": "r.first_user_campaign_id",
    "firstUserCampaignName": "r.first_user_campaign_name",
    "sessionDefaultChannelGroup": "r.session_default_channel_group",
    "sessionSource": "r.session_source",
    "sessionMedium": "r.session_medium",
    "sessionCampaignId": "r.session_campaign_id",
    "sessionCampaignName": "r.session_campaign_name",
    "sessionSourcePlatform": "r.session_source_platform",
    "country": "r.country",
    "countryCode": "r.country_code",
    "region": "r.region",
    "city": "r.city",
    "language": "r.language",
    "timeZone": "r.time_zone",
    "deviceCategory": "r.device_category",
    "browser": "r.browser",
    "browserVersion": "r.browser_version_major",
    "operatingSystem": "r.operating_system",
    "operatingSystemVersion": "r.operating_system_version_major",
    "deviceBrand": "r.device_brand",
    "deviceModel": "r.device_model",
    "screenResolution": "r.screen_resolution",
    "platform": "r.platform",
    "audienceName": "r.audience_name",
    "planTier": "r.plan_tier",
    "workspaceType": "r.workspace_type",
    "newVsReturning": "r.new_or_returning",
}

AGGREGATE_EVENT_DIMENSION_SQL = {
    "date": "strftime(r.date, '%Y%m%d')",
    "year": "strftime(r.date, '%Y')",
    "month": "strftime(r.date, '%m')",
    "yearMonth": "strftime(r.date, '%Y%m')",
    "hour": "lpad(CAST(r.hour AS VARCHAR), 2, '0')",
    "streamId": "r.stream_id",
    "workspaceId": "r.workspace_id",
    "websiteId": "r.website_id",
    "scenarioId": "r.scenario_id",
    "hostName": "r.host_name",
    "pageId": "r.page_id",
    "pagePath": "r.page_path",
    "pageTitle": "r.page_title",
    "contentGroup": "r.content_group",
    "eventCategory": "r.event_category",
    "isKeyEvent": "r.is_key_event",
    "sessionDefaultChannelGroup": "r.session_default_channel_group",
    "sessionSource": "r.session_source",
    "sessionMedium": "r.session_medium",
    "sessionCampaignId": "r.session_campaign_id",
    "countryCode": "r.country_code",
    "country": "r.country",
    "region": "r.region",
    "city": "r.city",
    "deviceCategory": "r.device_category",
    "browser": "r.browser",
    "operatingSystem": "r.operating_system",
    "platform": "r.platform",
    "eventName": "r.event_name",
    "keyEventName": "CASE WHEN r.is_key_event THEN r.event_name ELSE NULL END",
}

AGGREGATE_SESSION_METRIC_SQL = {
    "activeUsers": "COALESCE(SUM(r.active_user_count), 0)",
    "totalUsers": "COALESCE(SUM(r.active_user_count), 0)",
    "newUsers": "COALESCE(SUM(r.new_user_count), 0)",
    "returningUsers": "COALESCE(SUM(r.returning_user_count), 0)",
    "sessions": "COALESCE(SUM(r.session_count), 0)",
    "engagedSessions": "COALESCE(SUM(r.engaged_session_count), 0)",
    "engagementRate": "CASE WHEN SUM(r.session_count)=0 THEN 0 ELSE SUM(r.engaged_session_count)::DOUBLE/SUM(r.session_count) END",
    "bounceRate": "CASE WHEN SUM(r.session_count)=0 THEN 0 ELSE 1.0-SUM(r.engaged_session_count)::DOUBLE/SUM(r.session_count) END",
    "eventCount": "COALESCE(SUM(r.event_count), 0)",
    "screenPageViews": "COALESCE(SUM(r.view_count), 0)",
    "keyEvents": "COALESCE(SUM(r.key_event_count), 0)",
    "transactions": "COALESCE(SUM(r.transaction_count), 0)",
    "totalRevenue": "COALESCE(SUM(r.total_revenue), 0)",
    "averageEngagementTimePerSession": "CASE WHEN SUM(r.session_count)=0 THEN 0 ELSE SUM(r.engagement_duration_seconds)::DOUBLE/SUM(r.session_count) END",
    "averageEngagementTimePerActiveUser": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.engagement_duration_seconds)::DOUBLE/SUM(r.active_user_count) END",
    "engagedSessionsPerActiveUser": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.engaged_session_count)::DOUBLE/SUM(r.active_user_count) END",
    "sessionsPerActiveUser": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.session_count)::DOUBLE/SUM(r.active_user_count) END",
    "eventsPerSession": "CASE WHEN SUM(r.session_count)=0 THEN 0 ELSE SUM(r.event_count)::DOUBLE/SUM(r.session_count) END",
    "screenPageViewsPerActiveUser": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.view_count)::DOUBLE/SUM(r.active_user_count) END",
    "sessionKeyEventRate": "CASE WHEN SUM(r.session_count)=0 THEN 0 ELSE SUM(r.key_event_session_count)::DOUBLE/SUM(r.session_count) END",
    "userKeyEventRate": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.key_event_session_count)::DOUBLE/SUM(r.active_user_count) END",
    "newUserRate": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.new_user_count)::DOUBLE/SUM(r.active_user_count) END",
    "returningUserRate": "CASE WHEN SUM(r.active_user_count)=0 THEN 0 ELSE SUM(r.returning_user_count)::DOUBLE/SUM(r.active_user_count) END",
}

AGGREGATE_EVENT_METRIC_SQL = {
    "activeUsers": "COALESCE(SUM(r.active_user_count), 0)",
    "totalUsers": "COALESCE(SUM(r.active_user_count), 0)",
    "sessions": "COALESCE(SUM(r.session_count), 0)",
    "eventCount": "COALESCE(SUM(r.event_count), 0)",
    "screenPageViews": "COALESCE(SUM(r.event_count) FILTER (WHERE r.event_name='page_view'), 0)",
    "keyEvents": "COALESCE(SUM(r.key_event_count), 0)",
    "transactions": "COALESCE(SUM(r.transaction_count), 0)",
    "totalRevenue": "COALESCE(SUM(r.total_revenue), 0)",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _overlay_value(row: Mapping[str, Any], column: str) -> Any:
    value = row.get(column)
    if (
        column in {"is_new_user", "is_active_user", "engaged_session"}
        and value is not None
    ):
        return int(bool(value))
    return value


def build_analytics_overlay_from_seed(
    seed_path: Path, overlay_path: Path
) -> dict[str, int]:
    """Build a small SQLite overlay with upserts and tombstones."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise TypeError("Analytics overlay seed must be a JSON object")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_path.exists():
        overlay_path.unlink()
    connection = sqlite3.connect(overlay_path)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", ("format_version", "1")
        )
        for table, columns in (
            ("sessions", SESSION_COLUMNS),
            ("events", EVENT_COLUMNS),
        ):
            for row in seed.get(table, []):
                operation = row.get("operation", "upsert")
                if operation not in {"upsert", "delete"}:
                    raise ValueError(
                        f"Invalid Analytics overlay operation: {operation}"
                    )
                key = "session_id" if table == "sessions" else "event_id"
                if not row.get(key):
                    raise ValueError(f"Analytics overlay {table} row needs {key}")
                values = [_overlay_value(row, column) for column in columns]
                connection.execute(
                    f"INSERT INTO {table}({','.join(columns)},operation) "
                    f"VALUES ({','.join('?' for _ in range(len(columns) + 1))})",
                    [*values, operation],
                )
        connection.commit()
    finally:
        connection.close()
    return {
        "sessions": len(seed.get("sessions", [])),
        "events": len(seed.get("events", [])),
    }


class AnalyticsTools:
    """Run bounded reports over effective baseline and overlay Analytics facts."""

    def __init__(self, config: AnalyticsConfig | None = None, *, acquire_family=None):
        self.config = config or AnalyticsConfig.from_env()
        self.config.validate()
        self._lock = threading.RLock()
        self._acquire_family = acquire_family
        self._loaded_families = set()
        self._v1_ready = False
        self._validate_files()
        self._property = json.loads(
            (self.config.baseline_dir / "property.json").read_text(encoding="utf-8")
        )
        manifest_path = self.config.baseline_dir / "manifest.json"
        self._manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        self._aggregate_mode = int(self._manifest.get("schema_version", 1)) >= 2
        self._dimensions = {
            item["apiName"]: item
            for item in json.loads(
                (self.config.baseline_dir / "dimensions.json").read_text(
                    encoding="utf-8"
                )
            )
        }
        self._metrics = {
            item["apiName"]: item
            for item in json.loads(
                (self.config.baseline_dir / "metrics.json").read_text(encoding="utf-8")
            )
        }
        self._metric_types = {
            name: item.get("type", "TYPE_FLOAT") for name, item in self._metrics.items()
        }
        self._duck = duckdb.connect(
            database=":memory:",
            config={
                "memory_limit": self.config.duckdb_memory_limit,
                "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
            },
        )
        self._duck.execute(f"PRAGMA threads={self.config.duckdb_threads}")
        if acquire_family is None:
            self._ensure_report()

    def _validate_files(self) -> None:
        manifest_path = self.config.baseline_dir / "manifest.json"
        aggregate_mode = False
        if manifest_path.is_file():
            try:
                aggregate_mode = (
                    int(
                        json.loads(manifest_path.read_text(encoding="utf-8")).get(
                            "schema_version", 1
                        )
                    )
                    >= 2
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                aggregate_mode = False
        required = [
            self.config.baseline_dir / "property.json",
            self.config.baseline_dir / "dimensions.json",
            self.config.baseline_dir / "metrics.json",
            *(
                []
                if self._acquire_family
                else [
                    self.config.baseline_dir / "sessions",
                    self.config.baseline_dir / "events",
                ]
            ),
        ]
        if not aggregate_mode:
            required.append(self.config.overlay_dir / "overlay.sqlite")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Google Analytics benchmark data: " + ", ".join(missing)
            )

    def _create_views(self) -> None:
        sessions_path = self.config.baseline_dir / "sessions" / "**" / "*.parquet"
        events_path = self.config.baseline_dir / "events" / "**" / "*.parquet"
        if self._aggregate_mode:
            from .runtime import ensure

            ensure(self, "sessions")
            ensure(self, "events")
            return
        self._duck.execute(
            f"CREATE VIEW baseline_sessions AS SELECT * EXCLUDE (year, month) "
            f"FROM read_parquet('{_sql_path(sessions_path)}', union_by_name=true, hive_partitioning=true)"
        )
        self._duck.execute(
            f"CREATE VIEW baseline_events AS SELECT * EXCLUDE (year, month) "
            f"FROM read_parquet('{_sql_path(events_path)}', union_by_name=true, hive_partitioning=true)"
        )
        session_columns = ",".join(f'"{name}" {kind}' for name, kind in SESSION_SCHEMA)
        event_columns = ",".join(f'"{name}" {kind}' for name, kind in EVENT_SCHEMA)
        self._duck.execute(
            f"CREATE TABLE overlay_sessions ({session_columns}, operation VARCHAR)"
        )
        self._duck.execute(
            f"CREATE TABLE overlay_events ({event_columns}, operation VARCHAR)"
        )
        self._duck.execute(
            """
            CREATE VIEW effective_sessions AS
            SELECT * EXCLUDE (operation) FROM overlay_sessions WHERE operation = 'upsert'
            UNION ALL
            SELECT b.* FROM baseline_sessions b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_sessions o WHERE o.session_id = b.session_id
            )
            """
        )
        self._duck.execute(
            """
            CREATE VIEW effective_events AS
            SELECT * EXCLUDE (operation) FROM overlay_events WHERE operation = 'upsert'
            UNION ALL
            SELECT b.* FROM baseline_events b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_events o WHERE o.event_id = b.event_id
            )
            """
        )
        self._duck.execute(
            """
            CREATE VIEW analytics_rows AS
            SELECT
                e.property_id,
                e.event_id,
                e.session_id,
                e.user_id,
                e.date AS event_date,
                e.event_timestamp,
                e.event_name,
                e.page_path,
                e.page_title,
                e.hostname,
                e.event_count,
                e.key_event_count,
                e.total_revenue,
                s.country,
                s.region,
                s.city,
                s.device_category,
                s.session_default_channel_group,
                s.source,
                s.medium,
                s.campaign_name,
                s.landing_page,
                s.is_new_user,
                s.is_active_user,
                s.engaged_session,
                s.engagement_duration_seconds
            FROM effective_events e
            JOIN effective_sessions s USING (property_id, session_id, user_id)
            """
        )

    def _ensure_report(self, relation=None):
        if self._aggregate_mode:
            from .runtime import ensure

            for table in (
                (relation.removeprefix("aggregate_"),)
                if relation
                else ("sessions", "events")
            ):
                ensure(self, table)
        elif not self._v1_ready:
            if self._acquire_family:
                self._acquire_family("sessions")
                self._acquire_family("events")
            self._create_views()
            self._load_overlay()
            self._v1_ready = True

    def analytics_get_metadata(self, property_id):
        from .runtime import metadata

        with self._lock:
            return metadata(self, property_id)

    def analytics_read_rows(self, property_id, table, **kwargs):
        from .runtime import records

        with self._lock:
            return records(self, property_id, table, **kwargs)

    @staticmethod
    def _open_overlay(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _load_overlay(self) -> None:
        if self._aggregate_mode:
            return
        path = self.config.overlay_dir / "overlay.sqlite"
        with closing(self._open_overlay(path)) as connection:
            for table, columns in (
                ("sessions", SESSION_COLUMNS),
                ("events", EVENT_COLUMNS),
            ):
                rows = connection.execute(
                    f"SELECT {','.join(columns)}, operation FROM {table}"
                ).fetchall()
                if rows:
                    all_columns = (*columns, "operation")
                    self._duck.executemany(
                        f"INSERT INTO overlay_{table} VALUES "
                        f"({','.join('?' for _ in all_columns)})",
                        [tuple(row[column] for column in all_columns) for row in rows],
                    )

    def close(self) -> None:
        with self._lock:
            self._duck.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def analytics_run_report(
        self,
        property_id: str,
        *,
        dimensions: Sequence[str | Mapping[str, Any]] = (),
        metrics: Sequence[str | Mapping[str, Any]],
        date_ranges: Sequence[Mapping[str, Any]],
        dimension_filter: Mapping[str, Any] | None = None,
        metric_filter: Mapping[str, Any] | None = None,
        order_bys: Sequence[Mapping[str, Any]] = (),
        metric_aggregations: Sequence[str] = (),
        keep_empty_rows: bool = False,
        page_size: int = 100,
        start_cursor: str | None = None,
        return_property_quota: bool = False,
    ) -> dict[str, Any]:
        """Return one Google Analytics style report with stable cursor pagination."""

        with self._lock:
            self._check_access(property_id)
            dimension_names = self._field_names(
                dimensions, "dimension", self._dimensions, self.config.max_dimensions
            )
            metric_names = self._field_names(
                metrics, "metric", self._metrics, self.config.max_metrics
            )
            if not metric_names:
                raise AnalyticsToolError(
                    "invalid_argument", "At least one metric is required"
                )
            ranges = self._date_ranges(date_ranges)
            if not isinstance(keep_empty_rows, bool):
                raise AnalyticsToolError(
                    "invalid_argument", "keep_empty_rows must be a boolean"
                )
            if not isinstance(return_property_quota, bool):
                raise AnalyticsToolError(
                    "invalid_argument", "return_property_quota must be a boolean"
                )
            aggregations = [str(item).upper() for item in metric_aggregations]
            if any(item != "TOTAL" for item in aggregations) or len(
                set(aggregations)
            ) != len(aggregations):
                raise AnalyticsToolError(
                    "invalid_argument", "metric_aggregations supports TOTAL once"
                )
            relation, dimension_fields, metric_fields = self._query_contract(
                dimension_names, metric_names
            )
            self._ensure_report(relation)
            if self.config.snapshot_time and "scenarioId" in dimension_names:
                raise AnalyticsToolError(
                    "invalid_dimension", "Harness fields are unavailable"
                )
            order_specs = self._order_bys(order_bys, dimension_names, metric_names)
            dimension_sql, dimension_params = self._compile_filter(
                dimension_filter,
                "dimension",
                set(dimension_names),
                dimension_fields,
            )
            metric_sql, metric_params = self._compile_filter(
                metric_filter,
                "metric",
                set(metric_names),
                metric_fields,
            )
            limit = self._page_size(page_size)
            scope_payload = {
                "property": property_id,
                "dimensions": dimension_names,
                "metrics": metric_names,
                "dateRanges": ranges,
                "dimensionFilter": dimension_filter,
                "metricFilter": metric_filter,
                "orderBys": order_specs,
                "metricAggregations": aggregations,
                "keepEmptyRows": keep_empty_rows,
            }
            scope = hashlib.sha256(_json(scope_payload).encode()).hexdigest()[:24]
            offset = self._cursor_offset(start_cursor, scope)

            rows = self._report_rows(
                property_id,
                dimension_names,
                metric_names,
                ranges,
                dimension_sql,
                dimension_params,
                metric_sql,
                metric_params,
                relation,
                dimension_fields,
                metric_fields,
            )
            rows = self._sort_rows(rows, order_specs, dimension_names)
            total = len(rows)
            page = rows[offset : offset + limit]
            next_offset = offset + len(page)
            has_more = next_offset < total
            multiple_ranges = len(ranges) > 1
            response: dict[str, Any] = {
                "ok": True,
                "kind": "analyticsData#runReport",
                "dimensionHeaders": [
                    *([{"name": "dateRange"}] if multiple_ranges else []),
                    *[{"name": name} for name in dimension_names],
                ],
                "metricHeaders": [
                    {"name": name, "type": self._metric_types[name]}
                    for name in metric_names
                ],
                "rows": [
                    self._response_row(
                        row, dimension_names, metric_names, multiple_ranges
                    )
                    for row in page
                ],
                "rowCount": total,
                "hasMore": has_more,
                "nextCursor": (
                    self._encode_cursor(next_offset, scope) if has_more else None
                ),
                "metadata": {
                    "currencyCode": self._property["currencyCode"],
                    "timeZone": self._property["timeZone"],
                    "dataLossFromOtherRow": False,
                },
            }
            if "TOTAL" in aggregations:
                totals = self._total_rows(
                    property_id,
                    metric_names,
                    ranges,
                    dimension_sql,
                    dimension_params,
                    relation,
                    metric_fields,
                )
                response["totals"] = [
                    self._total_response_row(
                        row, dimension_names, metric_names, multiple_ranges
                    )
                    for row in totals
                ]
            if return_property_quota:
                response["propertyQuota"] = {
                    "tokensPerDay": {"consumed": 0, "remaining": 100_000},
                    "tokensPerHour": {"consumed": 0, "remaining": 10_000},
                    "concurrentRequests": {"consumed": 1, "remaining": 9},
                }
            if self._aggregate_mode:
                response["metadata"].update(self._property.get("reportMetadata", {}))
                response["metadata"].update(
                    exact=True,
                    subjectToThresholding=bool(self._property.get("privacyThreshold")),
                    snapshotTime=self.config.snapshot_time,
                    grain=relation.removeprefix("aggregate_"),
                )
            return response

    def _check_access(self, property_id: str) -> None:
        if not isinstance(property_id, str) or not re.fullmatch(
            r"properties/\d+", property_id
        ):
            raise AnalyticsToolError(
                "invalid_argument", "property_id must use properties/<digits>"
            )
        allowed_users = {
            item.get("userId")
            for item in self._property.get("access", [])
            if item.get("role") in {"viewer", "analyst", "editor", "administrator"}
        }
        if (
            property_id != self._property.get("id")
            or self.config.agent_user_id not in allowed_users
        ):
            raise AnalyticsToolError(
                "permission_denied", "The agent cannot access this Analytics property"
            )

    def _query_contract(
        self,
        dimensions: Sequence[str],
        metrics: Sequence[str],
    ) -> tuple[str, Mapping[str, str], Mapping[str, str]]:
        if not self._aggregate_mode:
            return "analytics_rows", DIMENSION_SQL, METRIC_SQL
        if "userKeyEventRate" in metrics:
            raise AnalyticsToolError(
                "unsupported_aggregation",
                "Unique key-event users are not stored at this grain",
            )
        event_dimensions = {
            "eventName",
            "eventCategory",
            "isKeyEvent",
            "keyEventName",
            "pageId",
            "pagePath",
            "pageTitle",
        }
        use_events = bool(event_dimensions.intersection(dimensions))
        relation = "aggregate_events" if use_events else "aggregate_sessions"
        dimension_fields = (
            AGGREGATE_EVENT_DIMENSION_SQL
            if use_events
            else AGGREGATE_SESSION_DIMENSION_SQL
        )
        metric_fields = (
            AGGREGATE_EVENT_METRIC_SQL if use_events else AGGREGATE_SESSION_METRIC_SQL
        )
        missing_dimensions = [
            name for name in dimensions if name not in dimension_fields
        ]
        if missing_dimensions:
            raise AnalyticsToolError(
                "invalid_dimension",
                "The selected Analytics grain does not support: "
                + ", ".join(missing_dimensions),
            )
        missing_metrics = [name for name in metrics if name not in metric_fields]
        if missing_metrics:
            raise AnalyticsToolError(
                "invalid_metric",
                "The selected Analytics grain does not support: "
                + ", ".join(missing_metrics),
            )
        return relation, dimension_fields, metric_fields

    def _field_names(
        self,
        fields: Sequence[str | Mapping[str, Any]],
        kind: str,
        definitions: Mapping[str, Any],
        maximum: int,
    ) -> list[str]:
        if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
            raise AnalyticsToolError("invalid_argument", f"{kind}s must be a list")
        if len(fields) > maximum:
            raise AnalyticsToolError(
                "invalid_argument", f"At most {maximum} {kind}s are allowed"
            )
        names: list[str] = []
        for field in fields:
            name = field if isinstance(field, str) else field.get("name")
            if not isinstance(name, str) or not name:
                raise AnalyticsToolError(
                    "invalid_argument", f"Each {kind} needs a name"
                )
            if name not in definitions:
                raise AnalyticsToolError(
                    f"invalid_{kind}", f"Unknown Analytics {kind}: {name}"
                )
            if name in names:
                raise AnalyticsToolError(
                    "invalid_argument", f"Duplicate Analytics {kind}: {name}"
                )
            names.append(name)
        return names

    def _date_ranges(
        self, date_ranges: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        if isinstance(date_ranges, (str, bytes)) or not isinstance(
            date_ranges, Sequence
        ):
            raise AnalyticsToolError("invalid_argument", "date_ranges must be a list")
        if not date_ranges or len(date_ranges) > self.config.max_date_ranges:
            raise AnalyticsToolError(
                "invalid_argument",
                f"Provide 1 to {self.config.max_date_ranges} date ranges",
            )
        parsed: list[dict[str, str]] = []
        for item in date_ranges:
            if not isinstance(item, Mapping):
                raise AnalyticsToolError(
                    "invalid_argument", "Each date range must be an object"
                )
            start_raw = item.get("startDate", item.get("start_date"))
            end_raw = item.get("endDate", item.get("end_date"))
            start = self._parse_date(start_raw)
            end = self._parse_date(end_raw)
            if start > end:
                raise AnalyticsToolError(
                    "invalid_argument", "A date range start must not be after its end"
                )
            parsed.append({"startDate": start.isoformat(), "endDate": end.isoformat()})
        return parsed

    def _parse_date(self, value: Any) -> date:
        if not isinstance(value, str):
            raise AnalyticsToolError("invalid_argument", "Report dates must be strings")
        if value == "today":
            return self.config.reference_date
        if value == "yesterday":
            return self.config.reference_date - timedelta(days=1)
        match = DATE_AGO_PATTERN.fullmatch(value)
        if match:
            return self.config.reference_date - timedelta(days=int(match.group(1)))
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise AnalyticsToolError(
                "invalid_argument",
                f"Invalid report date: {value}. Use YYYY-MM-DD, today, yesterday, or NdaysAgo",
            ) from error

    def _page_size(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AnalyticsToolError(
                "invalid_argument", "page_size must be a positive integer"
            )
        if value > self.config.max_page_size:
            raise AnalyticsToolError(
                "invalid_argument",
                f"page_size cannot exceed {self.config.max_page_size}",
            )
        return value

    def _cursor_offset(self, cursor: str | None, scope: str) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or len(cursor) > 512:
            raise AnalyticsToolError("invalid_cursor", "The report cursor is invalid")
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
            offset = payload["offset"]
            if payload["scope"] != scope or not isinstance(offset, int) or offset < 0:
                raise ValueError
            return offset
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise AnalyticsToolError(
                "invalid_cursor",
                "The report cursor is invalid or belongs to another report",
            ) from error

    @staticmethod
    def _encode_cursor(offset: int, scope: str) -> str:
        raw = _json({"offset": offset, "scope": scope}).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def _compile_filter(
        self,
        expression: Mapping[str, Any] | None,
        kind: str,
        requested: set[str],
        field_sql: Mapping[str, str],
    ) -> tuple[str, list[Any]]:
        if expression is None:
            return "", []
        counter = [0]

        def visit(node: Any, depth: int) -> tuple[str, list[Any]]:
            if depth > self.config.max_filter_depth:
                raise AnalyticsToolError(
                    "invalid_argument", "Analytics filter nesting is too deep"
                )
            if not isinstance(node, Mapping):
                raise AnalyticsToolError(
                    "invalid_argument",
                    "Each Analytics filter expression must be an object",
                )
            counter[0] += 1
            if counter[0] > self.config.max_filter_nodes:
                raise AnalyticsToolError(
                    "invalid_argument", "Analytics filter has too many expressions"
                )
            keys = [
                key
                for key in ("andGroup", "orGroup", "notExpression", "filter")
                if key in node
            ]
            if len(keys) != 1:
                raise AnalyticsToolError(
                    "invalid_argument",
                    "A filter expression must use exactly one expression type",
                )
            key = keys[0]
            if key in {"andGroup", "orGroup"}:
                group = node[key]
                expressions = (
                    group.get("expressions") if isinstance(group, Mapping) else None
                )
                if not isinstance(expressions, list) or not expressions:
                    raise AnalyticsToolError(
                        "invalid_argument", f"{key} needs a nonempty expressions list"
                    )
                compiled = [visit(item, depth + 1) for item in expressions]
                operator = " AND " if key == "andGroup" else " OR "
                return (
                    "(" + operator.join(sql for sql, _ in compiled) + ")",
                    [value for _, params in compiled for value in params],
                )
            if key == "notExpression":
                sql, params = visit(node[key], depth + 1)
                return f"NOT ({sql})", params
            return self._leaf_filter(node[key], kind, requested, field_sql)

        return visit(expression, 1)

    def _leaf_filter(
        self,
        leaf: Any,
        kind: str,
        requested: set[str],
        field_sql_map: Mapping[str, str],
    ) -> tuple[str, list[Any]]:
        if not isinstance(leaf, Mapping):
            raise AnalyticsToolError("invalid_argument", "filter must be an object")
        field_name = leaf.get("fieldName")
        definitions = self._dimensions if kind == "dimension" else self._metrics
        if not isinstance(field_name, str) or field_name not in definitions:
            raise AnalyticsToolError(
                f"invalid_{kind}", f"Unknown Analytics {kind}: {field_name}"
            )
        if field_name not in requested:
            raise AnalyticsToolError(
                "invalid_argument",
                f"Filtered {kind} must also be requested: {field_name}",
            )
        if field_name not in field_sql_map:
            raise AnalyticsToolError(
                f"invalid_{kind}",
                f"Analytics {kind} is not available at this report grain: {field_name}",
            )
        field_sql = field_sql_map[field_name]
        filter_types = [
            key
            for key in (
                "stringFilter",
                "inListFilter",
                "numericFilter",
                "betweenFilter",
                "emptyFilter",
            )
            if key in leaf
        ]
        if len(filter_types) != 1:
            raise AnalyticsToolError(
                "invalid_argument", "A filter must use exactly one filter type"
            )
        filter_type = filter_types[0]
        value = leaf[filter_type]
        if filter_type == "emptyFilter":
            if value not in ({}, True):
                raise AnalyticsToolError(
                    "invalid_argument", "emptyFilter must be an empty object"
                )
            return f"({field_sql} IS NULL OR CAST({field_sql} AS VARCHAR) = '')", []
        if not isinstance(value, Mapping):
            raise AnalyticsToolError(
                "invalid_argument", f"{filter_type} must be an object"
            )
        if filter_type == "stringFilter":
            if kind != "dimension":
                raise AnalyticsToolError(
                    "invalid_argument", "stringFilter can only filter dimensions"
                )
            raw = value.get("value")
            if not isinstance(raw, str):
                raise AnalyticsToolError(
                    "invalid_argument", "stringFilter.value must be a string"
                )
            match_type = str(value.get("matchType", "EXACT")).upper()
            case_sensitive = value.get("caseSensitive", False)
            if not isinstance(case_sensitive, bool):
                raise AnalyticsToolError(
                    "invalid_argument", "caseSensitive must be a boolean"
                )
            sql_field = (
                field_sql if case_sensitive else f"LOWER(CAST({field_sql} AS VARCHAR))"
            )
            parameter = raw if case_sensitive else raw.lower()
            if match_type == "EXACT":
                return f"{sql_field} = ?", [parameter]
            if match_type in {"BEGINS_WITH", "ENDS_WITH", "CONTAINS"}:
                escaped = (
                    parameter.replace("$", "$$").replace("%", "$%").replace("_", "$_")
                )
                if match_type == "BEGINS_WITH":
                    escaped = escaped + "%"
                elif match_type == "ENDS_WITH":
                    escaped = "%" + escaped
                else:
                    escaped = "%" + escaped + "%"
                return f"{sql_field} LIKE ? ESCAPE '$'", [escaped]
            if match_type in {"FULL_REGEXP", "PARTIAL_REGEXP"}:
                if len(parameter) > self.config.max_regex_characters:
                    raise AnalyticsToolError(
                        "invalid_argument", "Regular expression filter is too long"
                    )
                pattern = (
                    parameter
                    if match_type == "PARTIAL_REGEXP"
                    else f"^(?:{parameter})$"
                )
                return f"regexp_matches({sql_field}, ?)", [pattern]
            raise AnalyticsToolError(
                "invalid_argument", f"Unsupported stringFilter matchType: {match_type}"
            )
        if filter_type == "inListFilter":
            if kind != "dimension":
                raise AnalyticsToolError(
                    "invalid_argument", "inListFilter can only filter dimensions"
                )
            values = value.get("values")
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(item, str) for item in values)
            ):
                raise AnalyticsToolError(
                    "invalid_argument",
                    "inListFilter.values must be a nonempty string list",
                )
            case_sensitive = value.get("caseSensitive", False)
            if not isinstance(case_sensitive, bool):
                raise AnalyticsToolError(
                    "invalid_argument", "caseSensitive must be a boolean"
                )
            sql_field = (
                field_sql if case_sensitive else f"LOWER(CAST({field_sql} AS VARCHAR))"
            )
            parameters = values if case_sensitive else [item.lower() for item in values]
            return f"{sql_field} IN ({','.join('?' for _ in parameters)})", parameters
        if filter_type == "numericFilter":
            operation = str(value.get("operation", "EQUAL")).upper()
            operators = {
                "EQUAL": "=",
                "LESS_THAN": "<",
                "LESS_THAN_OR_EQUAL": "<=",
                "GREATER_THAN": ">",
                "GREATER_THAN_OR_EQUAL": ">=",
            }
            if operation not in operators:
                raise AnalyticsToolError(
                    "invalid_argument",
                    f"Unsupported numericFilter operation: {operation}",
                )
            return (
                f"CAST({field_sql} AS DOUBLE) {operators[operation]} ?",
                [self._numeric_value(value.get("value"))],
            )
        if filter_type == "betweenFilter":
            low = self._numeric_value(value.get("fromValue"))
            high = self._numeric_value(value.get("toValue"))
            return f"CAST({field_sql} AS DOUBLE) BETWEEN ? AND ?", [low, high]
        raise AssertionError("Unknown Analytics filter type")

    @staticmethod
    def _numeric_value(value: Any) -> float:
        if isinstance(value, bool):
            raise AnalyticsToolError(
                "invalid_argument", "Numeric filter value must be a number"
            )
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Mapping):
            raw = value.get("int64Value", value.get("doubleValue"))
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                raise AnalyticsToolError(
                    "invalid_argument", "Numeric filter value must be numeric"
                )
            try:
                return float(raw)
            except ValueError as error:
                raise AnalyticsToolError(
                    "invalid_argument", "Numeric filter value must be numeric"
                ) from error
        raise AnalyticsToolError(
            "invalid_argument", "Numeric filter value must be numeric"
        )

    def _order_bys(
        self,
        order_bys: Sequence[Mapping[str, Any]],
        dimensions: Sequence[str],
        metrics: Sequence[str],
    ) -> list[dict[str, Any]]:
        if isinstance(order_bys, (str, bytes)) or not isinstance(order_bys, Sequence):
            raise AnalyticsToolError("invalid_argument", "order_bys must be a list")
        if len(order_bys) > self.config.max_order_bys:
            raise AnalyticsToolError(
                "invalid_argument",
                f"At most {self.config.max_order_bys} order_bys are allowed",
            )
        parsed: list[dict[str, Any]] = []
        for item in order_bys:
            if not isinstance(item, Mapping):
                raise AnalyticsToolError(
                    "invalid_argument", "Each order_by must be an object"
                )
            has_dimension = "dimension" in item
            has_metric = "metric" in item
            if has_dimension == has_metric:
                raise AnalyticsToolError(
                    "invalid_argument", "Each order_by needs one dimension or metric"
                )
            desc = item.get("desc", False)
            if not isinstance(desc, bool):
                raise AnalyticsToolError(
                    "invalid_argument", "order_by.desc must be a boolean"
                )
            if has_dimension:
                detail = item["dimension"]
                name = (
                    detail.get("dimensionName") if isinstance(detail, Mapping) else None
                )
                order_type = (
                    str(detail.get("orderType", "ALPHANUMERIC")).upper()
                    if isinstance(detail, Mapping)
                    else ""
                )
                if name not in dimensions:
                    raise AnalyticsToolError(
                        "invalid_argument",
                        f"Ordered dimension must be requested: {name}",
                    )
                if order_type not in {
                    "ALPHANUMERIC",
                    "CASE_INSENSITIVE_ALPHANUMERIC",
                    "NUMERIC",
                }:
                    raise AnalyticsToolError(
                        "invalid_argument",
                        f"Unsupported dimension order type: {order_type}",
                    )
                parsed.append(
                    {
                        "kind": "dimension",
                        "name": name,
                        "desc": desc,
                        "orderType": order_type,
                    }
                )
            else:
                detail = item["metric"]
                name = detail.get("metricName") if isinstance(detail, Mapping) else None
                if name not in metrics:
                    raise AnalyticsToolError(
                        "invalid_argument", f"Ordered metric must be requested: {name}"
                    )
                parsed.append({"kind": "metric", "name": name, "desc": desc})
        return parsed

    def _report_rows(
        self,
        property_id: str,
        dimensions: Sequence[str],
        metrics: Sequence[str],
        ranges: Sequence[Mapping[str, str]],
        dimension_filter_sql: str,
        dimension_filter_params: Sequence[Any],
        metric_filter_sql: str,
        metric_filter_params: Sequence[Any],
        relation: str,
        dimension_fields: Mapping[str, str],
        metric_fields: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        select_dimensions = [
            f'{dimension_fields[name]} AS "__d{index}"'
            for index, name in enumerate(dimensions)
        ]
        self._guard_unique(
            metrics,
            dimensions,
            ranges,
            property_id,
            dimension_filter_sql,
            dimension_filter_params,
            relation,
            dimension_fields,
        )
        select_metrics = [
            f'{metric_fields[name]} AS "__m{index}"'
            for index, name in enumerate(metrics)
        ]
        group_sql = (
            " GROUP BY " + ",".join(dimension_fields[name] for name in dimensions)
            if dimensions
            else ""
        )
        having_sql = f" HAVING {metric_filter_sql}" if metric_filter_sql else ""
        queries: list[str] = []
        params: list[Any] = []
        for range_index, item in enumerate(ranges):
            selections = [
                f"{range_index} AS __date_range",
                *select_dimensions,
                *select_metrics,
            ]
            date_column = "r.date" if self._aggregate_mode else "r.event_date"
            where = f"r.property_id = ? AND {date_column} BETWEEN ? AND ?"
            if self.config.snapshot_time:
                from .runtime import observed_where

                visible, observed_params = observed_where(
                    self,
                    (
                        relation.removeprefix("aggregate_")
                        if self._aggregate_mode
                        else "events"
                    ),
                )
                where += " AND (" + visible + ")"
            else:
                observed_params = []
            if dimension_filter_sql:
                where += f" AND ({dimension_filter_sql})"
            queries.append(
                f"SELECT {','.join(selections)} FROM {relation} r "
                f"WHERE {where}{group_sql}{having_sql}"
            )
            params.extend(
                [
                    property_id,
                    item["startDate"],
                    item["endDate"],
                    *observed_params,
                    *dimension_filter_params,
                    *metric_filter_params,
                ]
            )
        query = " UNION ALL ".join(f"({item})" for item in queries)
        query += f" LIMIT {self.config.max_group_rows + 1}"
        values = self._query(query, params)
        if len(values) > self.config.max_group_rows:
            raise AnalyticsToolError(
                "resource_exhausted",
                f"Report exceeds the configured {self.config.max_group_rows} grouped row limit",
            )
        rows: list[dict[str, Any]] = []
        for value in values:
            rows.append(
                {
                    "dateRange": int(value["__date_range"]),
                    "dimensions": {
                        name: value[f"__d{index}"]
                        for index, name in enumerate(dimensions)
                    },
                    "metrics": {
                        name: value[f"__m{index}"] for index, name in enumerate(metrics)
                    },
                }
            )
        return rows

    def _total_rows(
        self,
        property_id: str,
        metrics: Sequence[str],
        ranges: Sequence[Mapping[str, str]],
        dimension_filter_sql: str,
        dimension_filter_params: Sequence[Any],
        relation: str,
        metric_fields: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        self._guard_unique(
            metrics,
            (),
            ranges,
            property_id,
            dimension_filter_sql,
            dimension_filter_params,
            relation,
            {},
        )
        results: list[dict[str, Any]] = []
        selections = [
            f'{metric_fields[name]} AS "__m{index}"'
            for index, name in enumerate(metrics)
        ]
        for range_index, item in enumerate(ranges):
            date_column = "r.date" if self._aggregate_mode else "r.event_date"
            where = f"r.property_id = ? AND {date_column} BETWEEN ? AND ?"
            if self.config.snapshot_time:
                from .runtime import observed_where

                visible, observed_params = observed_where(
                    self,
                    (
                        relation.removeprefix("aggregate_")
                        if self._aggregate_mode
                        else "events"
                    ),
                )
                where += " AND (" + visible + ")"
            else:
                observed_params = []
            if dimension_filter_sql:
                where += f" AND ({dimension_filter_sql})"
            rows = self._query(
                f"SELECT {','.join(selections)} FROM {relation} r WHERE {where}",
                [
                    property_id,
                    item["startDate"],
                    item["endDate"],
                    *observed_params,
                    *dimension_filter_params,
                ],
            )
            value = rows[0]
            results.append(
                {
                    "dateRange": range_index,
                    "metrics": {
                        name: value[f"__m{index}"] for index, name in enumerate(metrics)
                    },
                }
            )
        return results

    def _guard_unique(
        self,
        metrics,
        dimensions,
        ranges,
        property_id,
        filter_sql,
        filter_params,
        relation,
        dimension_fields,
    ):
        from .runtime import USER_METRICS, observed_where

        if not self._aggregate_mode or not (
            set(metrics) & USER_METRICS
            or relation == "aggregate_events"
            and "sessions" in metrics
        ):
            return
        for period in ranges:
            where, params = observed_where(self, relation.removeprefix("aggregate_"))
            where += " AND r.date BETWEEN ? AND ?"
            params += [period["startDate"], period["endDate"]]
            if filter_sql:
                where += " AND (" + filter_sql + ")"
                params += list(filter_params)
            groups = (
                " GROUP BY " + ",".join(dimension_fields[d] for d in dimensions)
                if dimensions
                else ""
            )
            overlap = self._query(
                f"SELECT count(*) AS n FROM {relation} r WHERE {where}{groups} HAVING count(*)>1 LIMIT 1",
                params,
            )
            if overlap:
                raise AnalyticsToolError(
                    "unsupported_aggregation",
                    "Unique users or event-grain sessions cannot be combined across stored buckets. Request native rows or a single bucket.",
                )

    def _query(self, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        interrupted = threading.Event()

        def interrupt() -> None:
            interrupted.set()
            self._duck.interrupt()

        timer = threading.Timer(self.config.query_timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()
        try:
            cursor = self._duck.execute(query, list(params))
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except duckdb.Error as error:
            if interrupted.is_set():
                raise AnalyticsToolError(
                    "deadline_exceeded", "Analytics report exceeded its query timeout"
                ) from error
            if isinstance(error, (duckdb.OutOfMemoryException, duckdb.IOException)):
                raise AnalyticsToolError(
                    "resource_exhausted",
                    "Exact Analytics query exceeds memory or scratch limits",
                ) from error
            raise AnalyticsToolError(
                "invalid_argument", "Analytics could not run this report"
            ) from error
        finally:
            timer.cancel()
            timer.join()

    def _sort_rows(
        self,
        rows: list[dict[str, Any]],
        order_specs: Sequence[Mapping[str, Any]],
        dimensions: Sequence[str],
    ) -> list[dict[str, Any]]:
        specs = list(order_specs)
        ordered_fields = {(item["kind"], item["name"]) for item in specs}
        for name in dimensions:
            if ("dimension", name) not in ordered_fields:
                specs.append(
                    {
                        "kind": "dimension",
                        "name": name,
                        "desc": False,
                        "orderType": "ALPHANUMERIC",
                    }
                )
        specs.append({"kind": "dateRange", "name": "dateRange", "desc": False})

        result = list(rows)
        for spec in reversed(specs):
            kind = spec["kind"]
            name = spec["name"]

            def key(
                row: Mapping[str, Any],
                row_kind: str = kind,
                row_name: str = name,
                row_spec: Mapping[str, Any] = spec,
            ) -> tuple[int, Any]:
                if row_kind == "dateRange":
                    value: Any = row["dateRange"]
                elif row_kind == "metric":
                    value = row["metrics"].get(row_name)
                else:
                    value = row["dimensions"].get(row_name)
                    order_type = row_spec.get("orderType")
                    if order_type == "NUMERIC":
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            value = float("inf")
                    elif order_type == "CASE_INSENSITIVE_ALPHANUMERIC":
                        value = str(value or "").lower()
                    else:
                        value = str(value or "")
                return (value is None, value)

            result.sort(key=key, reverse=bool(spec["desc"]))
        return result

    def _metric_value(self, name: str, value: Any) -> str:
        if value is None:
            value = 0
        metric_type = self._metric_types[name]
        if metric_type == "TYPE_INTEGER":
            return str(round(float(value)))
        if metric_type == "TYPE_CURRENCY":
            return f"{float(value):.2f}"
        rendered = f"{float(value):.8f}".rstrip("0").rstrip(".")
        return rendered if rendered and rendered != "-0" else "0"

    def _response_row(
        self,
        row: Mapping[str, Any],
        dimensions: Sequence[str],
        metrics: Sequence[str],
        multiple_ranges: bool,
    ) -> dict[str, Any]:
        dimension_values = []
        if multiple_ranges:
            dimension_values.append({"value": str(row["dateRange"])})
        dimension_values.extend(
            {
                "value": (
                    ""
                    if row["dimensions"].get(name) is None
                    else str(row["dimensions"][name])
                )
            }
            for name in dimensions
        )
        return {
            "dimensionValues": dimension_values,
            "metricValues": [
                {"value": self._metric_value(name, row["metrics"].get(name))}
                for name in metrics
            ],
        }

    def _total_response_row(
        self,
        row: Mapping[str, Any],
        dimensions: Sequence[str],
        metrics: Sequence[str],
        multiple_ranges: bool,
    ) -> dict[str, Any]:
        dimension_values = []
        if multiple_ranges:
            dimension_values.append({"value": str(row["dateRange"])})
        dimension_values.extend({"value": "RESERVED_TOTAL"} for _ in dimensions)
        return {
            "dimensionValues": dimension_values,
            "metricValues": [
                {"value": self._metric_value(name, row["metrics"].get(name))}
                for name in metrics
            ],
        }
