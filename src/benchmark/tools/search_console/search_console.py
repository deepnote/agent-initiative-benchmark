"""Local Google Search Console wrappers backed by Parquet, SQLite, and JSON."""

from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import threading
import tomllib
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from benchmark import duckdb_runtime as duckdb


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")
    results: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        results.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return results[0], results[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "search_console"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "search_console"
DEFAULT_PROPERTY_ID = "sc-domain:pagemantle.com"


class SearchConsoleToolError(ValueError):
    """A predictable error that is safe to return to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class SearchConsoleConfig:
    """Paths, access identity, limits, and query settings."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    property_id: str = DEFAULT_PROPERTY_ID
    default_row_limit: int = 1_000
    max_row_limit: int = 25_000
    max_start_row: int = 1_000_000
    max_dimensions: int = 7
    max_filter_groups: int = 10
    max_filters: int = 20
    max_filter_expression_characters: int = 4_096
    max_query_days: int = 366
    query_timeout_seconds: float = 10.0
    sqlite_timeout_seconds: float = 30.0
    duckdb_threads: int = 4
    inspection_cache_buckets: int = 8
    duckdb_memory_limit: str | None = None
    duckdb_max_temp_directory_size: str | None = None

    @classmethod
    def from_env(cls) -> "SearchConsoleConfig":
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_SEARCH_CONSOLE_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_SEARCH_CONSOLE_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            property_id=os.getenv(
                "BENCHMARK_SEARCH_CONSOLE_PROPERTY_ID", DEFAULT_PROPERTY_ID
            ),
            default_row_limit=int(
                os.getenv("SEARCH_CONSOLE_DEFAULT_ROW_LIMIT", cls.default_row_limit)
            ),
            max_row_limit=int(
                os.getenv("SEARCH_CONSOLE_MAX_ROW_LIMIT", cls.max_row_limit)
            ),
            max_start_row=int(
                os.getenv("SEARCH_CONSOLE_MAX_START_ROW", cls.max_start_row)
            ),
            max_dimensions=int(
                os.getenv("SEARCH_CONSOLE_MAX_DIMENSIONS", cls.max_dimensions)
            ),
            max_filter_groups=int(
                os.getenv("SEARCH_CONSOLE_MAX_FILTER_GROUPS", cls.max_filter_groups)
            ),
            max_filters=int(os.getenv("SEARCH_CONSOLE_MAX_FILTERS", cls.max_filters)),
            max_filter_expression_characters=int(
                os.getenv(
                    "SEARCH_CONSOLE_MAX_FILTER_EXPRESSION_CHARACTERS",
                    cls.max_filter_expression_characters,
                )
            ),
            max_query_days=int(
                os.getenv("SEARCH_CONSOLE_MAX_QUERY_DAYS", cls.max_query_days)
            ),
            query_timeout_seconds=float(
                os.getenv(
                    "SEARCH_CONSOLE_QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds
                )
            ),
            sqlite_timeout_seconds=float(
                os.getenv(
                    "SEARCH_CONSOLE_SQLITE_TIMEOUT_SECONDS",
                    cls.sqlite_timeout_seconds,
                )
            ),
            duckdb_threads=int(
                os.getenv("SEARCH_CONSOLE_DUCKDB_THREADS", cls.duckdb_threads)
            ),
            inspection_cache_buckets=int(
                os.getenv(
                    "SEARCH_CONSOLE_INSPECTION_CACHE_BUCKETS",
                    cls.inspection_cache_buckets,
                )
            ),
        )

    def validate(self) -> None:
        integer_fields = (
            "default_row_limit",
            "max_row_limit",
            "max_start_row",
            "max_dimensions",
            "max_filter_groups",
            "max_filters",
            "max_filter_expression_characters",
            "max_query_days",
            "duckdb_threads",
            "inspection_cache_buckets",
        )
        for name in integer_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.default_row_limit > self.max_row_limit:
            raise ValueError("default_row_limit cannot exceed max_row_limit")
        if self.query_timeout_seconds <= 0 or self.sqlite_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")


DIMENSION_EXPRESSIONS = {
    "date": "CAST(p.date AS VARCHAR)",
    "query": "q.query_text",
    "page": "pg.url",
    "country": "lower(p.country_code)",
    "device": "p.device",
    "searchAppearance": "coalesce(p.search_appearance, '')",
}
DIMENSION_ALIASES = {
    "searchappearance": "searchAppearance",
    "search_appearance": "searchAppearance",
}
FILTER_OPERATORS = {
    "equals",
    "notEquals",
    "contains",
    "notContains",
    "includingRegex",
    "excludingRegex",
}
COUNTRY_CODES = (
    "USA",
    "IND",
    "GBR",
    "DEU",
    "CAN",
    "BRA",
    "FRA",
    "AUS",
    "NLD",
    "ESP",
    "CHN",
    "JPN",
    "KOR",
    "ITA",
    "MEX",
    "IDN",
    "TUR",
    "SAU",
    "ARE",
    "SGP",
    "IRL",
    "CHE",
    "SWE",
    "NOR",
    "DNK",
    "FIN",
    "POL",
    "AUT",
    "BEL",
    "PRT",
    "CZE",
    "GRC",
    "ISR",
    "ZAF",
    "EGY",
    "NGA",
    "KEN",
    "GHA",
    "MAR",
    "DZA",
    "TUN",
    "ETH",
    "TZA",
    "UGA",
    "RWA",
    "SEN",
    "CIV",
    "CMR",
    "ZMB",
    "ZWE",
    "BWA",
    "NAM",
    "MOZ",
    "AGO",
    "COD",
    "PAK",
    "BGD",
    "LKA",
    "NPL",
    "MMR",
    "THA",
    "VNM",
    "MYS",
    "PHL",
    "KHM",
    "LAO",
    "TWN",
    "HKG",
    "NZL",
    "ARG",
    "CHL",
    "COL",
    "PER",
    "ECU",
    "URY",
    "PRY",
    "BOL",
    "VEN",
    "CRI",
    "PAN",
    "GTM",
    "HND",
    "SLV",
    "NIC",
    "DOM",
    "JAM",
    "TTO",
    "BHS",
    "BRB",
    "ISL",
    "LUX",
    "LIE",
    "MCO",
    "MLT",
    "CYP",
    "EST",
    "LVA",
    "LTU",
    "SVK",
    "SVN",
    "HRV",
    "SRB",
    "BIH",
    "MKD",
    "ALB",
    "BGR",
    "ROU",
    "HUN",
    "UKR",
    "GEO",
    "ARM",
    "AZE",
    "KAZ",
    "UZB",
    "QAT",
    "KWT",
    "BHR",
    "OMN",
    "JOR",
    "LBN",
)
PAGE_SCHEMA = (
    "page_id VARCHAR, property_id VARCHAR, url VARCHAR, title VARCHAR, "
    "page_type VARCHAR, page_rank INTEGER, active BOOLEAN, created_date DATE, "
    "canonical_url VARCHAR, inspection_bucket VARCHAR, inspection_verdict VARCHAR, "
    "coverage_state VARCHAR, last_crawl_time VARCHAR"
)
QUERY_SCHEMA = (
    "query_id VARCHAR, query_text VARCHAR, query_type VARCHAR, query_rank INTEGER, "
    "privacy_hidden BOOLEAN"
)
OVERLAY_COLUMNS = (
    "source_event_id",
    "operation",
    "property_id",
    "date",
    "query_id",
    "page_id",
    "country_code",
    "device",
    "search_type",
    "search_appearance",
    "clicks",
    "impressions",
    "position_sum",
)
OVERLAY_SQL = """
CREATE TABLE IF NOT EXISTS performance_rows (
    source_event_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    property_id TEXT,
    date TEXT,
    query_id TEXT,
    page_id TEXT,
    country_code TEXT,
    device TEXT,
    search_type TEXT,
    search_appearance TEXT,
    clicks INTEGER,
    impressions INTEGER,
    position_sum REAL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gsc_overlay_date ON performance_rows(date);
CREATE INDEX IF NOT EXISTS idx_gsc_overlay_page ON performance_rows(page_id);
CREATE INDEX IF NOT EXISTS idx_gsc_overlay_query ON performance_rows(query_id);
"""


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchConsoleToolError(
            "data_error", f"Could not read Search Console data: {path.name}"
        ) from error


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise SearchConsoleToolError("invalid_date", f"{field} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise SearchConsoleToolError(
            "invalid_date", f"{field} must be a valid YYYY-MM-DD date"
        ) from error


class SearchConsoleTools:
    """Read effective Search Console state without changing the baseline."""

    def __init__(
        self, config: SearchConsoleConfig | None = None, *, acquire_family=None
    ):
        self.config = config or SearchConsoleConfig.from_env()
        self.config.validate()
        self._lock = threading.RLock()
        self._acquire_family = acquire_family
        self._families = set()
        self._validate_files()
        properties = _load_json(self.config.baseline_dir / "properties.json")
        available = {item["id"]: item for item in properties.get("properties", [])}
        if self.config.property_id not in available:
            raise SearchConsoleToolError(
                "permission_denied",
                f"Configured property is not available: {self.config.property_id}",
            )
        self._property = available[self.config.property_id]
        self._duck = duckdb.connect(
            database=":memory:",
            config={
                k: v
                for k, v in {
                    "memory_limit": self.config.duckdb_memory_limit,
                    "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
                }.items()
                if v is not None
            },
        )
        self._duck.execute(f"PRAGMA threads={self.config.duckdb_threads}")
        self._overlay_signature: tuple[int, int, int, int] | None = None
        self._inspection_overrides: dict[str, Mapping[str, Any]] = {}
        self._inspection_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        try:
            self._create_views()
            self._reload_overlay(force=True)
        except BaseException:
            self._duck.close()
            raise

    def _validate_files(self) -> None:
        required = (
            self.config.baseline_dir / "properties.json",
            self.config.baseline_dir / "pages.parquet",
            self.config.baseline_dir / "queries.parquet",
            self.config.baseline_dir / "performance",
            self.config.baseline_dir / "url-inspection",
            self.config.overlay_dir / "overlay.sqlite",
            self.config.overlay_dir / "inspection-overrides.json",
        )
        if self._acquire_family is not None:
            required = (required[0], *required[-2:])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Search Console data: " + ", ".join(missing)
            )

    def _create_views(self) -> None:
        self._duck.execute(
            """
            CREATE TABLE overlay_performance (
                source_event_id VARCHAR, operation VARCHAR, property_id VARCHAR,
                date DATE, query_id VARCHAR, page_id VARCHAR, country_code VARCHAR,
                device VARCHAR, search_type VARCHAR, search_appearance VARCHAR,
                clicks BIGINT, impressions BIGINT, position_sum DOUBLE
            )
            """
        )

        if self._acquire_family is None:
            for family in ("pages", "queries", "performance"):
                self._ensure(family)

    def _ensure(self, family):
        if family in self._families:
            return
        if self._acquire_family is not None:
            self._acquire_family(family)
        baseline = self.config.baseline_dir
        if family == "performance":
            paths = sorted((baseline / "performance").rglob("*.parquet"))
            if not paths:
                raise SearchConsoleToolError(
                    "data_error", "Required performance files are missing"
                )
            # Explicit inventory, never an implicit sample or a partial glob.
            files = "[" + ",".join("'" + _sql_path(p) + "'" for p in paths) + "]"
            self._duck.execute(
                "CREATE VIEW baseline_performance AS SELECT * EXCLUDE (year, month) FROM read_parquet("
                + files
                + ", hive_partitioning=true, union_by_name=true)"
            )
        elif family in ("pages", "queries"):
            self._duck.execute(
                f"CREATE VIEW {family} AS SELECT * FROM read_parquet('{_sql_path(baseline / (family+'.parquet'))}')"
            )
        self._families.add(family)

    def _performance_view(self):
        self._ensure("performance")
        self._duck.execute(
            """
            CREATE OR REPLACE VIEW effective_performance AS
            SELECT source_event_id, property_id, date, query_id, page_id,
                   country_code, device, search_type, search_appearance,
                   clicks, impressions, position_sum, 'overlay' AS source
            FROM overlay_performance WHERE operation = 'upsert'
            UNION ALL
            SELECT b.source_event_id, b.property_id, b.date, b.query_id, b.page_id,
                   b.country_code, b.device, b.search_type, b.search_appearance,
                   b.clicks, b.impressions, b.position_sum, 'baseline' AS source
            FROM baseline_performance b
            WHERE NOT EXISTS (
                SELECT 1 FROM overlay_performance o
                WHERE o.source_event_id = b.source_event_id
            )
            """
        )

    def _current_overlay_signature(self) -> tuple[int, int, int, int]:
        database = self.config.overlay_dir / "overlay.sqlite"
        inspections = self.config.overlay_dir / "inspection-overrides.json"
        database_stat = database.stat()
        inspection_stat = inspections.stat()
        return (
            database_stat.st_mtime_ns,
            database_stat.st_size,
            inspection_stat.st_mtime_ns,
            inspection_stat.st_size,
        )

    def _reload_overlay(self, *, force: bool = False) -> None:
        signature = self._current_overlay_signature()
        if not force and signature == self._overlay_signature:
            return
        database = self.config.overlay_dir / "overlay.sqlite"
        uri = f"{database.as_uri()}?mode=ro"
        with closing(
            sqlite3.connect(
                uri,
                uri=True,
                timeout=self.config.sqlite_timeout_seconds,
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(
                f"SELECT {','.join(OVERLAY_COLUMNS)} FROM performance_rows ORDER BY source_event_id"
            ).fetchall()
        self._duck.execute("DELETE FROM overlay_performance")
        if rows:
            self._duck.executemany(
                f"INSERT INTO overlay_performance VALUES ({','.join('?' for _ in OVERLAY_COLUMNS)})",
                [tuple(row[column] for column in OVERLAY_COLUMNS) for row in rows],
            )
        overrides = _load_json(self.config.overlay_dir / "inspection-overrides.json")
        self._inspection_overrides = {
            item["inspection_url"]: item
            for item in overrides.get("inspection_overrides", [])
        }
        self._overlay_signature = signature

    def close(self) -> None:
        with self._lock:
            self._duck.close()

    def __enter__(self) -> "SearchConsoleTools":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _site_url(self, value: Any) -> str:
        site_url = self.config.property_id if value in {None, ""} else str(value)
        if site_url != self.config.property_id:
            raise SearchConsoleToolError(
                "permission_denied",
                "The agent cannot access this Search Console property",
            )
        return site_url

    def _dimensions(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            raw = [value.strip() for value in values.split(",") if value.strip()]
        elif isinstance(values, Sequence):
            raw = [str(value).strip() for value in values]
        else:
            raise SearchConsoleToolError(
                "invalid_argument",
                "dimensions must be a list or comma-separated string",
            )
        dimensions = [DIMENSION_ALIASES.get(value.lower(), value) for value in raw]
        if len(dimensions) > self.config.max_dimensions:
            raise SearchConsoleToolError(
                "invalid_argument",
                f"dimensions can contain at most {self.config.max_dimensions} values",
            )
        if len(set(dimensions)) != len(dimensions):
            raise SearchConsoleToolError(
                "invalid_argument", "dimensions cannot contain duplicates"
            )
        invalid = [value for value in dimensions if value not in DIMENSION_EXPRESSIONS]
        if invalid:
            raise SearchConsoleToolError(
                "invalid_argument", f"Unsupported dimension: {invalid[0]}"
            )
        return dimensions

    def _filters(self, groups: Any) -> tuple[list[str], list[Any], set[str]]:
        if groups is None:
            return [], [], set()
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
            raise SearchConsoleToolError(
                "invalid_filter", "dimension_filter_groups must be a list"
            )
        if len(groups) > self.config.max_filter_groups:
            raise SearchConsoleToolError(
                "invalid_filter",
                f"At most {self.config.max_filter_groups} filter groups are allowed",
            )
        clauses: list[str] = []
        params: list[Any] = []
        dimensions: set[str] = set()
        filter_count = 0
        for group in groups:
            if not isinstance(group, Mapping) or group.get("groupType", "and") != "and":
                raise SearchConsoleToolError(
                    "invalid_filter",
                    "Only filter groups with groupType 'and' are supported",
                )
            filters = group.get("filters", [])
            if not isinstance(filters, Sequence) or isinstance(filters, (str, bytes)):
                raise SearchConsoleToolError("invalid_filter", "filters must be a list")
            group_clauses: list[str] = []
            for item in filters:
                filter_count += 1
                if filter_count > self.config.max_filters:
                    raise SearchConsoleToolError(
                        "invalid_filter",
                        f"At most {self.config.max_filters} filters are allowed",
                    )
                if not isinstance(item, Mapping):
                    raise SearchConsoleToolError(
                        "invalid_filter", "Each filter must be an object"
                    )
                raw_dimension = str(item.get("dimension", ""))
                dimension = DIMENSION_ALIASES.get(raw_dimension.lower(), raw_dimension)
                if dimension not in DIMENSION_EXPRESSIONS or dimension == "date":
                    raise SearchConsoleToolError(
                        "invalid_filter",
                        f"Unsupported filter dimension: {raw_dimension}",
                    )
                operator = str(item.get("operator", "equals"))
                if operator not in FILTER_OPERATORS:
                    raise SearchConsoleToolError(
                        "invalid_filter", f"Unsupported filter operator: {operator}"
                    )
                expression = item.get("expression")
                if not isinstance(expression, str) or not expression:
                    raise SearchConsoleToolError(
                        "invalid_filter", "Filter expression must be a non-empty string"
                    )
                if len(expression) > self.config.max_filter_expression_characters:
                    raise SearchConsoleToolError(
                        "invalid_filter",
                        "Filter expression exceeds the configured character limit",
                    )
                column = DIMENSION_EXPRESSIONS[dimension]
                dimensions.add(dimension)
                if operator == "equals":
                    group_clauses.append(f"{column} = ?")
                    params.append(expression)
                elif operator == "notEquals":
                    group_clauses.append(f"{column} <> ?")
                    params.append(expression)
                elif operator == "contains":
                    group_clauses.append(f"contains(lower({column}), lower(?))")
                    params.append(expression)
                elif operator == "notContains":
                    group_clauses.append(f"NOT contains(lower({column}), lower(?))")
                    params.append(expression)
                elif operator in {"includingRegex", "excludingRegex"}:
                    try:
                        self._duck.execute(
                            "SELECT regexp_matches('', ?)", [expression]
                        ).fetchone()
                    except duckdb.Error as error:
                        raise SearchConsoleToolError(
                            "invalid_regex",
                            "Filter expression is not a valid RE2 pattern",
                        ) from error
                    sql = f"regexp_matches({column}, ?)"
                    group_clauses.append(
                        sql if operator == "includingRegex" else f"NOT {sql}"
                    )
                    params.append(expression)
            if group_clauses:
                clauses.append("(" + " AND ".join(group_clauses) + ")")
        return clauses, params, dimensions

    def _execute_rows(self, sql: str, params: Iterable[Any]) -> list[dict[str, Any]]:
        timed_out = threading.Event()

        def interrupt() -> None:
            timed_out.set()
            self._duck.interrupt()

        timer = threading.Timer(self.config.query_timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()
        try:
            cursor = self._duck.execute(sql, list(params))
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except duckdb.Error as error:
            if timed_out.is_set():
                raise SearchConsoleToolError(
                    "timeout", "Search Console query exceeded its time limit"
                ) from error
            raise SearchConsoleToolError(
                "data_error", "Search Console could not complete the query"
            ) from error
        finally:
            timer.cancel()

    def search_console_query_performance(
        self,
        start_date: str,
        end_date: str,
        *,
        site_url: str | None = None,
        dimensions: Sequence[str] | str | None = None,
        dimension_filter_groups: Sequence[Mapping[str, Any]] | None = None,
        search_type: str = "web",
        aggregation_type: str = "auto",
        row_limit: int | None = None,
        start_row: int = 0,
    ) -> dict[str, Any]:
        """Aggregate effective search performance using Search Console rules."""

        with self._lock:
            for family in ("pages", "queries", "performance"):
                self._ensure(family)
            self._performance_view()
            self._reload_overlay()
            property_id = self._site_url(site_url)
            start = _parse_date(start_date, "start_date")
            end = _parse_date(end_date, "end_date")
            if start > end:
                raise SearchConsoleToolError(
                    "invalid_date", "start_date must be on or before end_date"
                )
            if (end - start).days + 1 > self.config.max_query_days:
                raise SearchConsoleToolError(
                    "invalid_date",
                    f"Date range cannot exceed {self.config.max_query_days} days",
                )
            if search_type != "web":
                raise SearchConsoleToolError(
                    "invalid_argument", "This fixture contains only web search data"
                )
            selected_dimensions = self._dimensions(dimensions)
            filter_clauses, filter_params, filter_dimensions = self._filters(
                dimension_filter_groups
            )
            has_page = "page" in selected_dimensions or "page" in filter_dimensions
            if aggregation_type not in {"auto", "byPage", "byProperty"}:
                raise SearchConsoleToolError(
                    "invalid_argument",
                    "aggregation_type must be auto, byPage, or byProperty",
                )
            if aggregation_type == "byProperty" and has_page:
                raise SearchConsoleToolError(
                    "invalid_argument",
                    "byProperty cannot be used with a page dimension or page filter",
                )
            response_aggregation = (
                "byPage"
                if aggregation_type == "byPage"
                or (aggregation_type == "auto" and has_page)
                else "byProperty"
            )
            limit = self.config.default_row_limit if row_limit is None else row_limit
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 1 <= limit <= self.config.max_row_limit
            ):
                raise SearchConsoleToolError(
                    "invalid_argument",
                    f"row_limit must be between 1 and {self.config.max_row_limit}",
                )
            if (
                not isinstance(start_row, int)
                or isinstance(start_row, bool)
                or not 0 <= start_row <= self.config.max_start_row
            ):
                raise SearchConsoleToolError(
                    "invalid_argument",
                    f"start_row must be between 0 and {self.config.max_start_row}",
                )
            where = [
                "p.property_id = ?",
                "p.date BETWEEN ? AND ?",
                "p.search_type = ?",
            ]
            params: list[Any] = [
                property_id,
                start.isoformat(),
                end.isoformat(),
                search_type,
            ]
            where.extend(filter_clauses)
            params.extend(filter_params)
            if "query" in selected_dimensions or "query" in filter_dimensions:
                where.append("q.privacy_hidden = false")
            select_keys = [
                f"{DIMENSION_EXPRESSIONS[dimension]} AS key_{index}"
                for index, dimension in enumerate(selected_dimensions)
            ]
            group_keys = [f"key_{index}" for index in range(len(selected_dimensions))]
            select_prefix = ", ".join(select_keys) + (", " if select_keys else "")
            group_sql = "GROUP BY " + ", ".join(group_keys) if group_keys else ""
            if "date" in selected_dimensions:
                date_key = f"key_{selected_dimensions.index('date')}"
                other_keys = [key for key in group_keys if key != date_key]
                order_sql = f"ORDER BY {date_key} ASC"
                if other_keys:
                    order_sql += ", clicks DESC, " + ", ".join(
                        f"{key} ASC" for key in other_keys
                    )
            else:
                order_sql = "ORDER BY clicks DESC"
                if group_keys:
                    order_sql += ", " + ", ".join(f"{key} ASC" for key in group_keys)
            sql = f"""
                SELECT {select_prefix}
                       sum(p.clicks)::BIGINT AS clicks,
                       sum(p.impressions)::BIGINT AS impressions,
                       CASE WHEN sum(p.impressions) = 0 THEN 0
                            ELSE sum(p.clicks)::DOUBLE / sum(p.impressions) END AS ctr,
                       CASE WHEN sum(p.impressions) = 0 THEN 0
                            ELSE sum(p.position_sum ORDER BY p.source_event_id) / sum(p.impressions) END AS position
                FROM effective_performance p
                JOIN pages pg ON pg.page_id = p.page_id AND pg.property_id = p.property_id
                JOIN queries q ON q.query_id = p.query_id
                WHERE {' AND '.join(where)}
                {group_sql}
                HAVING sum(p.impressions) > 0
                {order_sql}
                LIMIT ? OFFSET ?
            """
            query_rows = self._execute_rows(sql, [*params, limit + 1, start_row])
            has_more = len(query_rows) > limit
            query_rows = query_rows[:limit]
            rows = [
                {
                    "keys": [
                        str(row[f"key_{index}"])
                        for index in range(len(selected_dimensions))
                    ],
                    "clicks": int(row["clicks"]),
                    "impressions": int(row["impressions"]),
                    "ctr": float(row["ctr"]),
                    "position": float(row["position"]),
                }
                for row in query_rows
            ]
            return {
                "ok": True,
                "siteUrl": property_id,
                "rows": rows,
                "responseAggregationType": response_aggregation,
                "dimensions": selected_dimensions,
                "startRow": start_row,
                "rowLimit": limit,
                "hasMore": has_more,
                "nextStartRow": start_row + len(rows) if has_more else None,
            }

    @staticmethod
    def _normalize_url(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SearchConsoleToolError(
                "invalid_url", "inspection_url must be a fully-qualified URL"
            )
        try:
            parsed = urlsplit(value.strip())
        except ValueError as error:
            raise SearchConsoleToolError(
                "invalid_url", "inspection_url is malformed"
            ) from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise SearchConsoleToolError(
                "invalid_url",
                "inspection_url must be a fully-qualified HTTP or HTTPS URL",
            )
        if parsed.fragment:
            raise SearchConsoleToolError(
                "invalid_url", "inspection_url cannot contain a fragment"
            )
        host = parsed.hostname.lower()
        try:
            parsed_port = parsed.port
        except ValueError as error:
            raise SearchConsoleToolError(
                "invalid_url", "inspection_url contains an invalid port"
            ) from error
        port = f":{parsed_port}" if parsed_port else ""
        path = parsed.path or "/"
        return urlunsplit((parsed.scheme.lower(), host + port, path, parsed.query, ""))

    def _inspection_bucket(self, bucket: str) -> dict[str, Any]:
        cached = self._inspection_cache.get(bucket)
        if cached is not None:
            self._inspection_cache.move_to_end(bucket)
            return cached
        if not re.fullmatch(r"\d{2}", bucket):
            raise SearchConsoleToolError(
                "data_error", "URL inspection bucket is invalid"
            )
        self._ensure("inspection:" + bucket)
        root = (self.config.baseline_dir / "url-inspection").resolve()
        path = (root / f"bucket={bucket}" / "snapshots.json").resolve()
        if root not in path.parents or not path.is_file():
            raise SearchConsoleToolError(
                "data_error", "URL inspection snapshot is missing"
            )
        payload = _load_json(path)
        indexed = {
            item["inspection_url"]: item for item in payload.get("snapshots", [])
        }
        self._inspection_cache[bucket] = indexed
        self._inspection_cache.move_to_end(bucket)
        while len(self._inspection_cache) > self.config.inspection_cache_buckets:
            self._inspection_cache.popitem(last=False)
        return indexed

    def search_console_inspect_url(
        self,
        inspection_url: str,
        *,
        site_url: str | None = None,
        language_code: str = "en-US",
    ) -> dict[str, Any]:
        """Return the stored Google index snapshot for one owned URL."""

        with self._lock:
            self._ensure("pages")
            self._reload_overlay()
            property_id = self._site_url(site_url)
            normalized = self._normalize_url(inspection_url)
            domain = str(self._property["domain"]).lower()
            host = urlsplit(normalized).hostname or ""
            if host != domain and not host.endswith("." + domain):
                raise SearchConsoleToolError(
                    "permission_denied", "The inspected URL is outside this property"
                )
            if not isinstance(language_code, str) or not re.fullmatch(
                r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language_code
            ):
                raise SearchConsoleToolError(
                    "invalid_argument", "language_code must be a valid BCP-47 tag"
                )
            page_rows = self._execute_rows(
                "SELECT page_id, inspection_bucket FROM pages WHERE property_id = ? AND url = ? LIMIT 2",
                [property_id, normalized],
            )
            if not page_rows:
                raise SearchConsoleToolError(
                    "url_not_found", "No URL inspection snapshot exists for this URL"
                )
            if len(page_rows) != 1:
                raise SearchConsoleToolError("data_error", "Ambiguous inspection URL")
            override = self._inspection_overrides.get(normalized)
            if override is not None:
                if override.get("operation") == "delete":
                    raise SearchConsoleToolError(
                        "url_not_found",
                        "No URL inspection snapshot exists for this URL",
                    )
                snapshot = override.get("snapshot")
                observed_at = override.get("observed_at")
                source = "overlay"
            else:
                bucket = str(page_rows[0]["inspection_bucket"])
                record = self._inspection_bucket(bucket).get(normalized)
                if record is None:
                    raise SearchConsoleToolError(
                        "url_not_found",
                        "No URL inspection snapshot exists for this URL",
                    )
                snapshot = record.get("snapshot")
                observed_at = record.get("observed_at")
                source = "baseline"
            if not isinstance(snapshot, Mapping):
                raise SearchConsoleToolError(
                    "data_error", "URL inspection snapshot is invalid"
                )
            return {
                "ok": True,
                "inspectionUrl": normalized,
                "siteUrl": property_id,
                "languageCode": language_code,
                "source": source,
                "inspectionResult": copy.deepcopy(snapshot),
                "observedAt": observed_at,
                "observationTimeAvailable": observed_at is not None,
            }


_DEFAULT_TOOLS: SearchConsoleTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> SearchConsoleTools:
    global _DEFAULT_TOOLS
    with _DEFAULT_LOCK:
        if _DEFAULT_TOOLS is None:
            _DEFAULT_TOOLS = SearchConsoleTools()
        return _DEFAULT_TOOLS


def search_console_query_performance(
    start_date: str,
    end_date: str,
    *,
    site_url: str | None = None,
    dimensions: Sequence[str] | str | None = None,
    dimension_filter_groups: Sequence[Mapping[str, Any]] | None = None,
    search_type: str = "web",
    aggregation_type: str = "auto",
    row_limit: int | None = None,
    start_row: int = 0,
) -> dict[str, Any]:
    return _default_tools().search_console_query_performance(
        start_date,
        end_date,
        site_url=site_url,
        dimensions=dimensions,
        dimension_filter_groups=dimension_filter_groups,
        search_type=search_type,
        aggregation_type=aggregation_type,
        row_limit=row_limit,
        start_row=start_row,
    )


def search_console_inspect_url(
    inspection_url: str,
    *,
    site_url: str | None = None,
    language_code: str = "en-US",
) -> dict[str, Any]:
    return _default_tools().search_console_inspect_url(
        inspection_url, site_url=site_url, language_code=language_code
    )
