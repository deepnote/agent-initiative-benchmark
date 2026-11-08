"""Local BigQuery benchmark wrappers backed by Parquet and SQLite."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import tomllib
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark import duckdb_runtime as duckdb


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROJECT_IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")
    resolved: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        resolved.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return resolved[0], resolved[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "bigquery"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "bigquery"


class BigQueryToolError(ValueError):
    """A stable error safe to return to a benchmark agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


DEFAULT_ALLOWED_DATASETS = (
    "customer_success",
    "finance",
    "marketing",
    "operations",
    "product_analytics",
    "crm_mirror",
    "billing_mirror",
    "product_events",
    "web_analytics",
    "organic_search",
    "feature_management",
    "site_operations",
    "warehouse_audit",
)


@dataclass(frozen=True)
class BigQueryConfig:
    """Paths, identity, access, limits, and execution settings."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    project_id: str = "pagemantle-analytics"
    location: str = "US"
    agent_principal: str = "agent@pagemantle.com"
    allowed_datasets: tuple[str, ...] = DEFAULT_ALLOWED_DATASETS
    default_page_size: int = 50
    max_page_size: int = 100
    max_query_characters: int = 20_000
    query_timeout_seconds: float = 2.0
    max_query_timeout_seconds: float = 10.0
    sqlite_timeout_seconds: float = 30.0
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "256MB"
    duckdb_max_temp_directory_size: str = "1GB"

    @classmethod
    def from_env(cls) -> "BigQueryConfig":
        allowed = os.getenv("BIGQUERY_ALLOWED_DATASETS")
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_BIGQUERY_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_BIGQUERY_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            project_id=os.getenv("BIGQUERY_PROJECT_ID", cls.project_id),
            location=os.getenv("BIGQUERY_LOCATION", cls.location),
            agent_principal=os.getenv("BIGQUERY_AGENT_PRINCIPAL", cls.agent_principal),
            allowed_datasets=(
                tuple(item.strip() for item in allowed.split(",") if item.strip())
                if allowed is not None
                else cls.allowed_datasets
            ),
            default_page_size=int(
                os.getenv("BIGQUERY_DEFAULT_PAGE_SIZE", cls.default_page_size)
            ),
            max_page_size=int(os.getenv("BIGQUERY_MAX_PAGE_SIZE", cls.max_page_size)),
            max_query_characters=int(
                os.getenv("BIGQUERY_MAX_QUERY_CHARACTERS", cls.max_query_characters)
            ),
            query_timeout_seconds=float(
                os.getenv("BIGQUERY_QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds)
            ),
            max_query_timeout_seconds=float(
                os.getenv(
                    "BIGQUERY_MAX_QUERY_TIMEOUT_SECONDS",
                    cls.max_query_timeout_seconds,
                )
            ),
            sqlite_timeout_seconds=float(
                os.getenv("BIGQUERY_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
            duckdb_threads=int(
                os.getenv("BIGQUERY_DUCKDB_THREADS", cls.duckdb_threads)
            ),
            duckdb_memory_limit=os.getenv(
                "BIGQUERY_DUCKDB_MEMORY_LIMIT", cls.duckdb_memory_limit
            ),
            duckdb_max_temp_directory_size=os.getenv(
                "BIGQUERY_DUCKDB_MAX_TEMP_DIRECTORY_SIZE",
                cls.duckdb_max_temp_directory_size,
            ),
        )

    def validate(self) -> None:
        if not PROJECT_IDENTIFIER.fullmatch(self.project_id):
            raise ValueError("project_id is not a valid benchmark project ID")
        if not self.location.strip():
            raise ValueError("location must not be empty")
        if not self.agent_principal.strip():
            raise ValueError("agent_principal must not be empty")
        if not self.allowed_datasets:
            raise ValueError("allowed_datasets must not be empty")
        if any(not IDENTIFIER.fullmatch(item) for item in self.allowed_datasets):
            raise ValueError("allowed_datasets contains an invalid dataset ID")
        if len(self.allowed_datasets) != len(set(self.allowed_datasets)):
            raise ValueError("allowed_datasets must not contain duplicates")
        for name in (
            "default_page_size",
            "max_page_size",
            "max_query_characters",
            "duckdb_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.default_page_size > self.max_page_size:
            raise ValueError("default_page_size must not exceed max_page_size")
        if self.query_timeout_seconds <= 0 or self.max_query_timeout_seconds <= 0:
            raise ValueError("query timeouts must be positive")
        if self.query_timeout_seconds > self.max_query_timeout_seconds:
            raise ValueError(
                "query_timeout_seconds must not exceed max_query_timeout_seconds"
            )
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")


OVERLAY_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rows (
    dataset_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    primary_key TEXT NOT NULL,
    row_json TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, table_id, primary_key)
);

CREATE INDEX IF NOT EXISTS rows_table_idx
    ON rows(dataset_id, table_id, operation, primary_key);
"""


BIGQUERY_TO_DUCKDB = {
    "STRING": "VARCHAR",
    "INTEGER": "BIGINT",
    "INT64": "BIGINT",
    "FLOAT": "DOUBLE",
    "FLOAT64": "DOUBLE",
    "BOOLEAN": "BOOLEAN",
    "BOOL": "BOOLEAN",
    "TIMESTAMP": "TIMESTAMP",
    "DATE": "DATE",
    "NUMERIC": "DECIMAL(38,9)",
    "JSON": "VARCHAR",
}

DUCKDB_TO_BIGQUERY = {
    "VARCHAR": "STRING",
    "BIGINT": "INT64",
    "INTEGER": "INT64",
    "DOUBLE": "FLOAT64",
    "FLOAT": "FLOAT64",
    "BOOLEAN": "BOOL",
    "TIMESTAMP": "TIMESTAMP",
    "DATE": "DATE",
    "DECIMAL": "NUMERIC",
}

BLOCKED_KEYWORDS = {
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "CALL",
    "CHECKPOINT",
    "COPY",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "EXPORT",
    "IMPORT",
    "INSERT",
    "INSTALL",
    "LOAD",
    "MERGE",
    "PRAGMA",
    "RESET",
    "SET",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}

BLOCKED_FUNCTIONS = {
    "current_setting",
    "getenv",
    "glob",
    "parquet_scan",
    "read_blob",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_parquet",
    "read_text",
    "query",
    "query_table",
    "query_table_range",
    "sqlite_scan",
    "sqlite_query",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _quote_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid identifier: {value}")
    return f'"{value}"'


def _duckdb_type(field: Mapping[str, Any]) -> str:
    kind = str(field.get("type", "")).upper()
    try:
        return BIGQUERY_TO_DUCKDB[kind]
    except KeyError as error:
        raise ValueError(f"Unsupported BigQuery field type: {kind}") from error


def _normalize_value(value: Any, field: Mapping[str, Any]) -> Any:
    if value is None:
        return None
    kind = str(field["type"]).upper()
    if kind == "JSON" and not isinstance(value, str):
        return _json(value)
    if kind in {"INTEGER", "INT64"}:
        return int(value)
    if kind in {"FLOAT", "FLOAT64"}:
        return float(value)
    if kind in {"BOOLEAN", "BOOL"}:
        return bool(value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _validate_field(field: Mapping[str, Any]) -> None:
    name = field.get("name")
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ValueError("Every field needs a valid name")
    _duckdb_type(field)
    if str(field.get("mode", "NULLABLE")).upper() not in {"NULLABLE", "REQUIRED"}:
        raise ValueError(f"Unsupported mode for field {name}")


def build_bigquery_overlay_from_seed(
    seed_path: Path, overlay_dir: Path
) -> dict[str, Any]:
    """Build the writable SQLite row delta and JSON schema delta."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise ValueError("BigQuery overlay seed must be a JSON object")
    rows = seed.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("BigQuery overlay rows must be a list")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / "overlay.sqlite"
    if overlay_path.exists():
        overlay_path.unlink()
    connection = sqlite3.connect(overlay_path)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("format_version", "1"),
        )
        for item in rows:
            dataset_id = item.get("dataset_id")
            table_id = item.get("table_id")
            operation = item.get("operation", "upsert")
            primary_key = item.get("primary_key")
            row = item.get("row", {})
            if not isinstance(dataset_id, str) or not IDENTIFIER.fullmatch(dataset_id):
                raise ValueError("Overlay row has an invalid dataset ID")
            if not isinstance(table_id, str) or not IDENTIFIER.fullmatch(table_id):
                raise ValueError("Overlay row has an invalid table ID")
            if operation not in {"upsert", "delete"}:
                raise ValueError("Overlay operation must be upsert or delete")
            if not isinstance(primary_key, dict) or not primary_key:
                raise ValueError("Overlay row needs a primary_key object")
            if not isinstance(row, dict):
                raise ValueError("Overlay row must be an object")
            connection.execute(
                """
                INSERT INTO rows(
                    dataset_id, table_id, primary_key, row_json, operation, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    table_id,
                    _json(primary_key),
                    _json(row),
                    operation,
                    item.get("updated_at", "2026-09-01T12:00:00Z"),
                ),
            )
        connection.commit()
    finally:
        connection.close()

    changes = {
        "format_version": 1,
        "tables": seed.get("schema_changes", []),
    }
    changes_path = overlay_dir / "schema_changes.json"
    changes_path.write_text(
        json.dumps(changes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"rows": len(rows), "schema_changes": len(changes["tables"])}


def _strip_comments(sql: str) -> str:
    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            output.append(char)
            if char == quote:
                if next_char == quote:
                    output.append(next_char)
                    index += 1
                else:
                    quote = None
            elif char == "\\" and next_char:
                output.append(next_char)
                index += 1
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "-" and next_char == "-":
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            output.append(" ")
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(sql) and sql[index : index + 2] != "*/":
                index += 1
            if index + 1 >= len(sql):
                raise BigQueryToolError("invalid_query", "SQL comment is not closed")
            index += 2
            output.append(" ")
            continue
        output.append(char)
        index += 1
    if quote:
        raise BigQueryToolError(
            "invalid_query", "SQL string or identifier is not closed"
        )
    return "".join(output)


def _mask_strings(sql: str) -> str:
    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            if char == quote:
                if next_char == quote:
                    output.extend("  ")
                    index += 2
                    continue
                quote = None
                output.append("'")
            else:
                output.append(" ")
            index += 1
            continue
        if char == "'":
            quote = char
            output.append("'")
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    start = 0
    quote: str | None = None
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            if char == quote:
                if next_char == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == ";":
            value = sql[start:index].strip()
            if value:
                statements.append(value)
            start = index + 1
        index += 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


class BigQueryTools:
    """Read effective warehouse state through a bounded SQL interface."""

    def __init__(
        self,
        config: BigQueryConfig | None = None,
        *,
        acquire_table=None,
        row_count_adjustments=None,
    ):
        self.config = config or BigQueryConfig.from_env()
        self.config.validate()
        self._lock = threading.RLock()
        self._closed = False
        self._manifest = self._load_json(self.config.baseline_dir / "manifest.json")
        project = self._manifest.get("project", {})
        if project.get("id") != self.config.project_id:
            raise BigQueryToolError(
                "invalid_configuration",
                f"Configured project {self.config.project_id} does not match the baseline",
            )
        self._datasets = {
            item["id"]: item for item in self._manifest.get("datasets", [])
        }
        self._tables = {
            (item["dataset_id"], item["table_id"]): dict(item)
            for item in self._manifest.get("tables", [])
        }
        unknown_allowed = set(self.config.allowed_datasets) - set(self._datasets)
        if unknown_allowed and self.config.allowed_datasets != DEFAULT_ALLOWED_DATASETS:
            raise BigQueryToolError(
                "invalid_configuration",
                "Allowed datasets are missing from the baseline: "
                + ", ".join(sorted(unknown_allowed)),
            )
        self._allowed_datasets = tuple(
            name for name in self.config.allowed_datasets if name in self._datasets
        )
        if not self._allowed_datasets:
            raise BigQueryToolError(
                "invalid_configuration",
                "No configured dataset is present in the BigQuery baseline",
            )
        self._schemas = self._load_schemas()
        self._apply_schema_changes()
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not overlay_path.exists():
            raise FileNotFoundError(f"Missing BigQuery overlay: {overlay_path}")
        self._overlay_path = overlay_path
        self._acquire_table = acquire_table
        self._row_count_adjustments = row_count_adjustments or {}
        self._loaded_tables = set()
        self._duck = duckdb.connect(
            config={
                "threads": self.config.duckdb_threads,
                "memory_limit": self.config.duckdb_memory_limit,
                "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            }
        )
        try:
            paths = [
                str((self.config.baseline_dir / f["path"]).resolve())
                for key, table in self._tables.items()
                if key[0] in self._allowed_datasets
                for f in table.get("files", [])
            ]
            self._duck.execute("SET allowed_paths=?", [paths])
            self._duck.execute("SET enable_external_access=false")
            self._duck.execute("SET lock_configuration=true")
            for dataset_id in self._allowed_datasets:
                self._duck.execute(f"CREATE SCHEMA {_quote_identifier(dataset_id)}")
            if acquire_table is None:
                for key in self._tables:
                    if key[0] in self._allowed_datasets:
                        self._ensure_table(key)
        except BaseException:
            self._duck.close()
            raise

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing BigQuery data file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"BigQuery data file must be an object: {path}")
        return value

    def _load_schemas(self) -> dict[tuple[str, str], dict[str, Any]]:
        schemas: dict[tuple[str, str], dict[str, Any]] = {}
        for key, table in self._tables.items():
            if key[0] not in self._allowed_datasets:
                continue
            schema = self._load_json(self.config.baseline_dir / table["schema_file"])
            for field in schema.get("fields", []):
                _validate_field(field)
            schemas[key] = schema
        return schemas

    def _apply_schema_changes(self) -> None:
        changes_path = self.config.overlay_dir / "schema_changes.json"
        if not changes_path.exists():
            return
        changes = self._load_json(changes_path)
        for change in changes.get("tables", []):
            key = (change.get("dataset_id"), change.get("table_id"))
            if key not in self._schemas:
                raise ValueError(f"Schema change refers to unknown table: {key}")
            fields = change.get("fields")
            if not isinstance(fields, list) or not fields:
                raise ValueError(f"Schema change for {key} needs fields")
            for field in fields:
                _validate_field(field)
            field_names = [field["name"] for field in fields]
            if len(field_names) != len(set(field_names)):
                raise ValueError(f"Schema change for {key} has duplicate fields")
            if any(
                name not in field_names for name in self._tables[key]["primary_key"]
            ):
                raise ValueError(f"Schema change for {key} removes a primary key field")
            updated = dict(self._schemas[key])
            updated["fields"] = fields
            if "description" in change:
                updated["description"] = change["description"]
            self._schemas[key] = updated

    def _create_baseline_views(self, keys) -> None:
        for key, table in self._tables.items():
            if key not in keys:
                continue
            dataset_id, table_id = key
            schema = self._schemas[key]
            baseline_fields = {
                field["name"]
                for field in self._load_json(
                    self.config.baseline_dir / table["schema_file"]
                ).get("fields", [])
            }
            selections = []
            for field in schema["fields"]:
                name = _quote_identifier(field["name"])
                kind = _duckdb_type(field)
                if field["name"] in baseline_fields:
                    selections.append(f"CAST({name} AS {kind}) AS {name}")
                else:
                    selections.append(f"NULL::{kind} AS {name}")
            view_name = self._internal_name("baseline", dataset_id, table_id)
            if table.get("files"):
                files = ",".join(
                    "'" + _sql_path(self.config.baseline_dir / f["path"]) + "'"
                    for f in table["files"]
                )
                query = (
                    f"SELECT {','.join(selections)} "
                    f"FROM read_parquet([{files}], "
                    "union_by_name=true, hive_partitioning=true)"
                )
            else:
                empty = ",".join(
                    f"NULL::{_duckdb_type(field)} AS {_quote_identifier(field['name'])}"
                    for field in schema["fields"]
                )
                query = f"SELECT {empty} WHERE FALSE"
            self._duck.execute(f"CREATE VIEW {_quote_identifier(view_name)} AS {query}")

    def _create_overlay_tables_and_effective_views(self, keys) -> None:
        for key, schema in self._schemas.items():
            if key not in keys:
                continue
            dataset_id, table_id = key
            if dataset_id not in self._allowed_datasets:
                continue
            overlay_name = self._internal_name("overlay", dataset_id, table_id)
            column_sql = ",".join(
                f'{_quote_identifier(field["name"])} {_duckdb_type(field)}'
                for field in schema["fields"]
            )
            self._duck.execute(
                f"CREATE TABLE {_quote_identifier(overlay_name)} "
                f"({column_sql}, _operation VARCHAR)"
            )
            key_fields = self._tables[key]["primary_key"]
            key_match = " AND ".join(
                f"o.{_quote_identifier(name)} IS NOT DISTINCT FROM "
                f"b.{_quote_identifier(name)}"
                for name in key_fields
            )
            field_names = ",".join(
                _quote_identifier(field["name"]) for field in schema["fields"]
            )
            baseline_name = self._internal_name("baseline", dataset_id, table_id)
            self._duck.execute(
                f"CREATE VIEW {_quote_identifier(dataset_id)}.{_quote_identifier(table_id)} AS "
                f"SELECT {field_names} FROM {_quote_identifier(overlay_name)} "
                "WHERE _operation = 'upsert' UNION ALL "
                f"SELECT {field_names} FROM {_quote_identifier(baseline_name)} b "
                f"WHERE NOT EXISTS (SELECT 1 FROM {_quote_identifier(overlay_name)} o "
                f"WHERE {key_match})"
            )

    def _ensure_table(self, key):
        self._require_table(*key)
        if key in self._loaded_tables:
            return
        if self._acquire_table is not None:
            self._acquire_table(*key)
        self._create_baseline_views({key})
        self._create_overlay_tables_and_effective_views({key})
        self._loaded_tables.add(key)
        self._reload_overlay()

    @staticmethod
    def _internal_name(prefix: str, dataset_id: str, table_id: str) -> str:
        return f"_{prefix}_{dataset_id}_{table_id}"

    def _read_overlay_rows(self) -> list[sqlite3.Row]:
        uri = f"{self._overlay_path.resolve().as_uri()}?mode=ro"
        with closing(
            sqlite3.connect(
                uri,
                uri=True,
                timeout=self.config.sqlite_timeout_seconds,
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            return connection.execute(
                """
                SELECT dataset_id, table_id, primary_key, row_json, operation
                FROM rows
                ORDER BY dataset_id, table_id, primary_key
                """
            ).fetchall()

    def _reload_overlay(self) -> None:
        grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in self._read_overlay_rows():
            key = (row["dataset_id"], row["table_id"])
            if key not in self._tables:
                raise ValueError(f"Overlay refers to unknown table: {key}")
            primary_key = json.loads(row["primary_key"])
            row_data = json.loads(row["row_json"])
            expected_keys = set(self._tables[key]["primary_key"])
            if not isinstance(primary_key, dict) or set(primary_key) != expected_keys:
                raise ValueError(f"Overlay has an invalid primary key for {key}")
            if not isinstance(row_data, dict):
                raise ValueError(f"Overlay row for {key} must be an object")
            if row["operation"] == "upsert":
                combined = {**row_data, **primary_key}
                missing = [
                    field["name"]
                    for field in self._schemas[key]["fields"]
                    if str(field.get("mode", "NULLABLE")).upper() == "REQUIRED"
                    and combined.get(field["name"]) is None
                ]
                if missing:
                    raise ValueError(f"Overlay row for {key} lacks {missing}")
            grouped[key].append(row)
        for key, schema in self._schemas.items():
            dataset_id, table_id = key
            if key not in self._loaded_tables:
                continue
            overlay_name = self._internal_name("overlay", dataset_id, table_id)
            self._duck.execute(f"DELETE FROM {_quote_identifier(overlay_name)}")
            values = []
            for overlay_row in grouped.get(key, []):
                row_data = json.loads(overlay_row["row_json"])
                key_data = json.loads(overlay_row["primary_key"])
                combined = {**row_data, **key_data}
                values.append(
                    tuple(
                        _normalize_value(combined.get(field["name"]), field)
                        for field in schema["fields"]
                    )
                    + (overlay_row["operation"],)
                )
            if values:
                placeholders = ",".join("?" for _ in range(len(schema["fields"]) + 1))
                self._duck.executemany(
                    f"INSERT INTO {_quote_identifier(overlay_name)} VALUES ({placeholders})",
                    values,
                )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._duck.close()
                self._closed = True

    def __enter__(self) -> "BigQueryTools":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def bigquery_list_datasets(
        self,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List datasets visible to the configured agent principal."""

        self._require_project(project_id)
        limit = self._page_size(page_size)
        scope = self._scope("datasets", self.config.project_id)
        offset = self._cursor_offset(start_cursor, scope)
        datasets = [
            {
                "kind": "bigquery#dataset",
                "id": f"{self.config.project_id}:{dataset_id}",
                "datasetReference": {
                    "projectId": self.config.project_id,
                    "datasetId": dataset_id,
                },
                "friendlyName": dataset_id.replace("_", " ").title(),
                "description": self._datasets[dataset_id].get("description", ""),
                "location": self._datasets[dataset_id].get(
                    "location", self.config.location
                ),
                "labels": self._datasets[dataset_id].get("labels", {}),
            }
            for dataset_id in sorted(self._allowed_datasets)
        ]
        page = datasets[offset : offset + limit]
        next_offset = offset + len(page)
        has_more = next_offset < len(datasets)
        return {
            "ok": True,
            "kind": "bigquery#datasetList",
            "datasets": page,
            "has_more": has_more,
            "next_cursor": (
                self._encode_cursor(next_offset, scope) if has_more else None
            ),
        }

    def bigquery_list_tables(
        self,
        dataset_id: str,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List tables in one visible dataset with stable pagination."""

        self._require_project(project_id)
        self._require_dataset(dataset_id)
        limit = self._page_size(page_size)
        scope = self._scope("tables", self.config.project_id, dataset_id)
        offset = self._cursor_offset(start_cursor, scope)
        with self._lock:
            self._reload_overlay()
            tables = [
                self._table_summary(key)
                for key in sorted(self._tables)
                if key[0] == dataset_id
            ]
        page = tables[offset : offset + limit]
        next_offset = offset + len(page)
        has_more = next_offset < len(tables)
        return {
            "ok": True,
            "kind": "bigquery#tableList",
            "tables": page,
            "has_more": has_more,
            "next_cursor": (
                self._encode_cursor(next_offset, scope) if has_more else None
            ),
        }

    def bigquery_get_table_schema(
        self,
        dataset_id: str,
        table_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the effective schema and metadata for one visible table."""

        self._require_project(project_id)
        key = self._require_table(dataset_id, table_id)
        with self._lock:
            self._reload_overlay()
            summary = self._table_summary(key)
        schema = self._schemas[key]
        return {
            "ok": True,
            "kind": "bigquery#table",
            **summary,
            "schema": {"fields": schema["fields"]},
            "primaryKey": self._tables[key]["primary_key"],
        }

    def bigquery_query(
        self,
        query: str,
        *,
        project_id: str | None = None,
        max_results: int | None = None,
        start_cursor: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one bounded, read-only SQL query over effective tables."""

        self._require_project(project_id)
        normalized = self._validate_query(query)
        limit = self._page_size(max_results)
        timeout = self._timeout(timeout_seconds)
        scope = self._scope("query", self.config.project_id, normalized)
        offset = self._cursor_offset(start_cursor, scope)
        wrapped = (
            f'SELECT * FROM ({normalized}) AS "_bigquery_result" '
            f"LIMIT {limit + 1} OFFSET {offset}"
        )
        timer_fired = threading.Event()

        def interrupt() -> None:
            timer_fired.set()
            self._duck.interrupt()

        with self._lock:
            for reference in self._query_tables(normalized):
                self._ensure_table(reference)
            self._reload_overlay()
            timer = threading.Timer(timeout, interrupt)
            timer.daemon = True
            timer.start()
            try:
                cursor = self._duck.execute(wrapped)
                raw_columns = [item[0] for item in cursor.description]
                raw_types = [str(item[1]) for item in cursor.description]
                values = cursor.fetchall()
            except duckdb.Error as error:
                if timer_fired.is_set() or "interrupt" in str(error).lower():
                    raise BigQueryToolError(
                        "query_timeout", f"Query exceeded {timeout:g} seconds"
                    ) from error
                raise BigQueryToolError(
                    "invalid_query", "Query could not be run against this warehouse"
                ) from error
            finally:
                timer.cancel()
                timer.join()

        columns = self._unique_columns(raw_columns)
        has_more = len(values) > limit
        page = values[:limit]
        next_offset = offset + len(page)
        fields = [
            {"name": name, "type": self._bigquery_result_type(kind), "mode": "NULLABLE"}
            for name, kind in zip(columns, raw_types)
        ]
        return {
            "ok": True,
            "kind": "bigquery#queryResponse",
            "jobComplete": True,
            "jobReference": {
                "projectId": self.config.project_id,
                "jobId": "job_" + hashlib.sha256(normalized.encode()).hexdigest()[:20],
                "location": self.config.location,
            },
            "schema": {"fields": fields},
            "rows": [
                {name: _json_value(value) for name, value in zip(columns, row)}
                for row in page
            ],
            "row_count": len(page),
            "has_more": has_more,
            "next_cursor": (
                self._encode_cursor(next_offset, scope) if has_more else None
            ),
        }

    def _table_summary(self, key: tuple[str, str]) -> dict[str, Any]:
        dataset_id, table_id = key
        table = self._tables[key]
        if self._acquire_table is None:
            row_count = self._duck.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(dataset_id)}.{_quote_identifier(table_id)}"
            ).fetchone()[0]
        else:
            row_count = int(
                table.get("row_count", 0)
            ) + self._row_count_adjustments.get(key, 0)
        return {
            "id": f"{self.config.project_id}:{dataset_id}.{table_id}",
            "tableReference": {
                "projectId": self.config.project_id,
                "datasetId": dataset_id,
                "tableId": table_id,
            },
            "friendlyName": table_id.replace("_", " ").title(),
            "description": self._schemas[key].get("description", ""),
            "location": table.get("location", self.config.location),
            "labels": table.get("labels", {}),
            "numRows": str(row_count),
            "numBytes": str(sum(int(item["bytes"]) for item in table.get("files", []))),
            "timePartitioning": (
                {"type": "DAY", "field": table["partition_field"]}
                if table.get("partition_field")
                else None
            ),
        }

    def _require_project(self, project_id: str | None) -> None:
        if project_id is not None and project_id != self.config.project_id:
            raise BigQueryToolError("not_found", f"Project {project_id} was not found")

    def _require_dataset(self, dataset_id: str) -> str:
        if not isinstance(dataset_id, str) or not IDENTIFIER.fullmatch(dataset_id):
            raise BigQueryToolError(
                "invalid_argument", "dataset_id must be a valid identifier"
            )
        if dataset_id not in self._datasets:
            raise BigQueryToolError("not_found", f"Dataset {dataset_id} was not found")
        if dataset_id not in self._allowed_datasets:
            raise BigQueryToolError(
                "access_denied",
                f"{self.config.agent_principal} cannot access dataset {dataset_id}",
            )
        return dataset_id

    def _require_table(self, dataset_id: str, table_id: str) -> tuple[str, str]:
        self._require_dataset(dataset_id)
        if not isinstance(table_id, str) or not IDENTIFIER.fullmatch(table_id):
            raise BigQueryToolError(
                "invalid_argument", "table_id must be a valid identifier"
            )
        key = (dataset_id, table_id)
        if key not in self._tables:
            raise BigQueryToolError(
                "not_found", f"Table {dataset_id}.{table_id} was not found"
            )
        return key

    def _page_size(self, value: int | None) -> int:
        if value is None:
            return self.config.default_page_size
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BigQueryToolError(
                "invalid_argument", "page size must be a positive integer"
            )
        return min(value, self.config.max_page_size)

    def _timeout(self, value: float | None) -> float:
        if value is None:
            return self.config.query_timeout_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise BigQueryToolError(
                "invalid_argument", "timeout_seconds must be positive"
            )
        return min(float(value), self.config.max_query_timeout_seconds)

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise BigQueryToolError("invalid_argument", "query must not be empty")
        if len(query) > self.config.max_query_characters:
            raise BigQueryToolError(
                "invalid_argument",
                f"query exceeds {self.config.max_query_characters} characters",
            )
        uncommented = _strip_comments(query)
        statements = _split_statements(uncommented)
        if len(statements) != 1:
            raise BigQueryToolError(
                "unsafe_query", "Exactly one read-only query is allowed"
            )
        statement = self._normalize_backticks(statements[0])
        masked = _mask_strings(statement)
        tokens = {
            item.upper() for item in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", masked)
        }
        first = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", masked)
        if first is None or first.group(1).upper() not in {"SELECT", "WITH"}:
            raise BigQueryToolError(
                "unsafe_query", "Only SELECT queries and read-only CTEs are allowed"
            )
        blocked = sorted(tokens & BLOCKED_KEYWORDS)
        if blocked:
            raise BigQueryToolError(
                "unsafe_query", f"SQL operation {blocked[0]} is not allowed"
            )
        lowered = masked.lower()
        for function in BLOCKED_FUNCTIONS:
            if re.search(rf"\b{re.escape(function)}\s*\(", lowered):
                raise BigQueryToolError(
                    "unsafe_query", f"SQL function {function} is not allowed"
                )
        if re.search(r"\b(?:read_[A-Za-z0-9_]*|[A-Za-z0-9_]*_scan)\s*\(", lowered):
            raise BigQueryToolError(
                "unsafe_query", "External data functions are not allowed"
            )
        if re.search(
            r"\b(?:duckdb_|information_schema|pg_catalog|sqlite_)\w*", lowered
        ):
            raise BigQueryToolError("unsafe_query", "System catalogs are not available")
        self._validate_table_references(masked)
        self._query_tables(statement)
        return statement

    def _query_tables(self, statement):
        # Serialize syntax without binding caller names or opening any files.
        # Unlike get_table_names(), this retains already-loaded view names.
        try:
            tree = json.loads(
                self._duck.execute(
                    "SELECT json_serialize_sql(?)", [statement]
                ).fetchone()[0]
            )
        except duckdb.Error:
            raise BigQueryToolError(
                "invalid_query", "Query could not be parsed"
            ) from None
        if tree.get("error") or len(tree.get("statements", [])) != 1:
            raise BigQueryToolError("invalid_query", "Query could not be parsed")
        result = set()

        def visit(node, ctes=frozenset()):
            if isinstance(node, list):
                for child in node:
                    visit(child, ctes)
            elif isinstance(node, dict):
                local = ctes | {
                    item["key"] for item in node.get("cte_map", {}).get("map", [])
                }
                if node.get("type") == "BASE_TABLE":
                    schema, table = node["schema_name"], node["table_name"]
                    if node.get("catalog_name") or (not schema and table not in local):
                        raise BigQueryToolError(
                            "unsafe_query", "Use a permitted dataset.table name"
                        )
                    if schema:
                        result.add(self._require_table(schema, table))
                if node.get("class") == "FUNCTION":
                    name = node.get("function_name", "").lower()
                    if (
                        name in BLOCKED_FUNCTIONS
                        or name.startswith(("read_", "duckdb_", "sqlite_"))
                        or name.endswith("_scan")
                    ):
                        raise BigQueryToolError(
                            "unsafe_query",
                            "External and system functions are not allowed",
                        )
                if node.get("type") == "TABLE_FUNCTION":
                    function = node.get("function", {})
                    if function.get("function_name", "").lower() not in {
                        "range",
                        "generate_series",
                        "unnest",
                    }:
                        raise BigQueryToolError(
                            "unsafe_query",
                            "External and system table functions are not allowed",
                        )
                for child in node.values():
                    visit(child, local)

        visit(tree)
        return result

    def _normalize_backticks(self, sql: str) -> str:
        def replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            parts = raw.split(".")
            if len(parts) == 3:
                if parts[0] != self.config.project_id:
                    raise BigQueryToolError(
                        "not_found", f"Project {parts[0]} was not found"
                    )
                parts = parts[1:]
            if len(parts) not in {1, 2} or any(
                not IDENTIFIER.fullmatch(part) for part in parts
            ):
                raise BigQueryToolError(
                    "invalid_query", "Backticks may contain only a table identifier"
                )
            return ".".join(_quote_identifier(part) for part in parts)

        return re.sub(r"`([^`]+)`", replace, sql)

    def _validate_table_references(self, masked: str) -> None:
        if re.search(r"(?i)\b(?:FROM|JOIN)\s*'", masked):
            raise BigQueryToolError(
                "unsafe_query", "File paths cannot be used as query tables"
            )
        ctes = {
            self._unquote_identifier(match.group(1))
            for match in re.finditer(
                r'(?i)(?:\bWITH\b|,)\s*((?:"[A-Za-z_][A-Za-z0-9_]*")|(?:[A-Za-z_][A-Za-z0-9_]*))\s+AS\s*\(',
                masked,
            )
        }
        reference_pattern = re.compile(
            r'(?i)\b(?:FROM|JOIN)\s+((?:"[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_]*))?)'
        )
        for match in reference_pattern.finditer(masked):
            parts = [
                self._unquote_identifier(item.strip())
                for item in re.split(r"\s*\.\s*", match.group(1))
            ]
            if len(parts) == 1:
                if parts[0] in ctes:
                    continue
                raise BigQueryToolError(
                    "invalid_query",
                    f"Table {parts[0]} must use dataset.table qualification",
                )
            dataset_id, table_id = parts
            self._require_table(dataset_id, table_id)

    @staticmethod
    def _unquote_identifier(value: str) -> str:
        return value[1:-1] if value.startswith('"') and value.endswith('"') else value

    @staticmethod
    def _unique_columns(columns: Sequence[str]) -> list[str]:
        counts: dict[str, int] = {}
        result: list[str] = []
        for column in columns:
            count = counts.get(column, 0) + 1
            counts[column] = count
            result.append(column if count == 1 else f"{column}_{count}")
        return result

    @staticmethod
    def _bigquery_result_type(value: str) -> str:
        upper = value.upper()
        for prefix, result in DUCKDB_TO_BIGQUERY.items():
            if upper.startswith(prefix):
                return result
        return "STRING"

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
            raise BigQueryToolError(
                "invalid_cursor", "BigQuery cursor is invalid for this request"
            ) from error


_DEFAULT_TOOLS: BigQueryTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> BigQueryTools:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_TOOLS is None:
                _DEFAULT_TOOLS = BigQueryTools()
    return _DEFAULT_TOOLS
