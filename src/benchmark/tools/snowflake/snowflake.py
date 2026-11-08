"""Local Snowflake benchmark wrappers over Parquet and SQLite overlays."""

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
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from benchmark import duckdb_runtime as duckdb


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
ACCOUNT_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")


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
DEFAULT_BASELINE_DIR = DATA_DIR / "snowflake"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "snowflake"


class SnowflakeToolError(ValueError):
    """A predictable error safe to expose to a benchmark agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class SnowflakeConfig:
    """Paths, role, warehouse, access, and bounded query settings."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    account_identifier: str = "PM_ORG.PM_PROD"
    role: str = "AGENT_ANALYST"
    warehouse: str = "AGENT_WH"
    allowed_databases: tuple[str, ...] = (
        "PAGEMANTLE_ANALYTICS",
        "PAGEMANTLE_GOVERNANCE",
        "ANALYTICS",
        "OPERATIONS",
    )
    default_page_size: int = 50
    max_page_size: int = 100
    default_query_limit: int = 100
    max_query_limit: int = 500
    max_query_characters: int = 20_000
    query_timeout_seconds: float = 2.0
    max_query_timeout_seconds: float = 10.0
    sqlite_timeout_seconds: float = 30.0
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "256MB"
    duckdb_max_temp_directory_size: str = "1GB"
    allowed_schemas: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        allowed = os.getenv("SNOWFLAKE_ALLOWED_DATABASES")
        schemas = os.getenv("SNOWFLAKE_ALLOWED_SCHEMAS")
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_SNOWFLAKE_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_SNOWFLAKE_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            account_identifier=os.getenv(
                "SNOWFLAKE_ACCOUNT_IDENTIFIER", cls.account_identifier
            ),
            role=os.getenv("SNOWFLAKE_ROLE", cls.role),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", cls.warehouse),
            allowed_databases=(
                tuple(
                    item.strip().upper() for item in allowed.split(",") if item.strip()
                )
                if allowed is not None
                else cls.allowed_databases
            ),
            default_page_size=int(
                os.getenv("SNOWFLAKE_DEFAULT_PAGE_SIZE", cls.default_page_size)
            ),
            max_page_size=int(os.getenv("SNOWFLAKE_MAX_PAGE_SIZE", cls.max_page_size)),
            default_query_limit=int(
                os.getenv("SNOWFLAKE_DEFAULT_QUERY_LIMIT", cls.default_query_limit)
            ),
            max_query_limit=int(
                os.getenv("SNOWFLAKE_MAX_QUERY_LIMIT", cls.max_query_limit)
            ),
            max_query_characters=int(
                os.getenv("SNOWFLAKE_MAX_QUERY_CHARACTERS", cls.max_query_characters)
            ),
            query_timeout_seconds=float(
                os.getenv("SNOWFLAKE_QUERY_TIMEOUT_SECONDS", cls.query_timeout_seconds)
            ),
            max_query_timeout_seconds=float(
                os.getenv(
                    "SNOWFLAKE_MAX_QUERY_TIMEOUT_SECONDS",
                    cls.max_query_timeout_seconds,
                )
            ),
            sqlite_timeout_seconds=float(
                os.getenv(
                    "SNOWFLAKE_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds
                )
            ),
            duckdb_threads=int(
                os.getenv("SNOWFLAKE_DUCKDB_THREADS", cls.duckdb_threads)
            ),
            duckdb_memory_limit=os.getenv(
                "SNOWFLAKE_DUCKDB_MEMORY_LIMIT", cls.duckdb_memory_limit
            ),
            duckdb_max_temp_directory_size=os.getenv(
                "SNOWFLAKE_DUCKDB_MAX_TEMP_DIRECTORY_SIZE",
                cls.duckdb_max_temp_directory_size,
            ),
            allowed_schemas=(
                tuple(s.strip().upper() for s in schemas.split(",") if s.strip())
                if schemas is not None
                else ()
            ),
        )

    def validate(self) -> None:
        if not ACCOUNT_IDENTIFIER.fullmatch(self.account_identifier):
            raise ValueError("account_identifier is invalid")
        if not IDENTIFIER.fullmatch(self.role) or not IDENTIFIER.fullmatch(
            self.warehouse
        ):
            raise ValueError("role and warehouse must be valid identifiers")
        if not self.allowed_databases:
            raise ValueError("allowed_databases must not be empty")
        if any(not IDENTIFIER.fullmatch(item) for item in self.allowed_databases):
            raise ValueError("allowed_databases contains an invalid identifier")
        if any(
            len(s.split(".")) != 2
            or s.split(".")[0] not in self.allowed_databases
            or any(not IDENTIFIER.fullmatch(p) for p in s.split("."))
            for s in self.allowed_schemas
        ):
            raise ValueError(
                "allowed_schemas must contain permitted database.schema names"
            )
        if len(set(self.allowed_databases)) != len(self.allowed_databases):
            raise ValueError("allowed_databases must not contain duplicates")
        for name in (
            "default_page_size",
            "max_page_size",
            "default_query_limit",
            "max_query_limit",
            "max_query_characters",
            "duckdb_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.default_page_size > self.max_page_size:
            raise ValueError("default_page_size must not exceed max_page_size")
        if self.default_query_limit > self.max_query_limit:
            raise ValueError("default_query_limit must not exceed max_query_limit")
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
    database_name TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    object_name TEXT NOT NULL,
    primary_key TEXT NOT NULL,
    row_json TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (database_name, schema_name, object_name, primary_key)
);

CREATE INDEX IF NOT EXISTS rows_object_idx
    ON rows(database_name, schema_name, object_name, operation, primary_key);
"""

SNOWFLAKE_TO_DUCKDB = {
    "VARCHAR": "VARCHAR",
    "TEXT": "VARCHAR",
    "NUMBER": "DECIMAL(38,9)",
    "INTEGER": "BIGINT",
    "FLOAT": "DOUBLE",
    "BOOLEAN": "BOOLEAN",
    "TIMESTAMP_TZ": "TIMESTAMP",
    "TIMESTAMP_NTZ": "TIMESTAMP",
    "DATE": "DATE",
    "VARIANT": "VARCHAR",
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


def _quote(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid identifier: {value}")
    return f'"{value}"'


def _duckdb_type(field: Mapping[str, Any]) -> str:
    kind = str(field.get("type", "")).upper()
    try:
        return SNOWFLAKE_TO_DUCKDB[kind]
    except KeyError as error:
        raise ValueError(f"Unsupported Snowflake type: {kind}") from error


def _normalize_value(value: Any, field: Mapping[str, Any]) -> Any:
    if value is None:
        return None
    kind = str(field["type"]).upper()
    if kind == "VARIANT" and not isinstance(value, str):
        return _json(value)
    if kind in {"INTEGER"}:
        return int(value)
    if kind == "NUMBER":
        return Decimal(str(value))
    if kind == "FLOAT":
        return float(value)
    if kind == "BOOLEAN":
        return bool(value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime, datetime_time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _validate_field(field: Mapping[str, Any]) -> None:
    name = field.get("name")
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ValueError("Every Snowflake field needs a valid name")
    _duckdb_type(field)
    if not isinstance(field.get("nullable"), bool):
        raise ValueError(f"Snowflake field {name} needs a nullable boolean")
    if not isinstance(field.get("description", ""), str):
        raise ValueError(f"Snowflake field {name} has an invalid description")


def _object_key(database: str, schema: str, object_name: str) -> tuple[str, str, str]:
    return database.upper(), schema.upper(), object_name.upper()


def _validate_seed(seed: Mapping[str, Any]) -> None:
    account = seed.get("account")
    if not isinstance(account, dict) or not ACCOUNT_IDENTIFIER.fullmatch(
        str(account.get("identifier", ""))
    ):
        raise ValueError("Snowflake seed needs a valid account")
    databases = seed.get("databases")
    if not isinstance(databases, list) or not databases:
        raise ValueError("Snowflake seed needs databases")
    seen_objects: set[tuple[str, str, str]] = set()
    for database in databases:
        database_name = str(database.get("name", ""))
        if not IDENTIFIER.fullmatch(database_name):
            raise ValueError("Every Snowflake database needs a valid name")
        schemas = database.get("schemas")
        if not isinstance(schemas, list) or not schemas:
            raise ValueError(f"Snowflake database {database_name} needs schemas")
        for schema in schemas:
            schema_name = str(schema.get("name", ""))
            if not IDENTIFIER.fullmatch(schema_name):
                raise ValueError("Every Snowflake schema needs a valid name")
            objects = schema.get("objects")
            if not isinstance(objects, list) or not objects:
                raise ValueError(
                    f"Snowflake schema {database_name}.{schema_name} needs objects"
                )
            for item in objects:
                object_name = str(item.get("name", ""))
                key = _object_key(database_name, schema_name, object_name)
                if not IDENTIFIER.fullmatch(object_name) or key in seen_objects:
                    raise ValueError("Snowflake object name is invalid or duplicated")
                seen_objects.add(key)
                if item.get("object_type") not in {
                    "TABLE",
                    "VIEW",
                    "DYNAMIC_TABLE",
                    "MATERIALIZED_VIEW",
                    "SNAPSHOT",
                }:
                    raise ValueError(
                        f"Snowflake object {'.'.join(key)} has an invalid type"
                    )
                fields = item.get("fields")
                if not isinstance(fields, list) or not fields:
                    raise ValueError(f"Snowflake object {'.'.join(key)} needs fields")
                for field in fields:
                    _validate_field(field)
                names = [str(field["name"]) for field in fields]
                if len(names) != len(set(names)):
                    raise ValueError(
                        f"Snowflake object {'.'.join(key)} repeats a field"
                    )
                primary_key = item.get("primary_key")
                if not isinstance(primary_key, list) or not primary_key:
                    raise ValueError(
                        f"Snowflake object {'.'.join(key)} needs a primary key"
                    )
                if any(name not in names for name in primary_key):
                    raise ValueError(
                        f"Snowflake object {'.'.join(key)} has an invalid key"
                    )
                partition_field = item.get("partition_field")
                if partition_field is not None and partition_field not in names:
                    raise ValueError(
                        f"Snowflake object {'.'.join(key)} has an invalid partition field"
                    )
                row_keys: set[str] = set()
                for row in item.get("rows", []):
                    for field in fields:
                        if not field["nullable"] and row.get(field["name"]) is None:
                            raise ValueError(
                                f"Snowflake row lacks required field {field['name']}"
                            )
                    row_key = _json([row.get(name) for name in primary_key])
                    if row_key in row_keys:
                        raise ValueError(
                            f"Snowflake object {'.'.join(key)} repeats a row"
                        )
                    row_keys.add(row_key)


def build_snowflake_overlay_from_seed(
    seed_path: Path, overlay_dir: Path
) -> dict[str, int]:
    """Build the writable row overlay and declarative schema changes."""

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed, dict):
        raise ValueError("Snowflake overlay seed must be a JSON object")
    rows = seed.get("rows", [])
    changes = seed.get("schema_changes", [])
    if not isinstance(rows, list) or not isinstance(changes, list):
        raise ValueError("Snowflake overlay rows and schema_changes must be lists")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / "overlay.sqlite"
    temporary = overlay_dir / ".overlay.sqlite.tmp"
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(OVERLAY_SQL)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("format_version", "1"),
        )
        for item in rows:
            names = [
                item.get("database_name"),
                item.get("schema_name"),
                item.get("object_name"),
            ]
            if any(
                not isinstance(name, str) or not IDENTIFIER.fullmatch(name)
                for name in names
            ):
                raise ValueError(
                    "Snowflake overlay row has an invalid object reference"
                )
            if item.get("operation", "upsert") not in {"upsert", "delete"}:
                raise ValueError("Snowflake overlay operation must be upsert or delete")
            primary_key = item.get("primary_key")
            row = item.get("row", {})
            if not isinstance(primary_key, dict) or not primary_key:
                raise ValueError("Snowflake overlay row needs primary_key")
            if not isinstance(row, dict):
                raise ValueError("Snowflake overlay row must be an object")
            connection.execute(
                """
                INSERT INTO rows(
                    database_name, schema_name, object_name, primary_key,
                    row_json, operation, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(names[0]).upper(),
                    str(names[1]).upper(),
                    str(names[2]).upper(),
                    _json(primary_key),
                    _json(row),
                    item.get("operation", "upsert"),
                    item.get("updated_at", "2026-09-01T12:00:00Z"),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, overlay_path)
    (overlay_dir / "schema_changes.json").write_text(
        json.dumps(
            {"format_version": 1, "objects": changes},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"rows": len(rows), "schema_changes": len(changes)}


def _strip_comments(sql: str) -> str:
    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            output.append(char)
            if char == quote:
                if following == quote:
                    output.append(following)
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(char)
        elif char == "-" and following == "-":
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            output.append(" ")
            continue
        elif char == "/" and following == "*":
            index += 2
            while index + 1 < len(sql) and sql[index : index + 2] != "*/":
                index += 1
            if index + 1 >= len(sql):
                raise SnowflakeToolError("invalid_query", "SQL comment is not closed")
            index += 2
            output.append(" ")
            continue
        else:
            output.append(char)
        index += 1
    if quote:
        raise SnowflakeToolError(
            "invalid_query", "SQL string or identifier is not closed"
        )
    return "".join(output)


def _mask_strings(sql: str) -> str:
    output: list[str] = []
    index = 0
    quote = False
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if quote:
            if char == "'":
                if following == "'":
                    output.extend("  ")
                    index += 2
                    continue
                quote = False
                output.append("'")
            else:
                output.append(" ")
        elif char == "'":
            quote = True
            output.append("'")
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _cursor(scope: str, offset: int) -> str:
    payload = _json({"scope": scope, "offset": offset})
    signature = hashlib.sha256(f"snowflake:v1:{payload}".encode()).hexdigest()[:16]
    return (
        base64.urlsafe_b64encode(f"{payload}.{signature}".encode()).decode().rstrip("=")
    )


def _decode_cursor(value: str | None, scope: str) -> int:
    if value is None:
        return 0
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        payload, signature = raw.rsplit(".", 1)
        expected = hashlib.sha256(f"snowflake:v1:{payload}".encode()).hexdigest()[:16]
        decoded = json.loads(payload)
        if signature != expected or decoded.get("scope") != scope:
            raise ValueError
        offset = decoded["offset"]
        if not isinstance(offset, int) or offset < 0:
            raise ValueError
        return offset
    except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SnowflakeToolError(
            "invalid_cursor", "Cursor is invalid for this listing"
        ) from error


def _page_size(value: int | None, config: SnowflakeConfig) -> int:
    result = config.default_page_size if value is None else value
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise SnowflakeToolError(
            "invalid_argument", "page_size must be a positive integer"
        )
    if result > config.max_page_size:
        raise SnowflakeToolError(
            "invalid_argument", f"page_size cannot exceed {config.max_page_size}"
        )
    return result


class SnowflakeTools:
    """Read effective Snowflake state through a bounded, read-only API."""

    def __init__(self, config: SnowflakeConfig | None = None, *, acquire_object=None):
        self.config = config or SnowflakeConfig.from_env()
        self.config.validate()
        self._acquire_object = acquire_object
        self._registered = {}
        self._query_tree = None
        self._lock = threading.RLock()
        self._closed = False
        self._manifest = self._load_json(self.config.baseline_dir / "manifest.json")
        account = self._manifest.get("account", {})
        if account.get("identifier") != self.config.account_identifier:
            raise SnowflakeToolError(
                "invalid_configuration",
                "Configured account does not match the Snowflake baseline",
            )
        self._databases = {
            item["name"].upper(): dict(item)
            for item in self._manifest.get("databases", [])
        }
        self._schemas_meta = {
            (item["database_name"].upper(), item["schema_name"].upper()): dict(item)
            for item in self._manifest.get("schemas", [])
        }
        self._objects = {
            _object_key(
                item["database_name"], item["schema_name"], item["object_name"]
            ): dict(item)
            for item in self._manifest.get("objects", [])
        }
        self._allowed_databases = tuple(
            name for name in self.config.allowed_databases if name in self._databases
        )
        if not self._allowed_databases:
            raise SnowflakeToolError(
                "invalid_configuration",
                "No configured database is present in the Snowflake baseline",
            )
        self._all_object_keys = set(self._objects)
        self._objects = {
            key: item
            for key, item in self._objects.items()
            if key[0] in self._allowed_databases
            and (
                not self.config.allowed_schemas
                or ".".join(key[:2]) in self.config.allowed_schemas
            )
        }
        self._schemas_meta = {
            key: item
            for key, item in self._schemas_meta.items()
            if key[0] in self._allowed_databases
            and (
                not self.config.allowed_schemas
                or ".".join(key) in self.config.allowed_schemas
            )
        }
        self._schemas = self._load_schemas()
        self._baseline_fields = {
            key: {field["name"] for field in value["fields"]}
            for key, value in self._schemas.items()
        }
        self._apply_schema_changes()
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not overlay_path.exists():
            raise FileNotFoundError(f"Missing Snowflake overlay: {overlay_path}")
        self._overlay_rows = self._load_overlay_rows(overlay_path)
        self._connection = self._open_connection()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SnowflakeToolError(
                "invalid_fixture", f"Cannot read {path.name}"
            ) from error
        if not isinstance(value, dict):
            raise SnowflakeToolError(
                "invalid_fixture", f"{path.name} must contain an object"
            )
        return value

    def _load_schemas(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        result: dict[tuple[str, str, str], dict[str, Any]] = {}
        for key, item in self._objects.items():
            path = self.config.baseline_dir / item["schema_file"]
            schema = self._load_json(path)
            for field in schema.get("fields", []):
                _validate_field(field)
            result[key] = schema
        return result

    def _apply_schema_changes(self) -> None:
        path = self.config.overlay_dir / "schema_changes.json"
        changes = self._load_json(path).get("objects", [])
        if not isinstance(changes, list):
            raise SnowflakeToolError(
                "invalid_fixture", "schema_changes objects must be a list"
            )
        for change in changes:
            key = _object_key(
                str(change.get("database_name", "")),
                str(change.get("schema_name", "")),
                str(change.get("object_name", "")),
            )
            if key not in self._schemas:
                raise SnowflakeToolError(
                    "invalid_fixture", "Schema change targets an unknown object"
                )
            operation = change.get("operation", "add_fields")
            if operation != "add_fields":
                raise SnowflakeToolError(
                    "invalid_fixture", "Unsupported Snowflake schema change"
                )
            fields = change.get("fields", [])
            if not isinstance(fields, list):
                raise SnowflakeToolError(
                    "invalid_fixture", "Schema change fields must be a list"
                )
            existing = {field["name"] for field in self._schemas[key]["fields"]}
            for field in fields:
                _validate_field(field)
                if field["name"] in existing:
                    raise SnowflakeToolError(
                        "invalid_fixture", "Schema change repeats a field"
                    )
                self._schemas[key]["fields"].append(field)
                existing.add(field["name"])

    def _load_overlay_rows(
        self, path: Path
    ) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
        result: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(
                uri, uri=True, timeout=self.config.sqlite_timeout_seconds
            )
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute(
                    "SELECT database_name, schema_name, object_name, primary_key, "
                    "row_json, operation, updated_at FROM rows ORDER BY 1,2,3,4"
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise SnowflakeToolError(
                "invalid_fixture", "Cannot read Snowflake overlay"
            ) from error
        for row in rows:
            key = _object_key(
                row["database_name"], row["schema_name"], row["object_name"]
            )
            if key not in self._objects:
                if key in self._all_object_keys:
                    continue
                raise SnowflakeToolError(
                    "invalid_fixture", "Overlay targets an unknown object"
                )
            try:
                primary_key = json.loads(row["primary_key"])
                value = json.loads(row["row_json"])
            except json.JSONDecodeError as error:
                raise SnowflakeToolError(
                    "invalid_fixture", "Overlay row contains invalid JSON"
                ) from error
            result[key].append(
                {
                    "primary_key": primary_key,
                    "row": value,
                    "operation": row["operation"],
                    "updated_at": row["updated_at"],
                }
            )
        return result

    def _open_connection(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(
            database=":memory:",
            config={
                "memory_limit": self.config.duckdb_memory_limit,
                "threads": self.config.duckdb_threads,
                "max_temp_directory_size": self.config.duckdb_max_temp_directory_size,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            },
        )
        connection.execute(
            "SET allowed_directories=?", [[str(self.config.baseline_dir.resolve())]]
        )
        connection.execute("SET enable_external_access=false")
        for database in sorted(self._allowed_databases):
            connection.execute(f"ATTACH ':memory:' AS {_quote(database)}")
            for database_name, schema_name in sorted(self._schemas_meta):
                if database_name == database:
                    connection.execute(
                        f"CREATE SCHEMA IF NOT EXISTS {_quote(database)}.{_quote(schema_name)}"
                    )
        return connection

    def _register_object(
        self,
        connection: duckdb.DuckDBPyConnection,
        key: tuple[str, str, str],
        partition_values=None,
    ) -> None:
        item = self._objects[key]
        partition = self._schemas[key].get("partition_field")
        selected_files = item["files"]
        if partition_values is not None and partition:
            # A file without a proven hive date can contain matching rows.
            # Keep the complete object if even one native path lacks that date.
            partitions = [
                [
                    p.split("=", 1)[1]
                    for p in Path(f["path"]).parts
                    if p.startswith(partition + "=")
                ]
                for f in selected_files
            ]
            if all(
                len(values) == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", values[0])
                for values in partitions
            ):
                selected_files = [
                    f
                    for f, values in zip(selected_files, partitions)
                    if values[0] in partition_values
                ]
        signature = tuple(f["path"] for f in selected_files)
        if self._registered.get(key) == signature:
            return
        if self._acquire_object is not None:
            self._acquire_object(signature)
        schema = self._schemas[key]
        fields = schema["fields"]
        primary_key = schema["primary_key"]
        overlay_name = (
            "__OVERLAY_" + hashlib.sha256(".".join(key).encode()).hexdigest()[:12]
        )
        qualified_overlay = f"{_quote(key[0])}.{_quote(key[1])}.{_quote(overlay_name)}"
        columns = ",".join(
            f'{_quote(field["name"])} {_duckdb_type(field)}' for field in fields
        )
        connection.execute("DROP VIEW IF EXISTS " + ".".join(_quote(p) for p in key))
        connection.execute(
            "DROP VIEW IF EXISTS "
            + ".".join(_quote(p) for p in (*key[:2], "__BASELINE_" + key[2]))
        )
        connection.execute("DROP TABLE IF EXISTS " + qualified_overlay)
        self._registered.pop(key, None)
        connection.execute(
            f"CREATE TABLE {qualified_overlay} ({columns}, __operation VARCHAR)"
        )
        overlay_rows = self._overlay_rows.get(key, [])
        if overlay_rows:
            placeholders = ",".join("?" for _ in range(len(fields) + 1))
            connection.executemany(
                f"INSERT INTO {qualified_overlay} VALUES ({placeholders})",
                [
                    tuple(
                        _normalize_value(
                            {**change["row"], **change["primary_key"]}.get(
                                field["name"]
                            ),
                            field,
                        )
                        for field in fields
                    )
                    + (change["operation"],)
                    for change in overlay_rows
                ],
            )
        paths = [self.config.baseline_dir / value["path"] for value in selected_files]
        baseline_fields = self._baseline_fields[key]
        baseline_select = ",".join(
            (
                f"b.{_quote(field['name'])}"
                if field["name"] in baseline_fields
                else f"CAST(NULL AS {_duckdb_type(field)}) AS {_quote(field['name'])}"
            )
            for field in fields
        )
        projected = ",".join(_quote(field["name"]) for field in fields)
        joins = " AND ".join(
            f"o.{_quote(name)} IS NOT DISTINCT FROM b.{_quote(name)}"
            for name in primary_key
        )
        target = f"{_quote(key[0])}.{_quote(key[1])}.{_quote(key[2])}"
        if paths:
            path_sql = "[" + ",".join(f"'{_sql_path(path)}'" for path in paths) + "]"
            baseline = (
                f"SELECT {baseline_select} "
                f"FROM read_parquet({path_sql}, union_by_name=true, hive_partitioning=true) b"
            )
        else:
            baseline = (
                "SELECT "
                + ",".join(
                    f"CAST(NULL AS {_duckdb_type(field)}) AS {_quote(field['name'])}"
                    for field in fields
                )
                + " WHERE FALSE"
            )
        baseline_target = (
            f'{_quote(key[0])}.{_quote(key[1])}.{_quote("__BASELINE_" + key[2])}'
        )
        connection.execute(f"CREATE VIEW {baseline_target} AS {baseline}")
        connection.execute(
            f"""
            CREATE VIEW {target} AS
            WITH baseline AS (
                {baseline}
            )
            SELECT {projected} FROM baseline b
            WHERE NOT EXISTS (SELECT 1 FROM {qualified_overlay} o WHERE {joins})
            UNION ALL
            SELECT {projected} FROM {qualified_overlay} WHERE __operation = 'upsert'
            """
        )

        self._registered[key] = signature

    def __enter__(self) -> "SnowflakeTools":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise SnowflakeToolError("closed", "Snowflake client is closed")

    def snowflake_list_datasets(
        self, *, page_size: int | None = None, start_cursor: str | None = None
    ) -> dict[str, Any]:
        """List accessible databases using the benchmark's dataset naming."""

        self._ensure_open()
        limit = _page_size(page_size, self.config)
        scope = "databases"
        offset = _decode_cursor(start_cursor, scope)
        names = sorted(self._allowed_databases)
        page = names[offset : offset + limit]
        datasets = []
        for name in page:
            value = self._databases[name]
            schemas = sorted(
                schema for database, schema in self._schemas_meta if database == name
            )
            datasets.append(
                {
                    "datasetReference": {
                        "accountIdentifier": self.config.account_identifier,
                        "databaseName": name,
                    },
                    "databaseName": name,
                    "description": value.get("description", ""),
                    "ownerRole": value.get("owner_role"),
                    "schemas": schemas,
                    "labels": value.get("labels", {}),
                }
            )
        next_offset = offset + len(page)
        return {
            "kind": "snowflake#databaseList",
            "accountIdentifier": self.config.account_identifier,
            "role": self.config.role,
            "datasets": datasets,
            "next_cursor": (
                _cursor(scope, next_offset) if next_offset < len(names) else None
            ),
        }

    def snowflake_list_tables(
        self,
        database_name: str,
        *,
        schema_name: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List accessible tables, views, models, and snapshots."""

        self._ensure_open()
        database = self._database_argument(database_name)
        schema = self._optional_identifier(schema_name, "schema_name")
        limit = _page_size(page_size, self.config)
        scope = f"objects:{database}:{schema or '*'}"
        offset = _decode_cursor(start_cursor, scope)
        keys = [
            key
            for key in sorted(self._objects)
            if key[0] == database and (schema is None or key[1] == schema)
        ]
        page = keys[offset : offset + limit]
        tables = [self._object_summary(key) for key in page]
        next_offset = offset + len(page)
        return {
            "kind": "snowflake#tableList",
            "databaseName": database,
            "schemaName": schema,
            "tables": tables,
            "next_cursor": (
                _cursor(scope, next_offset) if next_offset < len(keys) else None
            ),
        }

    def snowflake_get_table_schema(
        self, database_name: str, schema_name: str, table_name: str
    ) -> dict[str, Any]:
        """Return one effective schema including overlay field additions."""

        self._ensure_open()
        key = (
            self._database_argument(database_name),
            self._identifier_argument(schema_name, "schema_name"),
            self._identifier_argument(table_name, "table_name"),
        )
        if key not in self._objects:
            raise SnowflakeToolError("not_found", "Snowflake object was not found")
        return {
            "kind": "snowflake#tableSchema",
            "tableReference": {
                "accountIdentifier": self.config.account_identifier,
                "databaseName": key[0],
                "schemaName": key[1],
                "tableName": key[2],
            },
            "objectType": self._objects[key]["object_type"],
            "schema": self._schemas[key],
        }

    def snowflake_query(
        self,
        sql: str,
        *,
        row_limit: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one read-only SELECT against accessible Snowflake objects."""

        self._ensure_open()
        query = self._validate_query(sql)
        limit = self.config.default_query_limit if row_limit is None else row_limit
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= self.config.max_query_limit
        ):
            raise SnowflakeToolError(
                "invalid_argument",
                f"row_limit must be between 1 and {self.config.max_query_limit}",
            )
        timeout = (
            self.config.query_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 0 < timeout <= self.config.max_query_timeout_seconds
        ):
            raise SnowflakeToolError(
                "invalid_argument",
                f"timeout_seconds must be at most {self.config.max_query_timeout_seconds}",
            )
        with self._lock:
            started = time.monotonic()
            timer = threading.Timer(float(timeout), self._connection.interrupt)
            timer.start()
            try:
                references = self._query_objects(query)
                for key in references:
                    self._register_object(
                        self._connection, key, self._partition_values(key)
                    )
                remaining = float(timeout) - (time.monotonic() - started)
                if remaining <= 0:
                    raise SnowflakeToolError(
                        "query_timeout", "Snowflake query timed out"
                    )
                return self._execute_query(query, limit, remaining)
            except duckdb.InterruptException as error:
                raise SnowflakeToolError(
                    "query_timeout", "Snowflake query timed out"
                ) from error
            except duckdb.Error as error:
                raise SnowflakeToolError(
                    "invalid_query",
                    "Query could not be executed against accessible Snowflake objects",
                ) from error
            finally:
                timer.cancel()
                timer.join()

    def _execute_query(self, query, limit, timeout, parameters=None):
        wrapped = f"SELECT * FROM ({query}) AS __snowflake_result LIMIT {limit + 1}"
        started = time.monotonic()
        timer = threading.Timer(float(timeout), self._connection.interrupt)
        with self._lock:
            timer.start()
            try:
                cursor = self._connection.execute(wrapped, parameters or [])
                columns = [item[0] for item in cursor.description]
                values = cursor.fetchall()
            except duckdb.InterruptException as error:
                raise SnowflakeToolError(
                    "query_timeout", "Snowflake query timed out"
                ) from error
            except duckdb.Error as error:
                raise SnowflakeToolError(
                    "invalid_query",
                    "Query could not be executed against accessible Snowflake objects",
                ) from error
            finally:
                timer.cancel()
                timer.join()
        truncated = len(values) > limit
        values = values[:limit]
        query_id = (
            "01b"
            + hashlib.sha256(
                f"{self.config.account_identifier}:{self.config.role}:{query}".encode()
            )
            .hexdigest()[:21]
            .upper()
        )
        return {
            "kind": "snowflake#queryResult",
            "queryId": query_id,
            "role": self.config.role,
            "warehouse": self.config.warehouse,
            "columns": [{"name": name} for name in columns],
            "rows": [
                {name: _json_value(value) for name, value in zip(columns, row)}
                for row in values
            ],
            "rowCount": len(values),
            "truncated": truncated,
            "rowLimit": limit,
            "durationMs": max(0, round((time.monotonic() - started) * 1000)),
        }

    def _validate_query(self, sql: str) -> str:
        if not isinstance(sql, str) or not sql.strip():
            raise SnowflakeToolError("invalid_argument", "sql must not be empty")
        if len(sql) > self.config.max_query_characters:
            raise SnowflakeToolError("invalid_argument", "sql is too long")
        stripped = _strip_comments(sql).strip().rstrip(";").strip()
        if ";" in _mask_strings(stripped):
            raise SnowflakeToolError(
                "unsafe_query", "Only one SQL statement is allowed"
            )
        masked = _mask_strings(stripped)
        upper = masked.upper()
        if not re.match(r"^(SELECT|WITH)\b", upper):
            raise SnowflakeToolError(
                "unsafe_query", "Only read-only SELECT queries are allowed"
            )
        tokens = set(re.findall(r"\b[A-Z_][A-Z0-9_$]*\b", upper))
        if tokens & BLOCKED_KEYWORDS:
            raise SnowflakeToolError("unsafe_query", "SQL contains a blocked operation")
        function_names = {
            value.lower()
            for value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", masked)
        }
        if function_names & BLOCKED_FUNCTIONS:
            raise SnowflakeToolError(
                "unsafe_query", "SQL contains a blocked table or file function"
            )
        if re.search(r"\b(?:FROM|JOIN)\s*'", masked, re.IGNORECASE):
            raise SnowflakeToolError("unsafe_query", "SQL cannot read a file path")
        if re.search(
            r"\b(?:INFORMATION_SCHEMA|PG_CATALOG|SQLITE_MASTER|DUCKDB_)\b", upper
        ):
            raise SnowflakeToolError("unsafe_query", "System metadata is not available")
        return stripped

    def _query_objects(self, query):
        # Parse before binding. Never execute a user table function or expose
        # private overlay/baseline views, even through quoted names or CTEs.
        try:
            statements = self._connection.extract_statements(query)
            if (
                len(statements) != 1
                or statements[0].type != duckdb.StatementType.SELECT
            ):
                raise SnowflakeToolError("unsafe_query", "Only one SELECT is allowed")
            tree = json.loads(
                self._connection.execute(
                    "SELECT json_serialize_sql(?)", [query]
                ).fetchone()[0]
            )
            if tree.get("error"):
                raise SnowflakeToolError("invalid_query", "SQL could not be parsed")
            pending = [tree]
            while pending:
                node = pending.pop()
                if isinstance(node, dict):
                    if node.get("type") == "TABLE_FUNCTION":
                        raise SnowflakeToolError(
                            "unsafe_query", "Table and file functions are unavailable"
                        )
                    function = str(node.get("function_name", "")).lower()
                    if function in BLOCKED_FUNCTIONS or function.startswith(
                        ("duckdb_", "sqlite_", "pg_", "json_execute")
                    ):
                        raise SnowflakeToolError(
                            "unsafe_query", "System and file functions are unavailable"
                        )
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)
            references = set()

            def collect(value, ctes=frozenset()):
                if isinstance(value, dict):
                    ctes = ctes | {
                        item["key"].lower()
                        for item in value.get("cte_map", {}).get("map", [])
                    }
                    if value.get("type") == "BASE_TABLE":
                        parts = [
                            value.get("catalog_name", ""),
                            value.get("schema_name", ""),
                            value["table_name"],
                        ]
                        if not any(parts[:2]) and parts[2].lower() in ctes:
                            return
                        references.add(".".join(parts))
                    for child in value.values():
                        collect(child, ctes)
                elif isinstance(value, list):
                    for child in value:
                        collect(child, ctes)

            collect(tree)
            self._query_tree = tree
        except duckdb.Error as error:
            raise SnowflakeToolError(
                "invalid_query", "SQL could not be parsed"
            ) from error
        result = []
        for reference in sorted(references):
            parts = reference.split(".")
            if len(parts) != 3 or any(not IDENTIFIER.fullmatch(p) for p in parts):
                raise SnowflakeToolError(
                    "access_denied", "Use a permitted database.schema.table reference"
                )
            key = tuple(p.upper() for p in parts)
            if key not in self._objects:
                raise SnowflakeToolError(
                    "access_denied", "Object is outside the configured Snowflake grants"
                )
            result.append(key)
        return result

    def _partition_values(self, key):
        # Prune only an outer, single-table SELECT with one base reference.
        # OR, joins, CTEs, subqueries and computed predicates keep every file.
        field = self._schemas[key].get("partition_field")
        if not field or not self._query_tree:
            return None
        node = self._query_tree["statements"][0]["node"]
        source = node.get("from_table", {})
        if (
            node.get("type") != "SELECT_NODE"
            or source.get("type") != "BASE_TABLE"
            or node.get("cte_map", {}).get("map")
        ):
            return None
        pending, bases = [node], 0
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                bases += value.get("type") == "BASE_TABLE"
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        if bases != 1:
            return None

        def column(value):
            names = value.get("column_names", [])
            return (
                value.get("class") == "COLUMN_REF"
                and names
                and names[-1].upper() == field.upper()
            )

        def literal(value):
            if (
                value.get("class") == "CAST"
                and value.get("cast_type", {}).get("id") == "DATE"
            ):
                value = value["child"]
            if value.get("class") != "CONSTANT" or value.get("value", {}).get(
                "is_null"
            ):
                return None
            raw = value["value"].get("value")
            if not isinstance(raw, str):
                return None
            try:
                return date.fromisoformat(raw).isoformat()
            except ValueError:
                return None

        predicates = [node.get("where_clause") or {}]
        result = None
        while predicates:
            predicate = predicates.pop()
            if predicate.get("type") == "CONJUNCTION_AND":
                predicates.extend(predicate["children"])
                continue
            values = None
            if predicate.get("type") == "COMPARE_EQUAL":
                for left, right in (
                    (predicate["left"], predicate["right"]),
                    (predicate["right"], predicate["left"]),
                ):
                    if column(left) and (value := literal(right)) is not None:
                        values = {value}
            elif predicate.get("type") == "COMPARE_IN" and column(
                predicate["children"][0]
            ):
                choices = [literal(v) for v in predicate["children"][1:]]
                if choices and all(v is not None for v in choices):
                    values = set(choices)
            if values is not None:
                result = values if result is None else result & values
        return result

    def exact_record(self, database, schema, table, primary_key, *, baseline=False):
        key = (
            self._identifier_argument(database, "database_name"),
            self._identifier_argument(schema, "schema_name"),
            self._identifier_argument(table, "table_name"),
        )
        if key not in self._objects:
            raise SnowflakeToolError(
                "access_denied", "Object is outside the configured Snowflake grants"
            )
        fields = self._schemas[key]["primary_key"]
        if (
            not isinstance(primary_key, dict)
            or set(primary_key) != set(fields)
            or any(primary_key[k] is None for k in fields)
        ):
            raise SnowflakeToolError(
                "invalid_argument", "Supply the complete native primary key"
            )
        target = ".".join(
            _quote(p)
            for p in (*key[:2], "__BASELINE_" + key[2] if baseline else key[2])
        )
        query = (
            "SELECT * FROM "
            + target
            + " WHERE "
            + " AND ".join(_quote(k) + " IS NOT DISTINCT FROM ?" for k in fields)
        )
        with self._lock:
            partition = self._schemas[key].get("partition_field")
            values = {str(primary_key[partition])} if partition in primary_key else None
            self._register_object(self._connection, key, values)
            result = self._execute_query(
                query,
                2,
                self.config.query_timeout_seconds,
                [primary_key[k] for k in fields],
            )
        if result["truncated"] or len(result["rows"]) > 1:
            raise SnowflakeToolError(
                "ambiguous_record", "Native primary key matched multiple rows"
            )
        return result["rows"][0] if result["rows"] else None

    def _database_argument(self, value: str) -> str:
        database = self._identifier_argument(value, "database_name")
        if database not in self._databases:
            raise SnowflakeToolError("not_found", "Snowflake database was not found")
        if database not in self._allowed_databases:
            raise SnowflakeToolError(
                "access_denied", "Role cannot access this database"
            )
        return database

    @staticmethod
    def _identifier_argument(value: str, name: str) -> str:
        if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
            raise SnowflakeToolError(
                "invalid_argument", f"{name} is not a valid identifier"
            )
        return value.upper()

    def _optional_identifier(self, value: str | None, name: str) -> str | None:
        return None if value is None else self._identifier_argument(value, name)

    def _object_summary(self, key: tuple[str, str, str]) -> dict[str, Any]:
        item = self._objects[key]
        return {
            "tableReference": {
                "accountIdentifier": self.config.account_identifier,
                "databaseName": key[0],
                "schemaName": key[1],
                "tableName": key[2],
            },
            "objectType": item["object_type"],
            "description": item.get("description", ""),
            "modelComment": item.get("model_comment", ""),
            "labels": item.get("labels", {}),
            "rowCount": int(item.get("row_count", 0)),
            "partitionField": item.get("partition_field"),
        }


def _call(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    with SnowflakeTools() as tools:
        return getattr(tools, method)(*args, **kwargs)
