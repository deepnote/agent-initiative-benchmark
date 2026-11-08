"""Local HubSpot benchmark wrappers backed by two SQLite databases."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import tomllib

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
OBJECT_TYPES = ("contacts", "companies", "deals")
OBJECT_ALIASES = {
    "contact": "contacts",
    "contacts": "contacts",
    "company": "companies",
    "companies": "companies",
    "deal": "deals",
    "deals": "deals",
}
DEFAULT_PROPERTIES = {
    "contacts": (
        "firstname",
        "lastname",
        "email",
        "phone",
        "jobtitle",
        "lifecyclestage",
        "hs_lead_status",
        "hubspot_owner_id",
    ),
    "companies": (
        "name",
        "domain",
        "industry",
        "numberofemployees",
        "annualrevenue",
        "lifecyclestage",
        "hubspot_owner_id",
        "country",
        "stripe_customer_id",
    ),
    "deals": (
        "dealname",
        "amount",
        "dealstage",
        "pipeline",
        "closedate",
        "dealtype",
        "hubspot_owner_id",
        "next_step",
    ),
}
DEFAULT_ASSOCIATIONS = {
    "contacts": ("companies", "deals"),
    "companies": ("contacts", "deals"),
    "deals": ("contacts", "companies"),
}
CORE_COLUMNS = (
    "id",
    "owner_id",
    "access_team_id",
    "archived",
    "archived_at",
    "created_at",
    "updated_at",
)
FILTER_OPERATORS = {
    "EQ",
    "NEQ",
    "LT",
    "LTE",
    "GT",
    "GTE",
    "CONTAINS_TOKEN",
    "IN",
    "NOT_IN",
    "HAS_PROPERTY",
    "NOT_HAS_PROPERTY",
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
DEFAULT_BASELINE_DIR = DATA_DIR / "hubspot"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "hubspot"


class HubSpotToolError(ValueError):
    """A predictable error that can be returned to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class HubSpotConfig:
    """Paths, permissions, limits, and query settings for HubSpot."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    agent_owner_id: str | None = "OWN_AGENT"
    visible_team_ids: tuple[str, ...] = ("TEAM_GTM", "TEAM_CS")
    allow_restricted_properties: bool = False
    max_page_size: int = 200
    max_query_characters: int = 1_000
    max_filters: int = 12
    max_properties: int = 50
    max_associations: int = 200
    max_history_values: int = 100
    max_record_id_characters: int = 128
    sqlite_timeout_seconds: float = 30.0
    cursor_secret: str = "pagemantle-hubspot-cursor-v1"

    @classmethod
    def from_env(cls) -> HubSpotConfig:
        teams = tuple(
            item.strip()
            for item in os.getenv(
                "BENCHMARK_HUBSPOT_VISIBLE_TEAM_IDS",
                ",".join(cls.visible_team_ids),
            ).split(",")
            if item.strip()
        )
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_HUBSPOT_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_HUBSPOT_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            agent_owner_id=os.getenv(
                "BENCHMARK_HUBSPOT_AGENT_OWNER_ID", cls.agent_owner_id
            ),
            visible_team_ids=teams,
            allow_restricted_properties=_env_bool(
                "HUBSPOT_ALLOW_RESTRICTED_PROPERTIES",
                cls.allow_restricted_properties,
            ),
            max_page_size=int(os.getenv("HUBSPOT_MAX_PAGE_SIZE", cls.max_page_size)),
            max_query_characters=int(
                os.getenv("HUBSPOT_MAX_QUERY_CHARACTERS", cls.max_query_characters)
            ),
            max_filters=int(os.getenv("HUBSPOT_MAX_FILTERS", cls.max_filters)),
            max_properties=int(os.getenv("HUBSPOT_MAX_PROPERTIES", cls.max_properties)),
            max_associations=int(
                os.getenv("HUBSPOT_MAX_ASSOCIATIONS", cls.max_associations)
            ),
            max_history_values=int(
                os.getenv("HUBSPOT_MAX_HISTORY_VALUES", cls.max_history_values)
            ),
            max_record_id_characters=int(
                os.getenv(
                    "HUBSPOT_MAX_RECORD_ID_CHARACTERS",
                    cls.max_record_id_characters,
                )
            ),
            sqlite_timeout_seconds=float(
                os.getenv("HUBSPOT_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
            cursor_secret=os.getenv("HUBSPOT_CURSOR_SECRET", cls.cursor_secret),
        )

    def validate(self) -> None:
        for name in (
            "max_page_size",
            "max_query_characters",
            "max_filters",
            "max_properties",
            "max_associations",
            "max_history_values",
            "max_record_id_characters",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")
        if not self.visible_team_ids:
            raise ValueError("visible_team_ids must not be empty")
        if not self.cursor_secret:
            raise ValueError("cursor_secret must not be empty")


OVERLAY_SQL = """
PRAGMA foreign_keys = OFF;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE contacts (
    id TEXT PRIMARY KEY,
    owner_id TEXT,
    access_team_id TEXT,
    archived INTEGER,
    archived_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX contacts_access_order_idx
    ON contacts(access_team_id, archived, updated_at DESC, id);

CREATE TABLE companies (
    id TEXT PRIMARY KEY,
    owner_id TEXT,
    access_team_id TEXT,
    archived INTEGER,
    archived_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX companies_access_order_idx
    ON companies(access_team_id, archived, updated_at DESC, id);

CREATE TABLE deals (
    id TEXT PRIMARY KEY,
    owner_id TEXT,
    access_team_id TEXT,
    archived INTEGER,
    archived_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX deals_access_order_idx
    ON deals(access_team_id, archived, updated_at DESC, id);

CREATE TABLE property_values (
    object_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    property_name TEXT NOT NULL,
    value TEXT,
    updated_at TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    PRIMARY KEY (object_type, record_id, property_name)
);
CREATE INDEX property_values_search_idx
    ON property_values(object_type, property_name, value, record_id);

CREATE TABLE associations (
    id TEXT PRIMARY KEY,
    from_object_type TEXT NOT NULL,
    from_record_id TEXT NOT NULL,
    to_object_type TEXT NOT NULL,
    to_record_id TEXT NOT NULL,
    association_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX associations_from_idx
    ON associations(from_object_type, from_record_id, to_object_type, to_record_id);
CREATE INDEX associations_to_idx
    ON associations(to_object_type, to_record_id, from_object_type, from_record_id);

CREATE TABLE history (
    id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    property_name TEXT NOT NULL,
    value TEXT,
    timestamp TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX history_record_idx
    ON history(object_type, record_id, property_name, timestamp DESC, id);

CREATE TABLE activities (
    id TEXT PRIMARY KEY,
    activity_type TEXT,
    owner_id TEXT,
    title TEXT,
    body TEXT,
    status TEXT,
    outcome TEXT,
    direction TEXT,
    occurred_at TEXT,
    due_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    source_type TEXT,
    source_id TEXT,
    metadata_json TEXT,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))
);
CREATE INDEX activities_time_idx ON activities(occurred_at DESC, id);

CREATE TABLE activity_associations (
    activity_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    association_type TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    PRIMARY KEY (activity_id, object_type, record_id, association_type)
);
CREATE INDEX activity_associations_record_idx
    ON activity_associations(object_type, record_id, activity_id);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _rows(
    connection: sqlite3.Connection,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, tuple(params)).fetchall()]


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
    values = [tuple(row.get(column) for column in columns) for row in rows]
    connection.executemany(
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})", values
    )


def _property_rows(
    object_type: str,
    records: Sequence[Mapping[str, Any]],
    *,
    operation: str | None = None,
) -> Iterator[dict[str, Any]]:
    for record in records:
        if record.get("operation") == "delete":
            continue
        updated_at = str(record["updated_at"])
        properties = record.get("properties", {})
        if not isinstance(properties, Mapping):
            raise TypeError(f"HubSpot {object_type} properties must be an object")
        for name, value in sorted(properties.items()):
            row = {
                "object_type": object_type,
                "record_id": record["id"],
                "property_name": name,
                "value": None if value is None else str(value),
                "updated_at": updated_at,
            }
            if operation is not None:
                row["operation"] = operation
            yield row


def _validate_seed(seed: Mapping[str, Any], *, overlay: bool) -> None:
    required = ("contacts", "companies", "deals", "associations", "history")
    if not overlay:
        required = (
            "teams",
            "owners",
            "property_definitions",
            "pipelines",
            "pipeline_stages",
        ) + required
    for key in required:
        if not isinstance(seed.get(key), list):
            raise TypeError(f"HubSpot seed needs a {key} list")
    for key in ("activities", "activity_associations"):
        if key in seed and not isinstance(seed[key], list):
            raise TypeError(f"HubSpot seed needs an optional {key} list")

    record_ids: dict[str, set[str]] = {}
    for object_type in OBJECT_TYPES:
        records = seed[object_type]
        ids = [str(row["id"]) for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"HubSpot {object_type} IDs must be unique")
        record_ids[object_type] = set(ids)

    if overlay:
        for object_type in OBJECT_TYPES:
            for record in seed[object_type]:
                if record.get("operation") not in {"upsert", "delete"}:
                    raise ValueError(
                        f"HubSpot overlay {object_type} needs an operation"
                    )
                if record["operation"] == "upsert":
                    for field in tuple(
                        column for column in CORE_COLUMNS if column != "archived_at"
                    ):
                        if field not in record or record[field] is None:
                            raise ValueError(
                                f"HubSpot overlay {object_type} upsert needs {field}"
                            )
        for table in ("associations", "history", "activities", "activity_associations"):
            for row in seed.get(table, []):
                if row.get("operation") not in {"upsert", "delete"}:
                    raise ValueError(f"HubSpot overlay {table} needs an operation")
        return

    team_ids = {row["id"] for row in seed["teams"]}
    owner_ids = {row["id"] for row in seed["owners"]}
    definitions = {
        (row["object_type"], row["name"]) for row in seed["property_definitions"]
    }
    pipeline_ids = {row["pipeline_id"] for row in seed["pipelines"]}
    stage_keys = {
        (row["pipeline_id"], row["stage_id"]) for row in seed["pipeline_stages"]
    }
    for owner in seed["owners"]:
        if owner["team_id"] not in team_ids:
            raise ValueError(f"HubSpot owner {owner['id']} has an unknown team")
    for object_type in OBJECT_TYPES:
        for record in seed[object_type]:
            if record["owner_id"] not in owner_ids:
                raise ValueError(
                    f"HubSpot {object_type} {record['id']} has an unknown owner"
                )
            if record["access_team_id"] not in team_ids:
                raise ValueError(
                    f"HubSpot {object_type} {record['id']} has an unknown team"
                )
            unknown = {
                (object_type, name)
                for name in record.get("properties", {})
                if (object_type, name) not in definitions
            }
            if unknown:
                raise ValueError(
                    f"HubSpot {record['id']} has undefined properties: {unknown}"
                )
            if object_type == "deals":
                pipeline_id = record.get("properties", {}).get("pipeline")
                stage_id = record.get("properties", {}).get("dealstage")
                if pipeline_id not in pipeline_ids:
                    raise ValueError(
                        f"HubSpot deal {record['id']} has an unknown pipeline"
                    )
                if (pipeline_id, stage_id) not in stage_keys:
                    raise ValueError(
                        f"HubSpot deal {record['id']} has an unknown stage"
                    )

    for association in seed["associations"]:
        source = association["from_object_type"]
        target = association["to_object_type"]
        if association["from_record_id"] not in record_ids.get(source, set()):
            raise ValueError(
                f"HubSpot association {association['id']} has an unknown source"
            )
        if association["to_record_id"] not in record_ids.get(target, set()):
            raise ValueError(
                f"HubSpot association {association['id']} has an unknown target"
            )
    for event in seed["history"]:
        if event["record_id"] not in record_ids.get(event["object_type"], set()):
            raise ValueError(f"HubSpot history {event['id']} has an unknown record")
        if (event["object_type"], event["property_name"]) not in definitions:
            raise ValueError(f"HubSpot history {event['id']} has an unknown property")
    activity_ids = {str(row["id"]) for row in seed.get("activities", [])}
    if len(activity_ids) != len(seed.get("activities", [])):
        raise ValueError("HubSpot activity IDs must be unique")
    for activity in seed.get("activities", []):
        if activity.get("owner_id") not in owner_ids:
            raise ValueError(f"HubSpot activity {activity['id']} has an unknown owner")
    for link in seed.get("activity_associations", []):
        if link["activity_id"] not in activity_ids:
            raise ValueError("HubSpot activity association has an unknown activity")
        if link["record_id"] not in record_ids.get(link["object_type"], set()):
            raise ValueError("HubSpot activity association has an unknown record")


def _new_database(path: Path, schema: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA page_size = 4096")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    connection.executescript(schema)
    return connection


def _finish_database(connection: sqlite3.Connection, path: Path) -> None:
    temporary = Path(connection.execute("PRAGMA database_list").fetchone()[2])
    connection.commit()
    connection.execute("VACUUM")
    connection.close()
    os.replace(temporary, path)


def _effective_view_sql(table: str, columns: Sequence[str], keys: Sequence[str]) -> str:
    selected = ",".join(columns)
    match = " AND ".join(f"o.{key}=b.{key}" for key in keys)
    return f"""
        CREATE TEMP VIEW effective_{table} AS
        SELECT {selected} FROM overlay.{table} WHERE operation = 'upsert'
        UNION ALL
        SELECT {','.join(f'b.{column}' for column in columns)}
        FROM baseline.{table} AS b
        WHERE NOT EXISTS (SELECT 1 FROM overlay.{table} AS o WHERE {match})
    """


class HubSpotTools:
    """Read-only HubSpot wrappers over the effective baseline and task state."""

    def __init__(self, config: HubSpotConfig | None = None):
        self.config = config or HubSpotConfig.from_env()
        self.config.validate()
        baseline_path = self.config.baseline_dir / "hubspot.sqlite"
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing HubSpot baseline: {baseline_path}")
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Missing HubSpot overlay: {overlay_path}")

        self._connection = sqlite3.connect(
            ":memory:",
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
            uri=True,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            "ATTACH DATABASE ? AS baseline",
            (f"{baseline_path.as_uri()}?mode=ro&immutable=1",),
        )
        self._connection.execute(
            "ATTACH DATABASE ? AS overlay", (f"{overlay_path.as_uri()}?mode=ro",)
        )
        for object_type in OBJECT_TYPES:
            self._connection.execute(
                _effective_view_sql(object_type, CORE_COLUMNS, ("id",))
            )
        self._connection.execute(
            _effective_view_sql(
                "property_values",
                ("object_type", "record_id", "property_name", "value", "updated_at"),
                ("object_type", "record_id", "property_name"),
            )
        )
        self._connection.execute(
            _effective_view_sql(
                "associations",
                (
                    "id",
                    "from_object_type",
                    "from_record_id",
                    "to_object_type",
                    "to_record_id",
                    "association_type",
                    "created_at",
                ),
                ("id",),
            )
        )
        self._connection.execute(
            _effective_view_sql(
                "history",
                (
                    "id",
                    "object_type",
                    "record_id",
                    "property_name",
                    "value",
                    "timestamp",
                    "source_type",
                    "source_id",
                ),
                ("id",),
            )
        )
        self._connection.execute(
            _effective_view_sql(
                "activities",
                (
                    "id",
                    "activity_type",
                    "owner_id",
                    "title",
                    "body",
                    "status",
                    "outcome",
                    "direction",
                    "occurred_at",
                    "due_at",
                    "created_at",
                    "updated_at",
                    "source_type",
                    "source_id",
                    "metadata_json",
                ),
                ("id",),
            )
        )
        self._connection.execute(
            _effective_view_sql(
                "activity_associations",
                ("activity_id", "object_type", "record_id", "association_type"),
                ("activity_id", "object_type", "record_id", "association_type"),
            )
        )
        self._connection.execute("PRAGMA query_only = ON")
        self._lock = threading.RLock()
        self._definitions = self._load_definitions()
        owner = self._connection.execute(
            "SELECT id FROM baseline.owners WHERE id = ? AND active = 1",
            (self.config.agent_owner_id,),
        ).fetchone()
        # A technical read principal has no CRM ownership. Access is still
        # controlled only by explicit team and property grants.
        if self.config.agent_owner_id is not None and owner is None:
            self.close()
            raise ValueError("agent_owner_id must identify an active HubSpot owner")

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def hubspot_search_contacts(
        self,
        query: str = "",
        *,
        filters: Sequence[Mapping[str, Any]] | None = None,
        properties: Sequence[str] | None = None,
        sort: str = "-updatedAt",
        page_size: int = 50,
        after: str | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        return self._search(
            "contacts", query, filters, properties, sort, page_size, after, archived
        )

    def hubspot_search_companies(
        self,
        query: str = "",
        *,
        filters: Sequence[Mapping[str, Any]] | None = None,
        properties: Sequence[str] | None = None,
        sort: str = "-updatedAt",
        page_size: int = 50,
        after: str | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        return self._search(
            "companies", query, filters, properties, sort, page_size, after, archived
        )

    def hubspot_search_deals(
        self,
        query: str = "",
        *,
        filters: Sequence[Mapping[str, Any]] | None = None,
        properties: Sequence[str] | None = None,
        sort: str = "-updatedAt",
        page_size: int = 50,
        after: str | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        return self._search(
            "deals", query, filters, properties, sort, page_size, after, archived
        )

    def hubspot_get_record(
        self,
        object_type: str,
        record_id: str,
        *,
        properties: Sequence[str] | None = None,
        associations: Sequence[str] | None = None,
        properties_with_history: Sequence[str] | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        normalized = self._object_type(object_type)
        record_id = self._record_id(record_id)
        record = self._record(normalized, record_id)
        if (
            record is None
            or not self._can_read(record)
            or (bool(record["archived"]) and not archived)
        ):
            raise HubSpotToolError(
                "not_found", f"No readable {normalized[:-1]} has ID {record_id}."
            )
        selected = self._selected_properties(normalized, properties)
        result = self._record_result(normalized, record, selected)
        requested_associations = (
            DEFAULT_ASSOCIATIONS[normalized]
            if associations is None
            else self._association_types(normalized, associations)
        )
        if requested_associations:
            result["associations"] = self._associations(
                normalized, record_id, requested_associations
            )
        history_properties = self._selected_properties(
            normalized,
            () if properties_with_history is None else properties_with_history,
            allow_empty=True,
        )
        if history_properties:
            result["propertiesWithHistory"] = self._history(
                normalized, record_id, history_properties
            )
        result["recentActivities"] = self._activities(normalized, record_id)
        return result

    def _search(
        self,
        object_type: str,
        query: str,
        filters: Sequence[Mapping[str, Any]] | None,
        properties: Sequence[str] | None,
        sort: str,
        page_size: int,
        after: str | None,
        archived: bool,
    ) -> dict[str, Any]:
        if not isinstance(query, str):
            raise HubSpotToolError("invalid_input", "query must be a string")
        query = query.strip()
        if len(query) > self.config.max_query_characters:
            raise HubSpotToolError(
                "invalid_input",
                f"query cannot exceed {self.config.max_query_characters} characters",
            )
        if not isinstance(archived, bool):
            raise HubSpotToolError("invalid_input", "archived must be a boolean")
        selected = self._selected_properties(object_type, properties)
        page_size = self._page_size(page_size)
        filter_sql, filter_params, canonical_filters = self._filters(
            object_type, filters
        )
        sort_sql, sort_params, canonical_sort = self._sort(object_type, sort)
        visible_placeholders = ",".join("?" for _ in self.config.visible_team_ids)
        where = [
            f"r.access_team_id IN ({visible_placeholders})",
            "r.archived = ?",
        ]
        params: list[Any] = [*self.config.visible_team_ids, int(archived)]
        if query:
            # Match once over the merged properties. A correlated probe through
            # the UNION view can rescan the entire property catalog per record.
            where.append(
                """
                r.id IN (
                    SELECT q.record_id
                    FROM effective_property_values AS q
                    JOIN baseline.property_definitions AS d
                      ON d.object_type = q.object_type AND d.name = q.property_name
                    WHERE q.object_type = ?
                      AND d.searchable = 1 AND d.access_level = 'standard'
                      AND LOWER(COALESCE(q.value, '')) LIKE ? ESCAPE '\\'
                )
                """
            )
            params.extend((object_type, f"%{self._like(query.lower())}%"))
        where.extend(filter_sql)
        params.extend(filter_params)
        scope = hashlib.sha256(
            _json(
                {
                    "object_type": object_type,
                    "query": query,
                    "filters": canonical_filters,
                    "properties": selected,
                    "sort": canonical_sort,
                    "archived": archived,
                }
            ).encode()
        ).hexdigest()
        offset = self._decode_cursor(after, scope)
        where_sql = " AND ".join(f"({part.strip()})" for part in where)
        with self._lock:
            total = int(
                self._connection.execute(
                    f"SELECT COUNT(*) FROM effective_{object_type} AS r WHERE {where_sql}",
                    tuple(params),
                ).fetchone()[0]
            )
            records = _rows(
                self._connection,
                f"""
                SELECT r.* FROM effective_{object_type} AS r
                WHERE {where_sql}
                ORDER BY {sort_sql}, r.id ASC
                LIMIT ? OFFSET ?
                """,
                tuple(params + sort_params + [page_size, offset]),
            )
        results = [
            self._record_result(object_type, record, selected) for record in records
        ]
        response: dict[str, Any] = {"total": total, "results": results}
        if offset + len(results) < total:
            response["paging"] = {
                "next": {"after": self._encode_cursor(offset + len(results), scope)}
            }
        return response

    def _filters(
        self,
        object_type: str,
        filters: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[list[str], list[Any], list[dict[str, Any]]]:
        if filters is None:
            return [], [], []
        if isinstance(filters, (str, bytes)) or not isinstance(filters, Sequence):
            raise HubSpotToolError("invalid_input", "filters must be a list")
        if len(filters) > self.config.max_filters:
            raise HubSpotToolError(
                "invalid_input",
                f"filters cannot exceed {self.config.max_filters} items",
            )
        sql: list[str] = []
        params: list[Any] = []
        canonical: list[dict[str, Any]] = []
        comparison = {
            "EQ": "=",
            "NEQ": "<>",
            "LT": "<",
            "LTE": "<=",
            "GT": ">",
            "GTE": ">=",
        }
        for raw in filters:
            if not isinstance(raw, Mapping):
                raise HubSpotToolError("invalid_input", "each filter must be an object")
            name = raw.get("property_name")
            operator = str(raw.get("operator", "EQ")).upper()
            if not isinstance(name, str):
                raise HubSpotToolError(
                    "invalid_input", "filter property_name is required"
                )
            definition = self._property_definition(object_type, name)
            if operator not in FILTER_OPERATORS:
                raise HubSpotToolError(
                    "invalid_operator", f"Unsupported operator: {operator}"
                )
            value_expression = "p.value"
            value_transform = lambda value: str(value)
            if definition["type"] == "number":
                value_expression = "CAST(NULLIF(p.value, '') AS REAL)"
                value_transform = float
            exists_prefix = (
                "EXISTS (SELECT 1 FROM effective_property_values AS p "
                "WHERE p.object_type = ? AND p.record_id = r.id AND p.property_name = ?"
            )
            local_params: list[Any] = [object_type, name]
            canonical_item: dict[str, Any] = {
                "property_name": name,
                "operator": operator,
            }
            if operator in {"HAS_PROPERTY", "NOT_HAS_PROPERTY"}:
                expression = (
                    exists_prefix + " AND p.value IS NOT NULL AND p.value <> '')"
                )
                if operator == "NOT_HAS_PROPERTY":
                    expression = "NOT " + expression
            elif operator in {"IN", "NOT_IN"}:
                values = raw.get("values")
                if (
                    isinstance(values, (str, bytes))
                    or not isinstance(values, Sequence)
                    or not values
                ):
                    raise HubSpotToolError(
                        "invalid_input", f"{operator} needs a non-empty values list"
                    )
                normalized_values = [value_transform(value) for value in values]
                placeholders = ",".join("?" for _ in normalized_values)
                expression = (
                    exists_prefix
                    + f" AND {value_expression} {'NOT IN' if operator == 'NOT_IN' else 'IN'} ({placeholders}))"
                )
                local_params.extend(normalized_values)
                canonical_item["values"] = [str(value) for value in values]
            else:
                if "value" not in raw:
                    raise HubSpotToolError("invalid_input", f"{operator} needs value")
                value = raw["value"]
                if operator == "CONTAINS_TOKEN":
                    expression = (
                        exists_prefix
                        + " AND LOWER(COALESCE(p.value, '')) LIKE ? ESCAPE '\\')"
                    )
                    local_params.append(f"%{self._like(str(value).lower())}%")
                else:
                    expression = (
                        exists_prefix
                        + f" AND {value_expression} {comparison[operator]} ?)"
                    )
                    local_params.append(value_transform(value))
                canonical_item["value"] = str(value)
            sql.append(expression)
            params.extend(local_params)
            canonical.append(canonical_item)
        return sql, params, canonical

    def _sort(self, object_type: str, sort: str) -> tuple[str, list[Any], str]:
        if not isinstance(sort, str) or not sort.strip():
            raise HubSpotToolError("invalid_input", "sort must be a property name")
        value = sort.strip()
        direction = "DESC" if value.startswith("-") else "ASC"
        name = value[1:] if value.startswith(("-", "+")) else value
        core = {"createdAt": "r.created_at", "updatedAt": "r.updated_at", "id": "r.id"}
        if name in core:
            return f"{core[name]} {direction}", [], f"{direction}:{name}"
        definition = self._property_definition(object_type, name)
        expression = (
            "(SELECT p.value FROM effective_property_values AS p "
            "WHERE p.object_type = ? AND p.record_id = r.id AND p.property_name = ?)"
        )
        if definition["type"] == "number":
            expression = f"CAST({expression} AS REAL)"
        return f"{expression} {direction}", [object_type, name], f"{direction}:{name}"

    def _record_result(
        self,
        object_type: str,
        record: Mapping[str, Any],
        properties: Sequence[str],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {name: None for name in properties}
        if properties:
            placeholders = ",".join("?" for _ in properties)
            with self._lock:
                rows = _rows(
                    self._connection,
                    f"""
                    SELECT property_name, value
                    FROM effective_property_values
                    WHERE object_type = ? AND record_id = ?
                      AND property_name IN ({placeholders})
                    ORDER BY property_name
                    """,
                    (object_type, record["id"], *properties),
                )
            values.update({row["property_name"]: row["value"] for row in rows})
        result: dict[str, Any] = {
            "id": str(record["id"]),
            "properties": values,
            "createdAt": record["created_at"],
            "updatedAt": record["updated_at"],
            "archived": bool(record["archived"]),
            "url": f"hubspot://{object_type}/{record['id']}",
        }
        if record["archived_at"]:
            result["archivedAt"] = record["archived_at"]
        return result

    def _associations(
        self,
        object_type: str,
        record_id: str,
        targets: Sequence[str],
    ) -> dict[str, Any]:
        placeholders = ",".join("?" for _ in targets)
        with self._lock:
            rows = _rows(
                self._connection,
                f"""
                SELECT * FROM effective_associations
                WHERE (from_object_type = ? AND from_record_id = ? AND to_object_type IN ({placeholders}))
                   OR (to_object_type = ? AND to_record_id = ? AND from_object_type IN ({placeholders}))
                ORDER BY created_at, id
                LIMIT ?
                """,
                (
                    object_type,
                    record_id,
                    *targets,
                    object_type,
                    record_id,
                    *targets,
                    self.config.max_associations + 1,
                ),
            )
        if len(rows) > self.config.max_associations:
            raise HubSpotToolError(
                "result_too_large",
                f"Record has more than {self.config.max_associations} readable associations.",
            )
        result: dict[str, dict[str, list[dict[str, str]]]] = {}
        for row in rows:
            if (
                row["from_object_type"] == object_type
                and row["from_record_id"] == record_id
            ):
                target_type = row["to_object_type"]
                target_id = row["to_record_id"]
            else:
                target_type = row["from_object_type"]
                target_id = row["from_record_id"]
            target = self._record(target_type, target_id)
            if target is None or not self._can_read(target):
                continue
            bucket = result.setdefault(target_type, {"results": []})
            bucket["results"].append({"id": target_id, "type": row["association_type"]})
        for bucket in result.values():
            bucket["results"].sort(key=lambda item: (item["id"], item["type"]))
        return result

    def _history(
        self,
        object_type: str,
        record_id: str,
        properties: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        placeholders = ",".join("?" for _ in properties)
        with self._lock:
            rows = _rows(
                self._connection,
                f"""
                SELECT property_name, value, timestamp, source_type, source_id
                FROM effective_history
                WHERE object_type = ? AND record_id = ?
                  AND property_name IN ({placeholders})
                ORDER BY property_name, timestamp DESC, id DESC
                LIMIT ?
                """,
                (
                    object_type,
                    record_id,
                    *properties,
                    self.config.max_history_values + 1,
                ),
            )
        if len(rows) > self.config.max_history_values:
            raise HubSpotToolError(
                "result_too_large",
                f"History exceeds {self.config.max_history_values} values.",
            )
        result = {name: [] for name in properties}
        for row in rows:
            result[row["property_name"]].append(
                {
                    "value": row["value"],
                    "timestamp": row["timestamp"],
                    "sourceType": row["source_type"],
                    "sourceId": row["source_id"],
                }
            )
        return result

    def _activities(self, object_type: str, record_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = _rows(
                self._connection,
                """
                SELECT a.*, o.first_name, o.last_name, o.email AS owner_email,
                       aa.association_type
                FROM effective_activity_associations AS aa
                JOIN effective_activities AS a ON a.id = aa.activity_id
                LEFT JOIN baseline.owners AS o ON o.id = a.owner_id
                WHERE aa.object_type = ? AND aa.record_id = ?
                ORDER BY a.occurred_at DESC, a.id DESC
                LIMIT 100
                """,
                (object_type, record_id),
            )
        result = []
        for row in rows:
            try:
                metadata = json.loads(row.pop("metadata_json") or "{}")
            except json.JSONDecodeError:
                metadata = {}
            first_name = row.pop("first_name")
            last_name = row.pop("last_name")
            owner_email = row.pop("owner_email")
            row["metadata"] = metadata
            row["owner"] = (
                {
                    "id": row["owner_id"],
                    "name": f"{first_name or ''} {last_name or ''}".strip(),
                    "email": owner_email,
                }
                if row["owner_id"]
                else None
            )
            result.append(row)
        return result

    def _load_definitions(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (row["object_type"], row["name"]): row
            for row in _rows(
                self._connection,
                "SELECT * FROM baseline.property_definitions ORDER BY object_type, display_order, name",
            )
        }

    def _property_definition(self, object_type: str, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise HubSpotToolError(
                "invalid_property", "Property names must be non-empty strings."
            )
        definition = self._definitions.get((object_type, name))
        if definition is None:
            raise HubSpotToolError(
                "invalid_property", f"Unknown {object_type[:-1]} property: {name}"
            )
        if (
            definition["access_level"] == "restricted"
            and not self.config.allow_restricted_properties
        ):
            raise HubSpotToolError(
                "forbidden_property",
                f"Property is not available to this identity: {name}",
            )
        return definition

    def _selected_properties(
        self,
        object_type: str,
        properties: Sequence[str] | None,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        if properties is None:
            selected = tuple(
                name
                for name in DEFAULT_PROPERTIES[object_type]
                if self.config.allow_restricted_properties
                or self._definitions.get((object_type, name), {}).get("access_level")
                != "restricted"
            )
        else:
            if isinstance(properties, (str, bytes)) or not isinstance(
                properties, Sequence
            ):
                raise HubSpotToolError("invalid_input", "properties must be a list")
            selected = tuple(properties)
        if len(selected) > self.config.max_properties:
            raise HubSpotToolError(
                "invalid_input",
                f"properties cannot exceed {self.config.max_properties} items",
            )
        if not selected and not allow_empty:
            return ()
        if any(not isinstance(name, str) for name in selected):
            raise HubSpotToolError("invalid_input", "property names must be strings")
        deduplicated = tuple(dict.fromkeys(selected))
        for name in deduplicated:
            self._property_definition(object_type, name)
        return deduplicated

    def _association_types(
        self, object_type: str, associations: Sequence[str]
    ) -> tuple[str, ...]:
        if isinstance(associations, (str, bytes)) or not isinstance(
            associations, Sequence
        ):
            raise HubSpotToolError("invalid_input", "associations must be a list")
        if len(associations) > len(OBJECT_TYPES) - 1:
            raise HubSpotToolError("invalid_input", "too many association object types")
        normalized = tuple(
            dict.fromkeys(self._object_type(value) for value in associations)
        )
        if object_type in normalized:
            raise HubSpotToolError(
                "invalid_input", "self associations are not supported"
            )
        return normalized

    def _record(self, object_type: str, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT * FROM effective_{object_type} WHERE id = ?", (record_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def _can_read(self, record: Mapping[str, Any]) -> bool:
        return record["access_team_id"] in self.config.visible_team_ids

    @staticmethod
    def _object_type(value: str) -> str:
        if not isinstance(value, str):
            raise HubSpotToolError(
                "invalid_object_type", "object_type must be a string"
            )
        normalized = OBJECT_ALIASES.get(value.strip().lower())
        if normalized is None:
            raise HubSpotToolError(
                "invalid_object_type",
                "object_type must be contacts, companies, or deals",
            )
        return normalized

    def _record_id(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise HubSpotToolError(
                "invalid_input", "record_id must be a non-empty string"
            )
        normalized = value.strip()
        if len(normalized) > self.config.max_record_id_characters:
            raise HubSpotToolError("invalid_input", "record_id is too long")
        return normalized

    def _page_size(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise HubSpotToolError("invalid_input", "page_size must be an integer")
        if value < 1 or value > self.config.max_page_size:
            raise HubSpotToolError(
                "invalid_input",
                f"page_size must be between 1 and {self.config.max_page_size}",
            )
        return value

    @staticmethod
    def _like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _encode_cursor(self, offset: int, scope: str) -> str:
        payload = _json({"offset": offset, "scope": scope}).encode()
        signature = hmac.new(
            self.config.cursor_secret.encode(), payload, hashlib.sha256
        ).digest()[:16]
        return base64.urlsafe_b64encode(signature + payload).decode().rstrip("=")

    def _decode_cursor(self, cursor: str | None, scope: str) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or not cursor:
            raise HubSpotToolError("invalid_cursor", "after must be an opaque cursor")
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode())
            signature, payload = decoded[:16], decoded[16:]
            expected = hmac.new(
                self.config.cursor_secret.encode(), payload, hashlib.sha256
            ).digest()[:16]
            if not hmac.compare_digest(signature, expected):
                raise ValueError("bad signature")
            value = json.loads(payload)
            offset = value["offset"]
            if (
                value["scope"] != scope
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
            ):
                raise ValueError("bad scope")
            return offset
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise HubSpotToolError(
                "invalid_cursor", "after is invalid or belongs to another search"
            ) from error


_DEFAULT_TOOLS: HubSpotTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> HubSpotTools:
    global _DEFAULT_TOOLS
    with _DEFAULT_LOCK:
        if _DEFAULT_TOOLS is None:
            _DEFAULT_TOOLS = HubSpotTools()
        return _DEFAULT_TOOLS


def hubspot_search_contacts(
    query: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return _default_tools().hubspot_search_contacts(query, **kwargs)


def hubspot_search_companies(
    query: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return _default_tools().hubspot_search_companies(query, **kwargs)


def hubspot_search_deals(
    query: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return _default_tools().hubspot_search_deals(query, **kwargs)
