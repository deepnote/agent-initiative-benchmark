"""BigQuery task worlds using the existing SQL wrappers and native row overlay."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from benchmark.prompts import load_text
from benchmark.tools.bigquery.bigquery import (
    BigQueryConfig,
    BigQueryTools,
    BigQueryToolError,
    DEFAULT_ALLOWED_DATASETS,
    IDENTIFIER,
    _json_value,
    _quote_identifier,
    build_bigquery_overlay_from_seed,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/bigquery/seed.json"
# The sealed full-build manifest is 4,357,708 bytes. Keep a finite metadata
# allowance above that measured source size. Individual schemas remain small.
MANIFEST_MAX_BYTES = 8 * 1024 * 1024
SCHEMA_MAX_BYTES = 256 * 1024
TEXT = dict(type="string", minLength=1, maxLength=20000)
NULL_TEXT = dict(type=["string", "null"])
PAGE = dict(
    page_size=dict(type=["integer", "null"], minimum=1, maximum=100),
    start_cursor=NULL_TEXT,
)
PROJECT = dict(project_id=NULL_TEXT)
SCHEMAS = {
    "bigquery_list_datasets": ({}, {**PROJECT, **PAGE}),
    "bigquery_list_tables": ({"dataset_id": TEXT}, {**PROJECT, **PAGE}),
    "bigquery_get_table_schema": ({"dataset_id": TEXT, "table_id": TEXT}, PROJECT),
    "bigquery_query": (
        {"query": TEXT},
        {
            **PROJECT,
            "max_results": dict(type=["integer", "null"], minimum=1, maximum=100),
            "start_cursor": NULL_TEXT,
            "timeout_seconds": dict(
                type=["number", "null"], exclusiveMinimum=0, maximum=10
            ),
        },
    ),
}


class BigQueryEnvironmentError(ValueError):
    def __init__(self, reason, record=None):
        self.code, self.source, self.record, self.validation_reason = (
            "invalid_bigquery_environment",
            "bigquery",
            record,
            reason,
        )
        super().__init__(reason)


def check(condition, reason, record=None):
    if not condition:
        raise BigQueryEnvironmentError(reason, record)


def obj(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


def tool_definitions(scope, cap):
    descriptions = json.loads(load_text("tools/bigquery/descriptions.json"))
    return [
        dict(
            name=name,
            description=descriptions[name],
            parameters=dict(
                type="object",
                properties={**required, **optional},
                required=list(required),
                additionalProperties=False,
            ),
        )
        for name, (required, optional) in SCHEMAS.items()
        if name in scope
    ]


def record_key(record):
    check(
        isinstance(record, dict) and record.get("table") in ("rows", "schemas"),
        "Expected a native row or schema key",
        record,
    )
    columns = {"table", "dataset_id", "table_id"} | (
        {"primary_key"} if record["table"] == "rows" else set()
    )
    check(
        set(record) == columns
        and all(
            isinstance(record[k], str) and IDENTIFIER.fullmatch(record[k])
            for k in ("dataset_id", "table_id")
        ),
        "Invalid native BigQuery key",
        record,
    )
    if record["table"] == "rows":
        check(
            isinstance(record["primary_key"], dict)
            and bool(record["primary_key"])
            and all(
                isinstance(k, str)
                and IDENTIFIER.fullmatch(k)
                and type(v) in (str, int, float, bool)
                and (not isinstance(v, float) or math.isfinite(v))
                for k, v in record["primary_key"].items()
            ),
            "Invalid primary key",
            record,
        )
    return json.dumps(record, sort_keys=True, allow_nan=False)


class BigQueryEnvironment:
    def __init__(self, arguments, manifest, task, config, acquire):
        self.scope = set(config["tool_scope"]) & SCHEMAS.keys()
        self.limit, self.task, self.acquire = (
            arguments["max_result_bytes"],
            task,
            acquire,
        )
        self.snapshot = timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.settings = config.get("service_settings", {}).get("bigquery")
        if self.scope:
            check(
                isinstance(self.settings, dict)
                and set(self.settings)
                == {"project_id", "location", "identity", "read_dataset_ids"},
                "BigQuery requires project_id, location, identity and read_dataset_ids settings",
            )
            check(
                all(
                    isinstance(self.settings[k], str) and self.settings[k].strip()
                    for k in ("project_id", "location", "identity")
                ),
                "BigQuery requires an explicit project, location and identity",
            )
            grants = self.settings["read_dataset_ids"]
            check(
                isinstance(grants, list)
                and bool(grants)
                and all(isinstance(k, str) for k in grants)
                and len(grants) == len(set(grants))
                and set(grants) <= set(DEFAULT_ALLOWED_DATASETS),
                "Dataset grants exceed existing BigQuery access",
            )
            check(
                self.settings["identity"] == BigQueryConfig.agent_principal,
                "BigQuery identity differs from the existing reader",
            )
        self.work = Path(arguments["work_dir"]) / "bigquery"
        self.options = dict(
            duckdb_memory_limit=arguments.get("duckdb_memory_limit", "256MB"),
            duckdb_threads=arguments.get("duckdb_threads", 1),
            duckdb_max_temp_directory_size=arguments.get(
                "duckdb_max_temp_directory_size", "1GB"
            ),
            query_timeout_seconds=min(
                config["limits"].get("tool_timeout_seconds", 10), 10
            ),
            max_query_timeout_seconds=10,
        )
        self.tools, self.root, self.paths = None, None, ()
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/bigquery/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8_388_608 for f in entries),
            "BigQuery accepts only seed.json up to 8 MiB",
        )
        self.delta = load_json(Path(arguments["package"]) / FILE) if entries else {}
        check(
            isinstance(self.delta, dict)
            and not self.delta.keys() - {"rows", "schema_changes"},
            "Unknown BigQuery delta fields",
        )
        check(
            all(
                isinstance(self.delta.get(k, []), list)
                for k in ("rows", "schema_changes")
            ),
            "BigQuery delta collections must be arrays",
        )
        check(not entries or self.scope, "BigQuery delta requires configured reads")
        payload = set()
        for item in self.delta.get("rows", []):
            check(
                isinstance(item, dict)
                and set(item)
                == {
                    "dataset_id",
                    "table_id",
                    "primary_key",
                    "row",
                    "operation",
                    "updated_at",
                }
                and item["operation"] == "upsert"
                and isinstance(item["row"], dict),
                "Expected a complete native BigQuery upsert",
            )
            record = dict(
                table="rows",
                **{k: item[k] for k in ("dataset_id", "table_id", "primary_key")},
            )
            key = record_key(record)
            check(key not in payload, "Duplicate BigQuery delta key", record)
            payload.add(key)
            check(
                timestamp(item["updated_at"], FILE, "updated_at") <= self.snapshot,
                "BigQuery change is after snapshot",
            )
        for item in self.delta.get("schema_changes", []):
            check(
                isinstance(item, dict)
                and {"dataset_id", "table_id", "fields"} <= item.keys()
                and not item.keys()
                - {"dataset_id", "table_id", "fields", "description"},
                "Invalid native schema change",
            )
            key = record_key(
                dict(
                    table="schemas",
                    dataset_id=item["dataset_id"],
                    table_id=item["table_id"],
                )
            )
            check(key not in payload, "Duplicate BigQuery schema change")
            payload.add(key)
        inventory = set()
        self.adjustments = {}
        for change in task["changes"]:
            if change["service"] != "bigquery":
                continue
            record = change["source"]["record"]
            key = record_key(record)
            check(
                change["file"] == FILE
                and change["operation"] in ("add", "replace")
                and key not in inventory,
                "Invalid BigQuery change inventory",
                record,
            )
            check(
                record["table"] != "schemas" or change["operation"] == "replace",
                "Only existing schemas can be replaced",
            )
            inventory.add(key)
            table = (record["dataset_id"], record["table_id"])
            if record["table"] == "rows" and change["operation"] == "add":
                self.adjustments[table] = self.adjustments.get(table, 0) + 1
        check(
            payload == inventory,
            "BigQuery delta and change inventory must match exactly",
        )
        check(
            not (payload or any(e["service"] == "bigquery" for e in task["evidence"]))
            or self.scope,
            "BigQuery evidence requires configured reads",
        )
        self.validated = set()

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def open(self):
        if self.tools is not None:
            return
        check(bool(self.scope), "BigQuery reads are not configured")
        self.paths = tuple(self.acquire.inventory())
        self.root = self.acquire(
            ("bigquery/manifest.json",), max_file_bytes=MANIFEST_MAX_BYTES
        )
        self.manifest = load_json(self.root / "manifest.json")
        check(
            self.manifest["project"]["id"] == self.settings["project_id"]
            and self.manifest["project"].get("location", "US")
            == self.settings["location"],
            "BigQuery project settings differ from pinned source",
        )
        grants = set(self.settings["read_dataset_ids"])
        check(
            grants <= {d["id"] for d in self.manifest["datasets"]},
            "BigQuery grant names an unknown source dataset",
        )
        self.tables = {
            (t["dataset_id"], t["table_id"]): t for t in self.manifest["tables"]
        }
        schemas = []
        for key, table in self.tables.items():
            check(
                all(IDENTIFIER.fullmatch(k) for k in key), "Invalid source table name"
            )
            check(
                table["schema_file"] == "schemas/" + "/".join(key) + ".json",
                "Invalid pinned schema path",
            )
            for f in table["files"]:
                check(
                    f["path"].startswith("/".join(key) + "/")
                    and f["path"].endswith(".parquet")
                    and not any(p in ("", ".", "..") for p in f["path"].split("/"))
                    and "\\" not in f["path"],
                    "Invalid pinned Parquet path",
                )
            if key[0] in grants:
                schemas.append("bigquery/" + table["schema_file"])
        check(set(schemas) <= set(self.paths), "Pinned BigQuery schemas are missing")
        if schemas:
            self.acquire(tuple(schemas), max_file_bytes=SCHEMA_MAX_BYTES)
        self.work.mkdir(parents=True, exist_ok=True)
        seed = self.work / "seed.json"
        seed.write_text(json.dumps(self.delta, allow_nan=False), encoding="utf-8")
        build_bigquery_overlay_from_seed(seed, self.work)
        try:
            self.tools = BigQueryTools(
                BigQueryConfig(
                    baseline_dir=self.root,
                    overlay_dir=self.work,
                    project_id=self.settings["project_id"],
                    location=self.settings["location"],
                    agent_principal=self.settings["identity"],
                    allowed_datasets=tuple(self.settings["read_dataset_ids"]),
                    **self.options,
                ),
                acquire_table=self._family,
                row_count_adjustments=self.adjustments,
            )
            self._validate_delta()
        except BigQueryEnvironmentError:
            self.close()
            raise
        except (ValueError, TypeError, KeyError, OverflowError):
            self.close()
            raise BigQueryEnvironmentError(
                "Native BigQuery delta does not match the pinned schema"
            ) from None
        except BaseException:
            self.close()
            raise

    def _family(self, dataset, table):
        files = tuple(
            "bigquery/" + f["path"] for f in self.tables[dataset, table]["files"]
        )
        check(set(files) <= set(self.paths), "Pinned BigQuery table has missing files")
        if files:
            self.acquire(files)

    def _validate_delta(self):
        for item in self.delta.get("schema_changes", []):
            key = self.tools._require_table(item["dataset_id"], item["table_id"])
            old = load_json(self.root / self.tables[key]["schema_file"])["fields"]
            fields = item["fields"]
            check(
                isinstance(fields, list)
                and all(
                    isinstance(f, dict)
                    and set(f) == {"name", "type", "mode", "description"}
                    for f in fields
                ),
                "Schema changes require complete native fields",
            )
            by_name = {f["name"]: f for f in fields}
            check(
                all(
                    f["name"] in by_name
                    and by_name[f["name"]]["type"] == f["type"]
                    and by_name[f["name"]]["mode"] == f["mode"]
                    for f in old
                ),
                "Schema changes must preserve existing fields and types",
            )
            check(
                all(
                    f["mode"] == "NULLABLE"
                    for f in fields
                    if f["name"] not in {o["name"] for o in old}
                ),
                "New schema fields must be nullable",
            )
        for item in self.delta.get("rows", []):
            key = self.tools._require_table(item["dataset_id"], item["table_id"])
            schema = self.tools._schemas[key]
            check(
                set(item["primary_key"]) == set(self.tables[key]["primary_key"]),
                "Primary key differs from pinned schema",
            )
            check(
                set(item["row"]) == {f["name"] for f in schema["fields"]},
                "BigQuery changes need complete native rows",
            )
            check(
                all(item["row"][k] == v for k, v in item["primary_key"].items()),
                "Row and primary key disagree",
            )
            for field in schema["fields"]:
                value, kind = item["row"][field["name"]], field["type"]
                check(
                    value is not None or field["mode"] == "NULLABLE",
                    "Required BigQuery value is null",
                )
                if value is None:
                    continue
                if kind in ("INT64", "INTEGER"):
                    check(
                        type(value) is int and -(2**63) <= value < 2**63,
                        "BigQuery integer required",
                    )
                elif kind in ("FLOAT", "FLOAT64", "NUMERIC"):
                    check(
                        type(value) in (float, int) and math.isfinite(value),
                        "Finite BigQuery number required",
                    )
                elif kind in ("BOOL", "BOOLEAN"):
                    check(type(value) is bool, "BigQuery boolean required")
                else:
                    check(isinstance(value, str), "Native BigQuery text required")
                    if kind == "JSON":
                        json.loads(value)
                    elif kind == "DATE":
                        date.fromisoformat(value)
                    elif kind == "TIMESTAMP":
                        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        if field["name"] in (
                            "synced_at",
                            "ingested_at",
                            "occurred_at",
                            "created_at",
                            "updated_at",
                            "observed_at",
                            "published_at",
                            "started_at",
                            "completed_at",
                        ):
                            check(
                                when.replace(tzinfo=when.tzinfo or timezone.utc)
                                <= self.snapshot,
                                "BigQuery observation is after snapshot",
                            )

    def _validate_changes(self, key):
        if key in self.validated:
            return
        for change in self.task["changes"]:
            if change["service"] != "bigquery":
                continue
            record = change["source"]["record"]
            if (
                record["table"] == "rows"
                and (record["dataset_id"], record["table_id"]) == key
            ):
                before = self._row(record, "baseline")
                check(
                    (before is None) == (change["operation"] == "add"),
                    "BigQuery add/replace disagrees with baseline",
                    record,
                )
        self.validated.add(key)

    def _row(self, record, view):
        key = self.tools._require_table(record["dataset_id"], record["table_id"])
        check(
            set(record["primary_key"]) == set(self.tables[key]["primary_key"]),
            "Exact primary key differs from pinned schema",
        )
        self.tools._ensure_table(key)
        table = (
            _quote_identifier(self.tools._internal_name("baseline", *key))
            if view == "baseline"
            else ".".join(_quote_identifier(k) for k in key)
        )
        where = " AND ".join(
            _quote_identifier(k) + " IS NOT DISTINCT FROM ?"
            for k in record["primary_key"]
        )
        cursor = self.tools._duck.execute(
            "SELECT * FROM " + table + " WHERE " + where + " LIMIT 2",
            list(record["primary_key"].values()),
        )
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        check(len(rows) <= 1, "Ambiguous BigQuery native primary key", record)
        return {k: _json_value(v) for k, v in zip(columns, rows[0])} if rows else None

    def lookup(self, record, view="effective"):
        record_key(record)
        check(view in ("baseline", "effective"), "Invalid BigQuery record view")
        self.open()
        key = self.tools._require_table(record["dataset_id"], record["table_id"])
        if record["table"] == "schemas":
            return (
                load_json(self.root / self.tables[key]["schema_file"])
                if view == "baseline"
                else self.tools._schemas[key]
            )
        self._validate_changes(key)
        return self._row(record, view)

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False,
                error="unknown_tool",
                message="Unknown or disallowed BigQuery tool",
            )
        required, optional = SCHEMAS[name]
        if (
            not isinstance(arguments, dict)
            or not required.keys() <= arguments.keys()
            or arguments.keys() - (required.keys() | optional.keys())
        ):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid BigQuery arguments",
            )
        for k, v in arguments.items():
            schema = {**required, **optional}[k]
            kind = {
                str: "string",
                int: "integer",
                float: "number",
                type(None): "null",
            }.get(type(v))
            types = (
                schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            )
            if (
                (kind not in types and not (kind == "integer" and "number" in types))
                or (
                    isinstance(v, str)
                    and not schema.get("minLength", 0)
                    <= len(v)
                    <= schema.get("maxLength", 20000)
                )
                or (
                    type(v) in (int, float)
                    and (
                        not math.isfinite(v)
                        or not schema.get("minimum", 0)
                        <= v
                        <= schema.get("maximum", 1e100)
                        or v <= schema.get("exclusiveMinimum", -1e100)
                    )
                )
            ):
                return dict(
                    ok=False,
                    error="invalid_arguments",
                    message="Invalid BigQuery argument: " + k,
                )
        try:
            self.open()
            if name == "bigquery_query":
                normalized = self.tools._validate_query(arguments["query"])
                for key in self.tools._query_tables(normalized):
                    self._validate_changes(key)
            result = getattr(self.tools, name)(**arguments)
            if len(_encoded(result)) > self.limit:
                return dict(
                    ok=False,
                    error="result_limit",
                    message="BigQuery result exceeds max_result_bytes. Select fewer columns or rows.",
                )
            return result
        except BigQueryToolError as error:
            return error.as_dict()

    def close(self):
        if self.tools is not None:
            tools, self.tools = self.tools, None
            tools.close()
