"""Read-only Snowflake task world using native wrappers and leased source files."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath

from benchmark.prompts import load_text
from benchmark.tools.snowflake.snowflake import (
    IDENTIFIER,
    SnowflakeConfig,
    SnowflakeToolError,
    SnowflakeTools,
    build_snowflake_overlay_from_seed,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/snowflake/seed.json"
TEXT = dict(type="string", minLength=1, maxLength=4096)
PAGE = dict(
    page_size=dict(type="integer", minimum=1, maximum=100),
    start_cursor=dict(type=["string", "null"]),
)


def obj(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


def record_shape():
    return obj(
        dict(database_name=TEXT, schema_name=TEXT, object_name=TEXT, primary_key=TEXT)
    )


SCHEMAS = {
    "snowflake_list_datasets": ({}, PAGE),
    "snowflake_list_tables": ({"database_name": TEXT}, dict(schema_name=TEXT, **PAGE)),
    "snowflake_get_table_schema": (
        dict(database_name=TEXT, schema_name=TEXT, table_name=TEXT),
        {},
    ),
    "snowflake_query": (
        {"sql": dict(type="string", minLength=1, maxLength=20000)},
        dict(
            row_limit=dict(type="integer", minimum=1, maximum=500),
            timeout_seconds=dict(type="number", exclusiveMinimum=0, maximum=60),
        ),
    ),
}


def tool_definitions(scope, cap):
    descriptions = json.loads(load_text("tools/snowflake/descriptions.json"))
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


class SnowflakeEnvironmentError(ValueError):
    def __init__(self, reason, record=None):
        self.code, self.source = "invalid_snowflake_environment", "snowflake"
        self.record, self.validation_reason = record, reason
        super().__init__(reason)


def check(condition, reason, record=None):
    if not condition:
        raise SnowflakeEnvironmentError(reason, record)


def native_record(record):
    if isinstance(record, dict) and isinstance(record.get("primary_key"), str):
        try:
            record = dict(record, primary_key=json.loads(record["primary_key"]))
        except ValueError:
            check(False, "primary_key must contain a JSON object")
    return record


def key(record):
    record = native_record(record)
    check(
        isinstance(record, dict)
        and set(record)
        == {"database_name", "schema_name", "object_name", "primary_key"},
        "Complete native Snowflake reference required",
        record,
    )
    names = tuple(record[k] for k in ("database_name", "schema_name", "object_name"))
    check(
        all(
            isinstance(n, str) and IDENTIFIER.fullmatch(n) and n == n.upper()
            for n in names
        ),
        "Use uppercase native Snowflake identifiers",
        record,
    )
    check(
        isinstance(record["primary_key"], dict) and record["primary_key"],
        "Native primary key required",
        record,
    )
    return names, json.dumps(record["primary_key"], sort_keys=True, allow_nan=False)


class SnowflakeEnvironment:
    def __init__(self, arguments, manifest, task, config, acquire):
        self.scope = set(config["tool_scope"]) & SCHEMAS.keys()
        self.arguments, self.task, self.acquire = arguments, task, acquire
        self.limit, self.snapshot = arguments["max_result_bytes"], timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.settings = config.get("service_settings", {}).get("snowflake")
        if self.scope:
            check(
                isinstance(self.settings, dict)
                and set(self.settings)
                == {
                    "account_identifier",
                    "role",
                    "warehouse",
                    "allowed_databases",
                    "allowed_schemas",
                },
                "Snowflake requires explicit account, role, warehouse, allowed_databases and allowed_schemas",
            )
            check(
                all(
                    isinstance(self.settings[k], str) and self.settings[k]
                    for k in ("account_identifier", "role", "warehouse")
                ),
                "Snowflake identity fields must be nonempty",
            )
            databases, schemas = (
                self.settings["allowed_databases"],
                self.settings["allowed_schemas"],
            )
            check(
                isinstance(databases, list)
                and databases
                and all(
                    isinstance(d, str) and IDENTIFIER.fullmatch(d) and d == d.upper()
                    for d in databases
                ),
                "Explicit uppercase database grants required",
            )
            check(
                isinstance(schemas, list)
                and schemas
                and all(
                    isinstance(s, str)
                    and len(s.split(".")) == 2
                    and s.split(".")[0] in databases
                    and all(
                        IDENTIFIER.fullmatch(p) and p == p.upper() for p in s.split(".")
                    )
                    for s in schemas
                ),
                "Explicit database.schema grants required",
            )
        self.timeout = min(60, config["limits"].get("tool_timeout_seconds", 60))
        self.work = Path(arguments["work_dir"]) / "snowflake"
        self.tools, self.root, self.paths = None, None, ()
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/snowflake/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8388608 for f in entries),
            "Snowflake accepts only seed.json up to 8 MiB",
        )
        self.delta = (
            load_json(Path(arguments["package"]) / FILE)
            if entries
            else dict(rows=[], schema_changes=[])
        )
        check(
            isinstance(self.delta, dict)
            and set(self.delta) == {"rows", "schema_changes"}
            and self.delta["schema_changes"] == [],
            "Use native rows. Schema changes are not supported for task additions/replacements",
        )
        check(
            isinstance(self.delta["rows"], list),
            "Native Snowflake rows must be an array",
        )
        payload = set()
        for item in self.delta["rows"]:
            check(
                isinstance(item, dict)
                and set(item)
                == {
                    "database_name",
                    "schema_name",
                    "object_name",
                    "primary_key",
                    "row",
                    "operation",
                    "updated_at",
                },
                "Complete native overlay entries required",
            )
            record = {
                k: item[k]
                for k in ("database_name", "schema_name", "object_name", "primary_key")
            }
            native = key(record)
            check(
                native not in payload
                and item["operation"] == "upsert"
                and isinstance(item["row"], dict),
                "Unique native upserts required",
                record,
            )
            check(
                timestamp(item["updated_at"], FILE, "updated_at") <= self.snapshot,
                "Overlay update is after snapshot",
                record,
            )
            payload.add(native)
        inventory = set()
        for change in task["changes"]:
            if change["service"] == "snowflake":
                native = key(change["source"]["record"])
                check(
                    change["file"] == FILE
                    and change["operation"] in ("add", "replace")
                    and native not in inventory,
                    "Invalid Snowflake change inventory",
                )
                inventory.add(native)
        check(payload == inventory, "Snowflake delta and changes must match exactly")
        check(
            not (
                entries
                or inventory
                or any(e["service"] == "snowflake" for e in task["evidence"])
            )
            or self.scope,
            "Snowflake evidence and changes require configured reads",
        )
        self.validated = set()

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def _paths(self, paths):
        for path in paths:
            parts = PurePosixPath(path).parts
            check(
                not PurePosixPath(path).is_absolute()
                and ".." not in parts
                and "\\" not in path
                and "snowflake/" + path in self.paths,
                "Snowflake manifest path is outside pinned inventory",
            )
        return tuple("snowflake/" + p for p in paths)

    def _acquire_object(self, paths):
        if paths:
            self.acquire(self._paths(paths))

    def open(self):
        if self.tools is not None:
            return
        check(bool(self.scope), "Snowflake is not configured")
        self.paths = tuple(self.acquire.inventory())
        self.root = self.acquire(("snowflake/manifest.json",))
        manifest = load_json(self.root / "manifest.json")
        check(
            set(self.settings["allowed_databases"])
            <= {d["name"] for d in manifest["databases"]},
            "Database grant is absent from the pinned catalog",
        )
        check(
            set(self.settings["allowed_schemas"])
            <= {
                s["database_name"] + "." + s["schema_name"] for s in manifest["schemas"]
            },
            "Schema grant is absent from the pinned catalog",
        )
        objects = [
            o
            for o in manifest["objects"]
            if o["database_name"] in self.settings["allowed_databases"]
            and o["database_name"] + "." + o["schema_name"]
            in self.settings["allowed_schemas"]
        ]
        schemas = [o["schema_file"] for o in objects]
        self.acquire(self._paths(schemas), max_file_bytes=1048576)
        self.work.mkdir(parents=True, exist_ok=False)
        seed = self.work / "seed.json"
        seed.write_text(json.dumps(self.delta))
        build_snowflake_overlay_from_seed(seed, self.work)
        try:
            self.tools = SnowflakeTools(
                SnowflakeConfig(
                    baseline_dir=self.root,
                    overlay_dir=self.work,
                    account_identifier=self.settings["account_identifier"],
                    role=self.settings["role"],
                    warehouse=self.settings["warehouse"],
                    allowed_databases=tuple(self.settings["allowed_databases"]),
                    allowed_schemas=tuple(self.settings["allowed_schemas"]),
                    query_timeout_seconds=self.timeout,
                    max_query_timeout_seconds=self.timeout,
                    duckdb_memory_limit=self.arguments.get(
                        "duckdb_memory_limit", "256MB"
                    ),
                    duckdb_threads=self.arguments.get("duckdb_threads", 1),
                    duckdb_max_temp_directory_size=self.arguments.get(
                        "duckdb_max_temp_directory_size", "1GB"
                    ),
                ),
                acquire_object=self._acquire_object,
            )
            self._validate_rows()
        except BaseException:
            self.close()
            raise

    def _validate_rows(self):
        for item in self.delta["rows"]:
            names = tuple(
                item[k] for k in ("database_name", "schema_name", "object_name")
            )
            check(
                names in self.tools._schemas,
                "Overlay object is outside configured grants",
            )
            schema = self.tools._schemas[names]
            fields = {f["name"]: f for f in schema["fields"]}
            check(
                set(item["primary_key"]) == set(schema["primary_key"])
                and set(item["row"]) == set(fields),
                "Complete native fields and primary key required",
            )
            check(
                all(item["row"][k] == v for k, v in item["primary_key"].items()),
                "Row differs from native primary key",
            )
            for name, field in fields.items():
                value, kind = item["row"][name], field["type"]
                check(
                    value is not None or field["nullable"],
                    "Required native field is null: " + name,
                )
                if value is None:
                    continue
                if kind in ("INTEGER", "NUMBER", "FLOAT"):
                    check(
                        type(value) in (int, float, str)
                        and not isinstance(value, bool),
                        "Invalid native number: " + name,
                    )
                    try:
                        check(
                            math.isfinite(float(value)),
                            "Nonfinite native number: " + name,
                        )
                    except ValueError:
                        check(False, "Invalid native number: " + name)
                    if kind == "INTEGER":
                        check(type(value) is int, "Native integer required: " + name)
                elif kind == "BOOLEAN":
                    check(type(value) is bool, "Native boolean required: " + name)
                elif kind == "DATE":
                    check(
                        isinstance(value, str)
                        and date.fromisoformat(value) <= self.snapshot.date(),
                        "Native date is after snapshot: " + name,
                    )
                elif kind.startswith("TIMESTAMP"):
                    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    check(
                        when <= self.snapshot,
                        "Native timestamp is after snapshot: " + name,
                    )
                elif kind != "VARIANT":
                    check(isinstance(value, str), "Native text required: " + name)

    def _validate_objects(self, names):
        for name in names:
            if name in self.validated:
                continue
            for change in self.task["changes"]:
                if change["service"] != "snowflake":
                    continue
                record = native_record(change["source"]["record"])
                if key(record)[0] != name:
                    continue
                before = self.tools.exact_record(
                    *name, record["primary_key"], baseline=True
                )
                check(
                    (before is not None) == (change["operation"] == "replace"),
                    "Snowflake add/replace conflicts with pinned existence",
                    record,
                )
                check(
                    self.tools.exact_record(*name, record["primary_key"]) is not None,
                    "Effective Snowflake row is missing",
                    record,
                )
            self.validated.add(name)

    def lookup(self, record, view="effective"):
        record = native_record(record)
        names, _ = key(record)
        check(
            "snowflake_query" in self.scope,
            "Exact row evidence requires snowflake_query",
        )
        self.open()
        self._validate_objects([names])
        return self.tools.exact_record(
            *names, record["primary_key"], baseline=view == "baseline"
        )

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False,
                error="unknown_tool",
                message="Unknown or disallowed Snowflake tool",
            )
        required, optional = SCHEMAS[name]
        if (
            not isinstance(arguments, dict)
            or not set(required) <= arguments.keys()
            or arguments.keys() - (required.keys() | optional.keys())
        ):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Snowflake arguments",
            )
        try:
            self.open()
            if name == "snowflake_query":
                query = self.tools._validate_query(arguments["sql"])
                self._validate_objects(self.tools._query_objects(query))
            result = getattr(self.tools, name)(**arguments)
            if len(_encoded(result)) > self.limit:
                return dict(
                    ok=False,
                    error="result_limit",
                    message="Narrow the Snowflake query or use a smaller row/page limit",
                )
            return result
        except (TypeError, ValueError) as error:
            if isinstance(error, SnowflakeEnvironmentError):
                raise
            if isinstance(error, SnowflakeToolError):
                return error.as_dict()
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Snowflake arguments",
            )

    def close(self):
        if self.tools is not None:
            tools, self.tools = self.tools, None
            tools.close()
