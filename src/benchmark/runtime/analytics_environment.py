"""Read-only Analytics task world in the shared worker and file cache."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark.prompts import load_text
from benchmark.tools.analytics.analytics import (
    AnalyticsTools,
    AnalyticsConfig,
    AnalyticsToolError,
    build_analytics_overlay_from_seed,
)
from benchmark.tools.analytics.runtime import (
    V2_SCHEMAS,
    schemas,
    delta_schemas,
    keys,
    build_delta,
    public,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/analytics/seed.json"
TEXT = dict(type="string", minLength=1)
PAGE = dict(
    page_size=dict(type="integer", minimum=1, maximum=500),
    start_cursor=dict(type=["string", "null"]),
)
ARRAY = dict(type="array", items=TEXT, maxItems=50)
OBJECT = dict(type=["object", "null"], additionalProperties=True)
SCHEMAS = {
    "analytics_get_metadata": ({}, {"property_id": TEXT}),
    "analytics_read_rows": (
        {"property_id": TEXT, "table": dict(type="string", enum=list(schemas(2)))},
        {
            **PAGE,
            "filters": OBJECT,
            "start_date": dict(type=["string", "null"]),
            "end_date": dict(type=["string", "null"]),
        },
    ),
    "analytics_run_report": (
        {
            "property_id": TEXT,
            "metrics": ARRAY,
            "date_ranges": dict(
                type="array",
                minItems=1,
                maxItems=4,
                items=dict(
                    type="object",
                    properties={"startDate": TEXT, "endDate": TEXT},
                    required=["startDate", "endDate"],
                    additionalProperties=False,
                ),
            ),
        },
        {
            **PAGE,
            "dimensions": ARRAY,
            "dimension_filter": OBJECT,
            "metric_filter": OBJECT,
            "order_bys": dict(
                type="array",
                maxItems=10,
                items=dict(type="object", additionalProperties=True),
            ),
            "metric_aggregations": dict(
                type="array", items=dict(type="string", enum=["TOTAL"]), maxItems=1
            ),
            "keep_empty_rows": dict(type="boolean"),
            "return_property_quota": dict(type="boolean"),
        },
    ),
}


class AnalyticsEnvironmentError(ValueError):
    def __init__(self, reason, record=None):
        self.code, self.record, self.source, self.validation_reason = (
            "invalid_analytics_environment",
            record,
            "analytics",
            reason,
        )
        super().__init__(reason)


def check(ok, reason, record=None):
    if not ok:
        raise AnalyticsEnvironmentError(reason, record)


def tool_definitions(scope, cap):
    descriptions = json.loads(load_text("tools/analytics/descriptions.json"))
    return [
        dict(
            name=n,
            description=descriptions[n],
            parameters=dict(
                type="object",
                properties={**req, **opt},
                required=list(req),
                additionalProperties=False,
            ),
        )
        for n, (req, opt) in SCHEMAS.items()
        if n in scope
    ]


def record_key(record, version):
    table = record.get("table")
    check(
        table in keys(version) and set(record) == {"table", *keys(version)[table]},
        "Exact native Analytics key required",
        record,
    )
    for k in keys(version)[table]:
        check(
            (
                type(record[k]) is int and record[k] > 0
                if k == "touch_position"
                else isinstance(record[k], str) and bool(record[k])
            ),
            "Invalid native Analytics key",
            record,
        )
    return table, tuple(record[k] for k in keys(version)[table])


class AnalyticsEnvironment:
    def __init__(self, arguments, manifest, task, config, acquire):
        self.scope = set(config["tool_scope"]) & SCHEMAS.keys()
        self.arguments, self.task, self.config, self.acquire = (
            arguments,
            task,
            config,
            acquire,
        )
        self.limit = arguments["max_result_bytes"]
        self.snapshot = timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.settings = config.get("service_settings", {}).get("analytics")
        if self.scope:
            check(
                isinstance(self.settings, dict)
                and set(self.settings)
                == {"property_id", "identity", "read_property_ids"},
                "Analytics requires property_id, identity and read_property_ids settings",
            )
            check(
                all(
                    isinstance(self.settings[k], str) and self.settings[k].strip()
                    for k in ("property_id", "identity")
                )
                and isinstance(self.settings["read_property_ids"], list)
                and all(isinstance(p, str) for p in self.settings["read_property_ids"])
                and self.settings["property_id"] in self.settings["read_property_ids"],
                "Analytics identity has no property read grant",
            )
        self.work = Path(arguments["work_dir"]) / "analytics"
        self.tools, self.root, self.paths = None, None, ()
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/analytics/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8_388_608 for f in entries),
            "Analytics accepts only seed.json up to 8 MiB",
        )
        check(
            not any(c["service"] == "analytics" for c in task["changes"]) or self.scope,
            "Analytics changes require configured reads",
        )
        check(
            not any(e["service"] == "analytics" for e in task["evidence"])
            or self.scope,
            "Analytics evidence requires configured reads",
        )
        self.delta = load_json(Path(arguments["package"]) / FILE) if entries else None
        check(not entries or self.scope, "Analytics delta requires configured reads")
        if self.delta is not None:
            check(
                isinstance(self.delta, dict)
                and type(self.delta.get("schema_version")) is int
                and self.delta["schema_version"] in (1, 2),
                "Analytics delta requires schema_version 1 or 2",
            )
        self.version = None

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def _family(self, family):
        directory = (
            "samples/" + family.removeprefix("sample_")
            if family.startswith("sample_")
            else family
        )
        paths = tuple(
            p
            for p in self.paths
            if p == "analytics/" + family + ".parquet"
            or p.startswith("analytics/" + directory + "/")
            and p.endswith(".parquet")
        )
        expected = self.manifest.get("row_count_by_file", {})
        check(
            all(
                "analytics/" + p in self.paths
                for p in expected
                if p == family + ".parquet" or p.startswith(directory + "/")
            ),
            "Pinned Analytics family has missing files: " + family,
        )
        if not paths:
            raise AnalyticsToolError(
                "not_available", "Stored Analytics family is unavailable: " + family
            )
        self.acquire(paths)

    def _validate_row(self, table, row):
        schema = delta_schemas(self.version)[table]
        check(
            isinstance(row, dict) and set(row) == dict(schema).keys(),
            "Complete native Analytics rows required",
        )
        record_key(
            dict(table=table, **{k: row[k] for k in keys(self.version)[table]}),
            self.version,
        )
        for field, kind in schema:
            value = row[field]
            if value is None:
                check(
                    field
                    in {"campaign_id", "parent_page_id", "retired_at", "active_to"}
                    or field.endswith("_campaign_id")
                    or field.endswith("_campaign_name"),
                    "Unexpected null: " + field,
                )
            elif kind == "BOOLEAN":
                check(type(value) is bool, "Boolean required: " + field)
            elif kind in ("INTEGER", "BIGINT"):
                check(
                    type(value) is int and value >= 0,
                    "Nonnegative integer required: " + field,
                )
            elif kind == "DOUBLE":
                check(
                    type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "Finite nonnegative number required: " + field,
                )
            else:
                check(isinstance(value, str), "Text required: " + field)
            if kind == "DATE" and value is not None:
                try:
                    when = date.fromisoformat(value)
                except ValueError:
                    check(False, "Invalid date: " + field)
                check(
                    when <= self.local_snapshot.date(),
                    "Observation after snapshot: " + field,
                )
            if kind == "TIMESTAMP" and value is not None and field != "generated_at":
                try:
                    when = datetime.fromisoformat(value)
                except ValueError:
                    check(False, "Invalid timestamp: " + field)
                check(
                    when.tzinfo is None
                    and when <= self.local_snapshot.replace(tzinfo=None),
                    "Expected property-local observation before snapshot: " + field,
                )
        if "property_id" in row:
            check(
                row["property_id"] == self.settings["property_id"],
                "Delta property differs from pinned property",
            )
        if "stream_id" in row:
            check(
                row["stream_id"] in self.property.get("streamIds", []),
                "Delta stream is not in the pinned property",
            )
        if "hour" in row:
            check(
                row["hour"] < 24
                and row["year"] == date.fromisoformat(row["date"]).year
                and row["month"] == date.fromisoformat(row["date"]).month,
                "Invalid date/hour fields",
            )
            from datetime import timedelta

            check(
                datetime.fromisoformat(row["date"]) + timedelta(hours=row["hour"] + 1)
                <= self.local_snapshot.replace(tzinfo=None),
                "Aggregate hour ends after snapshot",
            )
        if table == "cohorts":
            from datetime import timedelta

            check(
                date.fromisoformat(row["cohort_start_date"])
                + timedelta(weeks=row["activity_week"])
                <= self.local_snapshot.date(),
                "Cohort observation after snapshot",
            )
            check(
                row["cohort_user_count"] > 0
                and row["returning_user_count"] <= row["cohort_user_count"]
                and abs(
                    row["retention_rate"]
                    - row["returning_user_count"] / row["cohort_user_count"]
                )
                <= 0.000001,
                "Cohort rate needs its actual numerator and denominator",
            )
        if table == "realtime":
            from datetime import timedelta

            check(
                datetime.fromisoformat(row["minute_timestamp"]) + timedelta(minutes=1)
                <= self.local_snapshot.replace(tzinfo=None),
                "Realtime minute ends after snapshot",
            )
        if table == "attribution":
            check(
                1 <= row["touch_position"] <= row["touch_count"],
                "Invalid attribution touch position",
            )
            check(
                row["touch_timestamp"][:10]
                <= row["key_event_date"]
                < self.local_snapshot.date().isoformat(),
                "Attribution requires a completed conversion date",
            )
        if self.version == 1 and table == "events":
            check(
                timestamp(row["event_timestamp"], "analytics", "event_timestamp")
                <= self.snapshot,
                "Event observation after snapshot",
            )
        if self.version == 2 and table == "sessions":
            check(
                row["engaged_session_count"] <= row["session_count"]
                and row["key_event_session_count"] <= row["session_count"],
                "Session numerator exceeds denominator",
            )

    def _validate_delta(self):
        changes = [c for c in self.task["changes"] if c["service"] == "analytics"]
        if self.delta is None:
            check(not changes, "Analytics changes require a native delta")
            return
        check(
            self.delta["schema_version"] == self.version,
            "Analytics delta version differs from pinned source",
        )
        allowed = V2_SCHEMAS if self.version == 2 else schemas(1)
        check(
            not self.delta.keys() - {"schema_version", *allowed},
            "Unsupported Analytics delta tables",
        )
        payload = set()
        for table, rows in self.delta.items():
            if table == "schema_version":
                continue
            check(isinstance(rows, list), "Analytics rows must be arrays")
            for row in rows:
                self._validate_row(table, row)
                identity = record_key(
                    dict(table=table, **{k: row[k] for k in keys(self.version)[table]}),
                    self.version,
                )
                check(identity not in payload, "Duplicate Analytics delta key")
                payload.add(identity)
        inventory = set()
        for change in changes:
            identity = record_key(change["source"]["record"], self.version)
            check(
                change["file"] == FILE
                and identity not in inventory
                and change["operation"] in ("add", "replace"),
                "Invalid Analytics change inventory",
            )
            inventory.add(identity)
        check(payload == inventory, "Analytics delta and inventory must match exactly")

    def open(self):
        if self.tools is not None:
            return
        check(bool(self.scope), "Analytics is not configured")
        self.paths = tuple(self.acquire.inventory())
        metadata = tuple(
            "analytics/" + p
            for p in (
                "manifest.json",
                "property.json",
                "dimensions.json",
                "metrics.json",
            )
        )
        check(set(metadata) <= set(self.paths), "Pinned Analytics metadata is missing")
        self.root = self.acquire(metadata, max_file_bytes=1_048_576)
        self.property, self.manifest = load_json(
            self.root / "property.json"
        ), load_json(self.root / "manifest.json")
        self.version = self.manifest.get("schema_version", 1)
        check(self.version in (1, 2), "Unsupported Analytics source version")
        check(
            self.property.get("id") == self.settings["property_id"],
            "Pinned Analytics property mismatch",
        )
        try:
            self.local_snapshot = self.snapshot.astimezone(
                ZoneInfo(self.property["timeZone"])
            )
        except (KeyError, ValueError):
            check(False, "Pinned Analytics timezone is missing or invalid")
        allowed = {
            g.get("userId")
            for g in self.property.get("access", [])
            if g.get("role") in ("viewer", "analyst", "editor", "administrator")
        }
        check(
            self.settings["identity"] in allowed,
            "Configured Analytics identity has no pinned property grant",
        )
        self._validate_delta()
        self.work.mkdir(parents=True, exist_ok=True)
        if self.version == 2:
            build_delta(
                self.delta or {"schema_version": 2}, self.work / "overlay.sqlite"
            )
        else:
            seed = self.work / "seed.json"
            seed.write_text(json.dumps(self.delta or {}))
            build_analytics_overlay_from_seed(seed, self.work / "overlay.sqlite")
        limits = self.config["limits"]
        timeout = (
            limits.get("judge_read_timeout_seconds", 60)
            if self.arguments.get("judge_work_root")
            else limits.get("tool_timeout_seconds", 60)
        )
        try:
            self.tools = AnalyticsTools(
                AnalyticsConfig(
                    baseline_dir=self.root,
                    overlay_dir=self.work,
                    agent_user_id=self.settings["identity"],
                    reference_date=self.local_snapshot.date(),
                    snapshot_time=self.snapshot.isoformat(),
                    duckdb_memory_limit=self.arguments.get(
                        "duckdb_memory_limit", "256MB"
                    ),
                    duckdb_threads=self.arguments.get("duckdb_threads", 1),
                    duckdb_max_temp_directory_size=self.arguments.get(
                        "duckdb_max_temp_directory_size", "1GB"
                    ),
                    query_timeout_seconds=timeout,
                ),
                acquire_family=self._family,
            )
            self.tools._check_access(self.settings["property_id"])
            for change in self.task["changes"]:
                if change["service"] == "analytics":
                    before = self.lookup(change["source"]["record"], "baseline")
                    check(
                        (before is None) == (change["operation"] == "add"),
                        "Analytics add/replace does not match visible baseline",
                        change["source"]["record"],
                    )
                    after = self.lookup(change["source"]["record"])
                    check(
                        after is not None,
                        "Analytics delta is inaccessible at snapshot",
                        change["source"]["record"],
                    )
                    if self.version == 2:
                        self._validate_links(after)
            for evidence in self.task["evidence"]:
                if (
                    evidence["service"] == "analytics"
                    and "record" in evidence["source"]
                ):
                    check(
                        self.lookup(evidence["source"]["record"]) is not None,
                        "Analytics evidence is unavailable",
                        evidence["source"]["record"],
                    )
        except BaseException:
            self.close()
            raise

    def _validate_links(self, row):
        for field, table in (
            ("website_id", "websites"),
            ("landing_page_id", "pages"),
            ("page_id", "pages"),
        ):
            if not row.get(field):
                continue
            match = self.tools.analytics_read_rows(
                self.settings["property_id"],
                table,
                filters={("page_id" if table == "pages" else "website_id"): row[field]},
                page_size=2,
            )["records"]
            check(len(match) == 1, "Delta must use actual pinned website/page links")
            for key in ("website_id", "workspace_id"):
                if key in row and key in match[0]:
                    check(
                        row[key] == match[0][key],
                        "Delta website/page relationship differs from pinned catalog",
                    )

    def lookup(self, record, view="effective"):
        if self.tools is None:
            self.open()
        table, values = record_key(record, self.version)
        check(
            "analytics_read_rows" in self.scope,
            "Exact Analytics evidence requires analytics_read_rows",
        )
        result = self.tools.analytics_read_rows(
            self.settings["property_id"],
            table,
            filters=dict(zip(keys(self.version)[table], values)),
            page_size=2,
            view=view,
        )
        check(len(result["records"]) <= 1, "Ambiguous Analytics native key", record)
        return result["records"][0] if result["records"] else None

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False,
                error="unknown_tool",
                message="Unknown or disallowed Analytics tool",
            )
        required, optional = SCHEMAS[name]
        if (
            not isinstance(arguments, dict)
            or set(arguments) - {*required, *optional}
            or not set(required) <= arguments.keys()
        ):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Analytics arguments",
            )
        for field, value in arguments.items():
            types = (required | optional)[field].get("type", [])
            types = [types] if isinstance(types, str) else types
            actual = {
                str: "string",
                list: "array",
                dict: "object",
                bool: "boolean",
                int: "integer",
                type(None): "null",
            }.get(type(value))
            if actual not in types:
                return dict(
                    ok=False,
                    error="invalid_arguments",
                    message="Invalid Analytics argument types",
                )
        try:
            self.open()
            if name == "analytics_get_metadata" and "property_id" not in arguments:
                arguments = dict(property_id=self.settings["property_id"])
            result = public(getattr(self.tools, name)(**arguments))
            if len(_encoded(result)) > self.limit:
                return dict(
                    ok=False,
                    error="result_limit",
                    message="Analytics result exceeds max_result_bytes. Narrow the query or page size.",
                )
            return result
        except AnalyticsToolError as error:
            return error.as_dict()
        except (TypeError, AttributeError):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Analytics argument types",
            )

    def close(self):
        if self.tools is not None:
            tools, self.tools = self.tools, None
            tools.close()
