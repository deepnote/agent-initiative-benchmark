"""PostHog access and small native deltas inside the shared worker."""

from __future__ import annotations

import json
import sqlite3
import os
import threading
from contextlib import closing
from pathlib import Path

from benchmark.prompts import load_text
from benchmark.tools.posthog.posthog import (
    PostHogTools,
    PostHogConfig,
    PostHogToolError,
    EVENT_COLUMNS,
    PERSON_COLUMNS,
    OVERLAY_SQL,
    _json,
)
from benchmark.tools.posthog.discovery import KEYS, CATALOGS, rows as native_rows
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/posthog/seed.json"
DELTA_COLUMNS = {
    "events": EVENT_COLUMNS,
    "person_changes": PERSON_COLUMNS,
    "insights": ("id", "definition_json", "updated_at"),
    "feature_flags": ("id", "definition_json", "updated_at"),
}
TEXT = {"type": "string", "minLength": 1}
NULL_TEXT = {"type": ["string", "null"]}
PAGE = {
    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
    "start_cursor": NULL_TEXT,
}
FILTER = {
    "type": "object",
    "additionalProperties": {
        "anyOf": [
            {"type": ["string", "number", "boolean", "null"]},
            {
                "type": "object",
                "properties": {
                    k: {"type": "number"} for k in ("lt", "lte", "gt", "gte")
                },
                "additionalProperties": False,
            },
            {
                "type": "array",
                "items": {"type": ["string", "number", "boolean", "null"]},
                "minItems": 1,
                "maxItems": 100,
            },
        ]
    },
    "maxProperties": 50,
}
SCHEMAS = {
    "posthog_query_events": (
        {},
        {
            **PAGE,
            **{
                k: NULL_TEXT
                for k in (
                    "start_time",
                    "end_time",
                    "person_id",
                    "distinct_id",
                    "workspace_id",
                    "website_id",
                    "session_id",
                )
            },
            **{
                k: {"type": ["array", "null"], "items": TEXT, "maxItems": 50}
                for k in ("event_names", "selected_properties")
            },
            **{
                k: FILTER
                for k in (
                    "event_properties",
                    "person_properties",
                    "workspace_properties",
                    "website_properties",
                )
            },
            "sort_direction": {"type": "string", "enum": ["asc", "desc"]},
        },
    ),
    "posthog_query_persons": (
        {},
        {
            **PAGE,
            **{
                k: NULL_TEXT
                for k in ("person_id", "distinct_id", "search", "workspace_id")
            },
            "property_filters": FILTER,
        },
    ),
    "posthog_query_insight": (
        {"insight_id": TEXT},
        {"date_from": NULL_TEXT, "date_to": NULL_TEXT, "property_filters": FILTER},
    ),
    "posthog_get_feature_flag": ({"flag_key_or_id": TEXT}, {}),
    "posthog_get_event": ({"event_uuid": TEXT}, {}),
    "posthog_list_catalog": (
        {"kind": {"type": "string", "enum": list(CATALOGS)}},
        {
            **PAGE,
            "projection": {
                "type": "string",
                "enum": ["full", "summary"],
                "default": "full",
            },
            "insight_id": TEXT,
        },
    ),
    "posthog_query_history": (
        {
            "kind": {
                "type": "string",
                "enum": ["person_history", "distinct_ids", "feature_flags"],
            }
        },
        {**PAGE, **{k: NULL_TEXT for k in ("person_id", "distinct_id", "flag_id")}},
    ),
}
DESCRIPTIONS = json.loads(load_text("tools/posthog/descriptions.json"))
# Remove only known harness bookkeeping. Native source_event_id, stored history,
# business source properties and arbitrary business values remain evidence.
ORIGIN_FIELDS = {
    "source",
    "_origin",
    "_task_origin",
    "_benchmark_origin",
    "baseline_or_delta",
    "origin_build_id",
    "origin_task_id",
    "generation_run_id",
    "harness_origin",
    "generator_event_id",
    "generator_prompt",
    "prompt_version",
    "reviewer_notes",
    "hidden_truth",
    "hidden_correct_value",
    "reasoning",
    "reason_summary",
    "world_seed",
    "slot_id",
    "generation_quota",
    "source_knowledge",
    "source_reference_ids",
    "customer_key",
    "customer_profile_id",
    "customer_internal_id",
    "fixture_version",
}


def public(value, properties=False):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k in ORIGIN_FIELDS - {"source"} or (
                k == "source" and not properties and v in ("baseline", "overlay")
            ):
                continue
            business = (
                properties
                or k.endswith("_properties")
                or k
                in (
                    "properties",
                    "filters",
                    "person_filters",
                    "group_filters",
                    "history",
                    "evidence",
                    "payloads",
                    "result",
                    "saved_result",
                )
            )
            if k.endswith("_properties_json") or k == "properties_json":
                result[k] = _json(public(json.loads(v), True))
            else:
                result[k] = public(v, business)
        return result
    if isinstance(value, list):
        return [public(v, properties) for v in value]
    return value


class PostHogEnvironmentError(ValueError):
    def __init__(self, reason, code="invalid_posthog_environment", record=None):
        self.code, self.record, self.source, self.validation_reason = (
            code,
            record,
            "posthog",
            reason,
        )
        super().__init__(reason)


def check(ok, reason, record=None):
    if not ok:
        raise PostHogEnvironmentError(reason, record=record)


def record_key(record):
    table = record.get("table")
    check(
        table in KEYS and set(record) == {"table", *KEYS[table]},
        "Exact native PostHog key required",
    )
    check(
        all(isinstance(record[k], str) and record[k] for k in KEYS[table]),
        "Nonempty native keys required",
    )
    return table, tuple(record[k] for k in KEYS[table])


def tool_definitions(scope, cap):
    return [
        dict(
            name=name,
            description=DESCRIPTIONS[name],
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


def valid_argument(schema, value):
    if "anyOf" in schema:
        return any(valid_argument(s, value) for s in schema["anyOf"])
    kind = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }.get(type(value))
    types = schema.get("type", [])
    types = types if isinstance(types, list) else [types]
    if kind not in types and not (kind == "integer" and "number" in types):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, str):
        return len(value) >= schema.get("minLength", 0) and len(value) <= 10000
    if kind in ("integer", "number"):
        import math

        return math.isfinite(value) and schema.get(
            "minimum", -1e100
        ) <= value <= schema.get("maximum", 1e100)
    if kind == "array":
        return schema.get("minItems", 0) <= len(value) <= schema.get(
            "maxItems", 1000
        ) and all(valid_argument(schema["items"], v) for v in value)
    if kind == "object":
        fields = schema.get("properties", {})
        extra = schema.get("additionalProperties", False)
        return (
            len(value) <= schema.get("maxProperties", 100)
            and set(schema.get("required", [])) <= value.keys()
            and all(
                (
                    valid_argument(fields[k], v)
                    if k in fields
                    else isinstance(extra, dict) and valid_argument(extra, v)
                )
                for k, v in value.items()
            )
        )
    return True


class PostHogEnvironment:
    def __init__(self, arguments, manifest, task, config, acquire):
        self.scope = set(config["tool_scope"]) & SCHEMAS.keys()
        self.task, self.limit, self.acquire = (
            task,
            arguments["max_result_bytes"],
            acquire,
        )
        self.snapshot = timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.settings = config.get("service_settings", {}).get("posthog")
        if self.scope:
            check(
                isinstance(self.settings, dict)
                and set(self.settings)
                == {"project_id", "identity", "read_project_ids"},
                "PostHog requires explicit project_id, identity and read_project_ids service settings",
            )
            check(
                all(
                    isinstance(self.settings[k], str) and self.settings[k].strip()
                    for k in ("project_id", "identity")
                )
                and isinstance(self.settings["read_project_ids"], list)
                and self.settings["project_id"] in self.settings["read_project_ids"],
                "PostHog identity has no project read grant",
            )
        self.work = Path(arguments["work_dir"]) / "posthog"
        self.native_config = dict(
            duckdb_memory_limit=arguments.get("duckdb_memory_limit", "256MB"),
            duckdb_threads=arguments.get("duckdb_threads", 1),
            duckdb_max_temp_directory_size=arguments.get(
                "duckdb_max_temp_directory_size", "1GB"
            ),
            max_query_seconds=(
                config["limits"].get("judge_read_timeout_seconds", 60)
                if arguments.get("judge_work_root")
                else config["limits"].get("tool_timeout_seconds", 60)
            ),
        )
        self.tools, self.paths, self.root = None, (), None
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/posthog/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8_388_608 for f in entries),
            "PostHog accepts only seed.json up to 8 MiB",
        )
        self.delta = load_json(Path(arguments["package"]) / FILE) if entries else {}
        check(
            isinstance(self.delta, dict)
            and not self.delta.keys() - DELTA_COLUMNS.keys(),
            "Unsupported PostHog delta tables",
        )
        check(
            not entries or self.scope, "PostHog delta requires configured PostHog reads"
        )
        payload = set()
        for table, records in self.delta.items():
            check(isinstance(records, list), "PostHog rows must be arrays")
            for row in records:
                check(
                    isinstance(row, dict) and set(row) == set(DELTA_COLUMNS[table]),
                    "Complete native PostHog rows required",
                )
                self._validate_row(table, row)
                key = (table, tuple(row[k] for k in KEYS[table]))
                check(key not in payload, "Duplicate PostHog delta key")
                payload.add(key)
        inventory = set()
        for change in task["changes"]:
            if change["service"] == "posthog":
                key = record_key(change["source"]["record"])
                check(
                    change["file"] == FILE and key not in inventory,
                    "Invalid or duplicate PostHog change inventory",
                )
                inventory.add(key)
        check(
            payload == inventory,
            "PostHog delta and change inventory must match exactly",
        )
        check(
            not (
                entries
                or inventory
                or any(e["service"] == "posthog" for e in task["evidence"])
            )
            or self.scope,
            "PostHog task data requires configured PostHog reads",
        )

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def _observed(self, value, field):
        when = timestamp(value, "posthog", field)
        check(when <= self.snapshot, field + ": observed state after snapshot")
        check(
            value == when.isoformat().replace("+00:00", "Z"),
            field + ": use canonical UTC timestamps ending in Z",
        )
        return when

    def _validate_row(self, table, row):
        nullable = (
            {"person_id", "website_id", "session_id"}
            if table == "events"
            else {"valid_to"}
        )
        for key, value in row.items():
            check(
                (value is None and key in nullable) or isinstance(value, str),
                "Invalid PostHog row type: " + key,
            )
        record_key({"table": table, **{k: row[k] for k in KEYS[table]}})
        if table in ("events", "person_changes"):
            check(
                row["project_id"] == self.settings["project_id"],
                "Delta project differs from configured project",
            )
            for key, value in row.items():
                if key.endswith("_json"):
                    try:
                        parsed = json.loads(value)
                    except ValueError:
                        check(False, "Invalid native JSON: " + key)
                    check(
                        isinstance(
                            parsed, list if key == "distinct_ids_json" else dict
                        ),
                        "Invalid native JSON: " + key,
                    )
            created = self._observed(
                row["timestamp"] if table == "events" else row["created_at"],
                "timestamp" if table == "events" else "created_at",
            )
            if table == "person_changes":
                start = self._observed(row["valid_from"], "valid_from")
                check(created <= start, "Person created after interval start")
                if row["valid_to"] is not None:
                    check(
                        start < self._observed(row["valid_to"], "valid_to"),
                        "Invalid profile interval",
                    )
            check(bool(row["workspace_id"]), "Workspace identity is required")
        else:
            try:
                definition = json.loads(row["definition_json"])
            except ValueError:
                check(False, "Invalid complete definition JSON")
            check(
                isinstance(definition, dict)
                and definition.get("id") == row["id"]
                and definition.get("project_id") == self.settings["project_id"],
                "Invalid complete definition identity/project",
            )
            check(
                "_complete_replacement" not in definition
                and "source" not in definition,
                "Definition contains harness fields",
            )
            self._observed(row["updated_at"], "updated_at")
            required = (
                {
                    "id",
                    "name",
                    "project_id",
                    "type",
                    "date_from",
                    "date_to",
                    "event_names",
                    "filters",
                    "breakdowns",
                    "last_refreshed_at",
                    "result",
                }
                if table == "insights"
                else {
                    "id",
                    "key",
                    "name",
                    "project_id",
                    "active",
                    "created_at",
                    "updated_at",
                }
            )
            check(required <= definition.keys(), "Incomplete saved definition")
            check(
                isinstance(definition["name"], str) and definition["name"].strip(),
                "Definition name required",
            )
            for key in (
                "created_at",
                "updated_at",
                "last_refreshed_at",
                "last_modified_at",
                "last_viewed_at",
            ):
                if key in definition:
                    check(
                        self._observed(definition[key], key)
                        <= timestamp(row["updated_at"], "posthog", "updated_at"),
                        "Definition observation is after task update",
                    )
            if "created_at" in definition and "updated_at" in definition:
                check(
                    definition["created_at"] <= definition["updated_at"],
                    "Definition updated before creation",
                )
            check(
                isinstance(definition.get("history", []), list),
                "Flag history must be an array",
            )
            for history in definition.get("history", []):
                check(
                    isinstance(history, dict) and "timestamp" in history,
                    "Stored history timestamp is required",
                )
                check(
                    self._observed(history["timestamp"], "history.timestamp")
                    <= timestamp(row["updated_at"], "posthog", "updated_at"),
                    "Flag history is after task update",
                )
            if table == "insights":
                check(
                    isinstance(definition["type"], str)
                    and isinstance(definition["filters"], dict)
                    and all(
                        isinstance(definition[k], list)
                        and all(isinstance(v, str) and v for v in definition[k])
                        for k in ("event_names", "breakdowns")
                    ),
                    "Invalid saved insight definition",
                )
                check(
                    timestamp(definition["date_from"], "posthog", "date_from")
                    < timestamp(definition["date_to"], "posthog", "date_to")
                    <= self.snapshot,
                    "Invalid insight dates",
                )
            else:
                check(
                    isinstance(definition["key"], str)
                    and definition["key"]
                    and type(definition["active"]) is bool,
                    "Invalid feature flag definition",
                )
                if "rollout_percentage" in definition:
                    check(
                        type(definition["rollout_percentage"]) in (int, float)
                        and 0 <= definition["rollout_percentage"] <= 100,
                        "Invalid feature flag rollout percentage",
                    )

    def _family(self, family):
        suffix = family + (
            ".json"
            if family in ("insights", "feature_flags", "event_definitions")
            else ".parquet"
        )
        paths = (
            tuple(
                p
                for p in self.paths
                if p.startswith("posthog/events/") and p.endswith(".parquet")
            )
            if family == "events"
            else tuple(p for p in self.paths if p == "posthog/" + suffix)
        )
        if not paths:
            raise PostHogToolError(
                "not_available", "Required PostHog data family is absent: " + family
            )
        self.acquire(paths)
        if family in ("insights", "feature_flags", "event_definitions"):
            check(
                (self.root / suffix).stat().st_size <= 16_777_216,
                "PostHog catalog exceeds 16 MiB",
            )

    def open(self):
        if self.tools is not None:
            return
        check(bool(self.scope), "PostHog is not configured")
        self.paths = tuple(self.acquire.inventory())
        check(
            "posthog/manifest.json" in self.paths, "Pinned PostHog manifest is missing"
        )
        self.root = self.acquire(("posthog/manifest.json",))
        check(
            (self.root / "manifest.json").stat().st_size <= 1_048_576,
            "PostHog project manifest exceeds 1 MiB",
        )
        from zoneinfo import ZoneInfo

        project = load_json(self.root / "manifest.json").get("project", {})
        check(
            project.get("id") == self.settings["project_id"]
            and isinstance(project.get("timezone"), str),
            "Pinned project settings are missing or incompatible",
        )
        try:
            ZoneInfo(project["timezone"])
        except (ValueError, KeyError):
            check(False, "Pinned PostHog timezone is invalid")
        self.work.mkdir(parents=True)
        self._compile()
        prior_scratch = os.environ.get("BENCHMARK_SCRATCH_DIR")
        scratch = self.work / "scratch"
        scratch.mkdir()
        os.environ["BENCHMARK_SCRATCH_DIR"] = str(scratch)
        try:
            self.tools = PostHogTools(
                PostHogConfig(
                    baseline_dir=self.root,
                    overlay_dir=self.work,
                    project_id=self.settings["project_id"],
                    default_page_size=100,
                    max_page_size=1000,
                    max_result_bytes=self.limit,
                    snapshot_time=self.snapshot.isoformat(),
                    **self.native_config,
                ),
                acquire_family=self._family,
            )
            self._validate_effective()
        except BaseException:
            self.close()
            raise
        finally:
            if prior_scratch is None:
                os.environ.pop("BENCHMARK_SCRATCH_DIR", None)
            else:
                os.environ["BENCHMARK_SCRATCH_DIR"] = prior_scratch

    def _compile(self):
        # This compiler consumes complete task records only. It never reads a
        # seed, regenerates baseline events or uses fixture event positions.
        with closing(sqlite3.connect(self.work / "overlay.sqlite")) as db:
            db.executescript(OVERLAY_SQL)
            db.execute(
                "CREATE TABLE person_versions ("
                + ",".join(c + " TEXT" for c in PERSON_COLUMNS)
                + ", PRIMARY KEY(person_id,valid_from))"
            )
            for table, records in self.delta.items():
                for row in records:
                    if table == "events":
                        values = [row[c] for c in EVENT_COLUMNS] + [
                            "upsert",
                            self.snapshot.isoformat(),
                        ]
                        db.execute(
                            "INSERT INTO events VALUES ("
                            + ",".join("?" for _ in values)
                            + ")",
                            values,
                        )
                    elif table == "person_changes":
                        db.execute(
                            "INSERT INTO person_versions VALUES ("
                            + ",".join("?" for _ in PERSON_COLUMNS)
                            + ")",
                            [row[c] for c in PERSON_COLUMNS],
                        )
                    else:
                        definition = json.loads(row["definition_json"])
                        db.execute(
                            "INSERT INTO " + table + " VALUES (?,?,?,?)",
                            (
                                row["id"],
                                _json(dict(definition, _complete_replacement=True)),
                                "upsert",
                                row["updated_at"],
                            ),
                        )
            db.commit()

    def lookup(self, record, view="effective"):
        table, _ = record_key(record)
        self.open()
        result = native_rows(
            self.tools,
            table,
            {k: record[k] for k in KEYS[table]},
            baseline=view == "baseline",
            limit=2,
        )
        check(len(result) <= 1, "Ambiguous native PostHog key", record)
        return result[0] if result else None

    def _validate_effective(self):
        for change in self.task["changes"]:
            if change["service"] != "posthog":
                continue
            record = change["source"]["record"]
            before = self.lookup(record, "baseline")
            check(
                (before is not None) == (change["operation"] == "replace"),
                "PostHog add/replace conflicts with pinned existence",
                record,
            )
            after = self.lookup(record)
            check(after is not None, "Missing effective PostHog record", record)
            table = record["table"]
            if table in ("events", "person_changes"):
                workspace = native_rows(
                    self.tools,
                    "workspace_groups",
                    {"id": after["workspace_id"]},
                    limit=2,
                )
                check(len(workspace) == 1, "Unknown PostHog workspace", record)
                if table == "events":
                    website = after.get("website_id")
                    if website:
                        groups = native_rows(
                            self.tools, "website_groups", {"id": website}, limit=2
                        )
                        check(
                            len(groups) == 1
                            and groups[0]["workspace_id"] == after["workspace_id"],
                            "Unknown or incompatible website",
                            record,
                        )
                    identities = native_rows(
                        self.tools,
                        "distinct_ids",
                        {"distinct_id": after["distinct_id"]},
                        limit=2,
                    )
                    person = after.get("person_id")
                    if (
                        not identities
                        and person is None
                        and website
                        and after["distinct_id"] == "service:" + website
                    ):
                        # System publishers occur in the pinned event stream but
                        # are not people in the distinct-ID directory. Reuse only
                        # a previously observed actor for this exact site/scope.
                        known = self.tools._bounded_rows(
                            "SELECT 1 AS known FROM baseline_events WHERE project_id=? AND workspace_id=? "
                            "AND website_id=? AND distinct_id=? AND person_id IS NULL AND timestamp<=? LIMIT 1",
                            [
                                self.settings["project_id"],
                                after["workspace_id"],
                                website,
                                after["distinct_id"],
                                after["timestamp"],
                            ],
                        )
                        check(
                            bool(known)
                            and json.loads(after["person_properties_json"]) == {},
                            "Unknown or incompatible distinct ID",
                            record,
                        )
                    else:
                        check(
                            len(identities) == 1
                            and identities[0]["workspace_id"] == after["workspace_id"],
                            "Unknown or incompatible distinct ID",
                            record,
                        )
                        check(
                            not person or identities[0]["person_id"] == person,
                            "Conflicting native identity link",
                            record,
                        )
                    if person:
                        people = native_rows(
                            self.tools, "person_history", {"person_id": person}
                        )
                        check(
                            any(
                                p["workspace_id"] == after["workspace_id"]
                                and p["valid_from"] <= after["timestamp"]
                                and (
                                    p["valid_to"] is None
                                    or after["timestamp"] < p["valid_to"]
                                )
                                for p in people
                            ),
                            "Missing person interval at event time",
                            record,
                        )
                    if "posthog/event_definitions.json" in self.paths:
                        check(
                            len(
                                native_rows(
                                    self.tools,
                                    "event_definitions",
                                    {"name": after["event_name"]},
                                    limit=2,
                                )
                            )
                            == 1,
                            "Unknown or duplicate event definition",
                            record,
                        )
                    else:
                        check(
                            self.tools._duck.execute(
                                "SELECT 1 FROM baseline_events WHERE project_id=? AND event_name=? LIMIT 1",
                                [self.settings["project_id"], after["event_name"]],
                            ).fetchone()
                            is not None,
                            "Unknown event name",
                            record,
                        )
                else:
                    check(
                        all(
                            isinstance(i, str) and i
                            for i in json.loads(after["distinct_ids_json"])
                        ),
                        "Invalid person distinct IDs",
                        record,
                    )
                    for distinct in json.loads(after["distinct_ids_json"]):
                        links = native_rows(
                            self.tools,
                            "distinct_ids",
                            {"distinct_id": distinct},
                            limit=2,
                        )
                        check(
                            len(links) == 1
                            and links[0]["person_id"] == after["person_id"]
                            and links[0]["workspace_id"] == after["workspace_id"],
                            "Person profile cannot invent an identity link",
                            record,
                        )
                    people = native_rows(
                        self.tools, "person_history", {"person_id": after["person_id"]}
                    )
                    ordered = sorted(people, key=lambda r: r["valid_from"])
                    for left, right in zip(ordered, ordered[1:]):
                        check(
                            left["valid_to"] is not None
                            and left["valid_to"] <= right["valid_from"],
                            "Overlapping profile intervals",
                            record,
                        )
            elif table == "feature_flags":
                if before:
                    check(
                        before["key"] == after["key"],
                        "Flag replacement cannot change native key",
                        record,
                    )
                    old_history = before.get("history", [])
                    check(
                        after.get("history", [])[: len(old_history)] == old_history,
                        "Flag replacement must preserve stored history",
                        record,
                    )
                flags = self.tools._feature_flags.values()
                check(
                    sum(f["key"] == after["key"] for f in flags) == 1,
                    "Duplicate effective feature flag key",
                    record,
                )
        # A replacement is a whole logical definition. The native patch table
        # stores an internal complete marker and never retains omitted old keys.

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False,
                error="unknown_tool",
                message="Unknown or disallowed PostHog tool",
            )
        schema = next(
            d["parameters"] for d in self.tool_definitions if d["name"] == name
        )
        if not valid_argument(schema, arguments):
            return dict(
                ok=False, error="invalid_arguments", message="Invalid PostHog arguments"
            )
        try:
            self.open()
            families = {
                "posthog_query_events": ["events"],
                "posthog_get_event": ["events"],
                "posthog_query_persons": ["person_history", "events"],
                "posthog_query_insight": ["insights", "events"],
                "posthog_get_feature_flag": ["feature_flags"],
            }.get(name, [arguments.get("kind")])
            if name == "posthog_query_persons" and arguments.get("distinct_id"):
                families += ["distinct_ids"]
            for family in families:
                self.tools._ensure(family)
            timer = threading.Timer(
                self.tools.config.max_query_seconds, self.tools._duck.interrupt
            )
            timer.start()
            try:
                result = public(getattr(self.tools, name)(**arguments))
            finally:
                timer.cancel()
                timer.join()
            if len(_encoded(result)) > self.limit:
                return dict(
                    ok=False,
                    error="result_limit",
                    message="PostHog result exceeds max_result_bytes. Narrow the query or page size.",
                )
            return result
        except PostHogToolError as error:
            return error.as_dict()
        except Exception as error:
            import duckdb

            if isinstance(error, duckdb.InterruptException):
                return dict(
                    ok=False,
                    error="query_timeout",
                    message="PostHog query exceeded its time limit",
                )
            if isinstance(error, (duckdb.OutOfMemoryException, duckdb.IOException)):
                return dict(
                    ok=False,
                    error="query_resource_limit",
                    message="Exact PostHog query cannot fit the configured memory or scratch limits",
                )
            raise

    def close(self):
        if self.tools is not None:
            tools, self.tools = self.tools, None
            tools.close()
