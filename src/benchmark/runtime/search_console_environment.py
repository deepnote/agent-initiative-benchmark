"""Native Search Console task state in the existing bounded worker/cache."""

from __future__ import annotations
import json
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from benchmark.prompts import load_text
from benchmark.tools.search_console.search_console import (
    SearchConsoleTools,
    SearchConsoleConfig,
    SearchConsoleToolError,
    OVERLAY_SQL,
    OVERLAY_COLUMNS,
    _parse_date,
)
from benchmark.tools.search_console.discovery import (
    KEYS,
    PERFORMANCE_COLUMNS,
    rows,
    catalog,
    clean,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/search_console/seed.json"
DELTA_COLUMNS = {
    "performance_rows": PERFORMANCE_COLUMNS,
    "inspection_overrides": (
        "property_id",
        "inspection_url",
        "observed_at",
        "snapshot_json",
    ),
}
TEXT = dict(type="string", minLength=1, maxLength=4096)
PAGE = {
    "row_limit": dict(type="integer", minimum=1, maximum=1000),
    "start_row": dict(type="integer", minimum=0, maximum=9007199254740991),
}
SITE = {"site_url": TEXT}
FILTER = dict(
    type="array",
    maxItems=10,
    items=dict(
        type="object",
        properties={
            "groupType": dict(type="string", enum=["and"]),
            "filters": dict(
                type="array",
                maxItems=20,
                items=dict(
                    type="object",
                    properties={
                        "dimension": dict(
                            type="string",
                            enum=[
                                "query",
                                "page",
                                "country",
                                "device",
                                "searchAppearance",
                            ],
                        ),
                        "operator": dict(
                            type="string",
                            enum=[
                                "equals",
                                "notEquals",
                                "contains",
                                "notContains",
                                "includingRegex",
                                "excludingRegex",
                            ],
                        ),
                        "expression": TEXT,
                    },
                    required=["dimension", "operator", "expression"],
                    additionalProperties=False,
                ),
            ),
        },
        required=["filters"],
        additionalProperties=False,
    ),
)


def obj(props):
    return dict(
        type="object",
        properties=props,
        required=list(props),
        additionalProperties=False,
    )


def record_shapes():
    return [
        obj(dict(table=dict(type="string", enum=[t]), **{k: TEXT for k in keys}))
        for t, keys in KEYS.items()
    ]


SCHEMAS = {
    "search_console_query_performance": (
        {"start_date": TEXT, "end_date": TEXT},
        {
            **SITE,
            **PAGE,
            "dimensions": dict(
                type="array",
                maxItems=6,
                items=dict(
                    type="string",
                    enum=[
                        "date",
                        "query",
                        "page",
                        "country",
                        "device",
                        "searchAppearance",
                    ],
                ),
            ),
            "dimension_filter_groups": FILTER,
            "search_type": dict(type="string", enum=["web"]),
            "aggregation_type": dict(
                type="string", enum=["auto", "byPage", "byProperty"]
            ),
        },
    ),
    "search_console_inspect_url": (
        {"inspection_url": TEXT},
        {**SITE, "language_code": TEXT},
    ),
    "search_console_list_properties": ({}, PAGE),
    "search_console_list_pages": ({}, {**SITE, **PAGE}),
    "search_console_list_queries": ({}, {**SITE, **PAGE}),
    "search_console_get_record": ({"record": dict(anyOf=record_shapes())}, {}),
}
DESCRIPTIONS = json.loads(load_text("tools/search_console/descriptions.json"))


def valid(schema, value):
    if "anyOf" in schema:
        return any(valid(s, value) for s in schema["anyOf"])
    kind = {
        str: "string",
        int: "integer",
        float: "number",
        dict: "object",
        list: "array",
        bool: "boolean",
        type(None): "null",
    }.get(type(value))
    types = schema.get("type", [])
    if kind not in (types if isinstance(types, list) else [types]) and not (
        kind == "integer" and types == "number"
    ):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "string":
        return (
            schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 100000)
        )
    if kind in ("number", "integer"):
        return math.isfinite(value) and schema.get(
            "minimum", -1e100
        ) <= value <= schema.get("maximum", 1e100)
    if kind == "array":
        return len(value) <= schema.get("maxItems", 1000) and all(
            valid(schema["items"], v) for v in value
        )
    if kind == "object":
        fields = schema.get("properties", {})
        return (
            set(schema.get("required", [])) <= value.keys()
            and not value.keys() - fields.keys()
            and all(valid(fields[k], v) for k, v in value.items())
        )
    return True


def tool_definitions(scope, cap):
    return [
        dict(
            name=n,
            description=DESCRIPTIONS[n],
            parameters=dict(
                type="object",
                properties={**r, **o},
                required=list(r),
                additionalProperties=False,
            ),
        )
        for n, (r, o) in SCHEMAS.items()
        if n in scope
    ]


class SearchConsoleEnvironmentError(ValueError):
    def __init__(self, reason, record=None):
        self.code, self.source, self.record, self.validation_reason = (
            "invalid_search_console_environment",
            "search_console",
            record,
            reason,
        )
        super().__init__(reason)


def check(ok, reason, record=None):
    if not ok:
        raise SearchConsoleEnvironmentError(reason, record)


def record_key(record):
    table = record.get("table")
    check(
        table in KEYS
        and set(record) == {"table", *KEYS[table]}
        and all(isinstance(record[k], str) and record[k] for k in KEYS[table]),
        "Exact native Search Console key required",
    )
    return table, tuple(record[k] for k in KEYS[table])


class SearchConsoleEnvironment:
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
        self.day = self.snapshot.date().isoformat()
        self.settings = config.get("service_settings", {}).get("search_console")
        if self.scope:
            check(
                isinstance(self.settings, dict)
                and set(self.settings)
                == {"property_id", "identity", "read_property_ids"},
                "Search Console requires explicit property_id, identity and read_property_ids",
            )
            check(
                all(
                    isinstance(self.settings[k], str) and self.settings[k].strip()
                    for k in ("property_id", "identity")
                )
                and isinstance(self.settings["read_property_ids"], list)
                and self.settings["property_id"] in self.settings["read_property_ids"],
                "Search Console identity has no property grant",
            )
        self.work = Path(arguments["work_dir"]) / "search_console"
        self.options = dict(
            duckdb_memory_limit=arguments.get("duckdb_memory_limit", "256MB"),
            duckdb_threads=arguments.get("duckdb_threads", 1),
            duckdb_max_temp_directory_size=arguments.get(
                "duckdb_max_temp_directory_size", "1GB"
            ),
            query_timeout_seconds=config["limits"].get("tool_timeout_seconds", 60),
            max_start_row=9007199254740991,
            default_row_limit=100,
            max_row_limit=1000,
        )
        self.tools, self.paths, self.root = None, (), None
        entries = [
            f
            for f in manifest["files"]
            if f["path"].startswith("delta/search_console/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8_388_608 for f in entries),
            "Search Console accepts only seed.json up to 8 MiB",
        )
        self.delta = load_json(Path(arguments["package"]) / FILE) if entries else {}
        check(
            isinstance(self.delta, dict)
            and not self.delta.keys() - DELTA_COLUMNS.keys(),
            "Unsupported Search Console delta tables",
        )
        check(
            not entries or self.scope, "Search Console delta requires configured reads"
        )
        payload = set()
        for table, records in self.delta.items():
            check(isinstance(records, list), "Native rows must be arrays")
            for row in records:
                check(
                    isinstance(row, dict) and set(row) == set(DELTA_COLUMNS[table]),
                    "Complete native Search Console rows required",
                )
                self._validate_row(table, row)
                key = record_key(dict(table=table, **{k: row[k] for k in KEYS[table]}))
                check(key not in payload, "Duplicate Search Console delta key")
                payload.add(key)
        inventory = set()
        for change in task["changes"]:
            if change["service"] == "search_console":
                key = record_key(change["source"]["record"])
                check(
                    change["file"] == FILE
                    and change["operation"] in ("add", "replace")
                    and key not in inventory,
                    "Invalid Search Console change inventory",
                )
                inventory.add(key)
        check(
            payload == inventory, "Search Console delta and changes must match exactly"
        )
        check(
            not (
                entries
                or inventory
                or any(e["service"] == "search_console" for e in task["evidence"])
            )
            or self.scope,
            "Search Console evidence requires configured reads",
        )
        self._validating = False
        self._validated = set()

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def observed(self, value, field):
        when = timestamp(value, "search_console", field)
        check(when <= self.snapshot, field + ": observation after snapshot")
        return when

    def _validate_row(self, table, row):
        check(
            row["property_id"] == self.settings["property_id"],
            "Delta property differs from pinned access",
        )
        for k in KEYS[table]:
            check(
                isinstance(row[k], str) and row[k],
                "Native identities must be nonempty strings",
            )
        if table == "performance_rows":
            check(
                _parse_date(row["date"], "date").isoformat() <= self.day,
                "Performance date after snapshot",
            )
            check(
                all(
                    isinstance(row[k], str) and row[k]
                    for k in (
                        "query_id",
                        "page_id",
                        "country_code",
                        "device",
                        "search_type",
                    )
                ),
                "Invalid performance dimensions",
            )
            check(
                re.fullmatch("[A-Z]{3}", row["country_code"])
                and row["device"] in ("DESKTOP", "MOBILE", "TABLET")
                and row["search_type"] == "web",
                "Invalid performance dimensions",
            )
            check(
                row["search_appearance"] is None
                or isinstance(row["search_appearance"], str),
                "Invalid search appearance",
            )
            check(
                type(row["clicks"]) is int
                and type(row["impressions"]) is int
                and 0 <= row["clicks"] <= row["impressions"] <= 9223372036854775807,
                "Invalid clicks/impressions",
            )
            check(
                type(row["position_sum"]) in (int, float)
                and math.isfinite(row["position_sum"])
                and row["position_sum"] >= row["impressions"]
                and (row["impressions"] > 0 or row["position_sum"] == 0),
                "Invalid position_sum denominator",
            )
        else:
            check(
                SearchConsoleTools._normalize_url(row["inspection_url"])
                == row["inspection_url"],
                "Use the normalized inspection URL",
            )
            observed = self.observed(row["observed_at"], "observed_at")
            check(
                isinstance(row["snapshot_json"], str),
                "snapshot_json must contain the complete stored result",
            )
            try:
                snapshot = json.loads(row["snapshot_json"])
            except (ValueError, TypeError):
                check(False, "Invalid snapshot_json")
            self._inspection_time(snapshot, observed)
            check(
                isinstance(snapshot, dict)
                and {"inspectionResultLink", "indexStatusResult"} <= snapshot.keys(),
                "Incomplete inspection payload",
            )
            status = snapshot["indexStatusResult"]
            check(
                isinstance(status, dict)
                and {
                    "verdict",
                    "coverageState",
                    "robotsTxtState",
                    "indexingState",
                    "pageFetchState",
                    "userCanonical",
                    "crawledAs",
                }
                <= status.keys(),
                "Incomplete stored index status",
            )

    def _inspection_time(self, snapshot, observed=None):
        check(isinstance(snapshot, dict), "Invalid inspection payload")
        status = snapshot.get("indexStatusResult", {})
        check(isinstance(status, dict), "Invalid indexStatusResult")
        crawl = status.get("lastCrawlTime")
        if crawl:
            check(
                self.observed(crawl, "lastCrawlTime") <= (observed or self.snapshot),
                "Crawl time after inspection observation",
            )

    def _family(self, family):
        if family == "performance":
            paths = tuple(
                p
                for p in self.paths
                if p.startswith("search_console/performance/")
                and p.endswith(".parquet")
            )
        elif family.startswith("inspection:"):
            bucket = family.split(":")[1]
            check(re.fullmatch(r"\d{2}", bucket), "Invalid inspection bucket")
            paths = (
                "search_console/url-inspection/bucket=" + bucket + "/snapshots.json",
            )
        else:
            paths = ("search_console/" + family + ".parquet",)
        if not paths or not set(paths) <= set(self.paths):
            raise SearchConsoleToolError(
                "data_error",
                "Required native Search Console data is missing: " + family,
            )
        self.acquire(
            paths,
            max_file_bytes=16_777_216 if family.startswith("inspection:") else None,
        )

    def open(self):
        if self.tools is not None:
            return
        check(bool(self.scope), "Search Console is not configured")
        self.paths = tuple(self.acquire.inventory())
        self.root = self.acquire(
            ("search_console/properties.json",), max_file_bytes=1_048_576
        )
        self.work.mkdir(parents=True)
        self._compile()
        self.tools = SearchConsoleTools(
            SearchConsoleConfig(
                baseline_dir=self.root,
                overlay_dir=self.work,
                property_id=self.settings["property_id"],
                **self.options,
            ),
            acquire_family=self._family,
        )
        if self.tools._property.get("permission_level") not in (
            "siteOwner",
            "siteFullUser",
            "siteRestrictedUser",
        ):
            self.close()
            check(False, "Pinned property has no read permission")

    def _validate_delta(self, table, requested=None):
        validation_key = (
            (table, requested["inspection_url"])
            if table == "inspection_overrides" and requested
            else table
        )
        if self._validating or validation_key in self._validated:
            return
        self._validating = True
        try:
            if table == "performance_rows":
                for row in self.delta.get(table, []):
                    # Task associations cannot grant access to a global query row.
                    query = rows(
                        self.tools,
                        "queries",
                        dict(property_id=row["property_id"], query_id=row["query_id"]),
                        baseline=True,
                        limit=2,
                        snapshot_date=self.day,
                    )
                    check(
                        len(query) == 1,
                        "Performance query is not visible in the pinned property baseline",
                        dict(
                            table=table,
                            property_id=row["property_id"],
                            source_event_id=row["source_event_id"],
                        ),
                    )
            for change in self.task["changes"]:
                if (
                    change["service"] != "search_console"
                    or change["source"]["record"]["table"] != table
                ):
                    continue
                record = change["source"]["record"]
                if (
                    table == "inspection_overrides"
                    and requested
                    and record != requested
                ):
                    continue
                if table == "performance_rows":
                    self.tools._ensure("queries")
                    self.tools._performance_view()
                    original = self.tools._execute_rows(
                        "SELECT p.property_id,q.privacy_hidden FROM baseline_performance p LEFT JOIN queries q ON q.query_id=p.query_id WHERE source_event_id=? LIMIT 2",
                        [record["source_event_id"]],
                    )
                    check(
                        len(original) <= 1
                        and all(
                            r["property_id"] == self.settings["property_id"]
                            and r["privacy_hidden"] is False
                            for r in original
                        ),
                        "Replacement source is private or outside the property",
                        record,
                    )
                before = self.lookup(record, "baseline")
                check(
                    (before is not None) == (change["operation"] == "replace"),
                    "Search Console add/replace conflicts with pinned existence",
                    record,
                )
                check(
                    self.lookup(record) is not None,
                    "Missing or private effective Search Console record",
                    record,
                )
            self._validated.add(validation_key)
        finally:
            self._validating = False

    def _compile(self):
        with closing(sqlite3.connect(self.work / "overlay.sqlite")) as db:
            db.executescript(OVERLAY_SQL)
            for row in self.delta.get("performance_rows", []):
                values = [
                    row[k] if k != "operation" else "upsert" for k in OVERLAY_COLUMNS
                ] + [self.snapshot.isoformat()]
                db.execute(
                    "INSERT INTO performance_rows VALUES ("
                    + ",".join("?" for _ in values)
                    + ")",
                    values,
                )
            db.commit()
        overrides = [
            dict(
                property_id=r["property_id"],
                inspection_url=r["inspection_url"],
                observed_at=r["observed_at"],
                operation="upsert",
                snapshot=json.loads(r["snapshot_json"]),
            )
            for r in self.delta.get("inspection_overrides", [])
        ]
        (self.work / "inspection-overrides.json").write_text(
            json.dumps(dict(inspection_overrides=overrides))
        )

    def lookup(self, record, view="effective"):
        table, _ = record_key(record)
        self.open()
        self._validate_delta(
            "performance_rows" if table == "queries" else table, record
        )
        if record.get("property_id", record.get("id")) != self.settings["property_id"]:
            raise SearchConsoleToolError(
                "permission_denied", "Property is outside the configured grant"
            )
        if table == "inspection_overrides":
            url = record["inspection_url"]
            self._check_url(url)
            prior = self.tools._inspection_overrides
            if view == "baseline":
                self.tools._inspection_overrides = {}
            try:
                result = self.tools.search_console_inspect_url(url)
                self._inspection_time(
                    result["inspectionResult"],
                    (
                        self.observed(result["observedAt"], "observedAt")
                        if result["observedAt"]
                        else None
                    ),
                )
                return dict(
                    property_id=result["siteUrl"],
                    inspection_url=result["inspectionUrl"],
                    observed_at=result["observedAt"],
                    observation_time_available=result["observationTimeAvailable"],
                    snapshot=result["inspectionResult"],
                )
            except SearchConsoleToolError as error:
                if error.code == "url_not_found":
                    return None
                raise
            finally:
                self.tools._inspection_overrides = prior
        result = rows(
            self.tools,
            table,
            {k: record[k] for k in KEYS[table]},
            baseline=view == "baseline",
            limit=2,
            snapshot_date=self.day,
        )
        check(len(result) <= 1, "Ambiguous Search Console key", record)
        if table == "performance_rows" and result:
            row = result[0]
            page = rows(
                self.tools,
                "pages",
                dict(property_id=row["property_id"], page_id=row["page_id"]),
                limit=2,
                snapshot_date=self.day,
            )
            check(
                len(page) == 1 and page[0]["created_date"] <= row["date"],
                "Performance references an unavailable page",
            )
        if table == "pages" and result and result[0].get("last_crawl_time"):
            self.observed(result[0]["last_crawl_time"], "last_crawl_time")
        return result[0] if result else None

    def _check_url(self, url):
        normalized = SearchConsoleTools._normalize_url(url)
        property_id = self.tools._site_url(None)
        if property_id.startswith(
            ("http://", "https://")
        ) and not normalized.startswith(property_id):
            raise SearchConsoleToolError(
                "permission_denied", "URL is outside the configured URL-prefix property"
            )
        self.tools._ensure("pages")
        pages = self.tools._execute_rows(
            "SELECT created_date FROM pages WHERE property_id=? AND url=? LIMIT 2",
            [property_id, normalized],
        )
        if pages and str(pages[0]["created_date"]) > self.day:
            raise SearchConsoleToolError(
                "state_after_snapshot", "URL was created after the task snapshot"
            )

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False,
                error="unknown_tool",
                message="Unknown or disallowed Search Console tool",
            )
        schema = next(
            d["parameters"] for d in self.tool_definitions if d["name"] == name
        )
        if not valid(schema, arguments):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Search Console arguments",
            )
        try:
            self.open()
            if name in (
                "search_console_query_performance",
                "search_console_list_queries",
            ):
                self._validate_delta("performance_rows")
            elif name == "search_console_inspect_url":
                self._validate_delta(
                    "inspection_overrides",
                    dict(
                        table="inspection_overrides",
                        property_id=self.settings["property_id"],
                        inspection_url=SearchConsoleTools._normalize_url(
                            arguments["inspection_url"]
                        ),
                    ),
                )
            if name.startswith("search_console_list_"):
                result = catalog(
                    self.tools,
                    name.removeprefix("search_console_list_"),
                    snapshot_date=self.day,
                    **arguments,
                )
                if name == "search_console_list_pages":
                    for row in result["records"]:
                        if row.get("last_crawl_time"):
                            self.observed(row["last_crawl_time"], "last_crawl_time")
            elif name == "search_console_get_record":
                result = dict(ok=True, record=self.lookup(arguments["record"]))
            elif name == "search_console_inspect_url":
                self._check_url(arguments["inspection_url"])
                result = clean(self.tools.search_console_inspect_url(**arguments))
                self._inspection_time(
                    result["inspectionResult"],
                    (
                        self.observed(result["observedAt"], "observedAt")
                        if result["observedAt"]
                        else None
                    ),
                )
            else:
                check(
                    _parse_date(arguments["end_date"], "end_date").isoformat()
                    <= self.day,
                    "Performance report ends after snapshot",
                )
                result = self.tools.search_console_query_performance(**arguments)
            if len(_encoded(result)) > self.limit:
                return dict(
                    ok=False,
                    error="result_limit",
                    message="Search Console result exceeds max_result_bytes. Use a smaller page or narrower report.",
                )
            return result
        except SearchConsoleToolError as error:
            return error.as_dict()

    def close(self):
        if self.tools is not None:
            tools, self.tools = self.tools, None
            tools.close()
