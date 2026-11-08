"""Versioned native deltas and lazy, exact stored Analytics reads."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import v2_schema as v2

V2_SCHEMAS = dict(
    sessions=v2.AGGREGATE_SESSION_SCHEMA,
    events=v2.AGGREGATE_EVENT_SCHEMA,
    realtime=v2.REALTIME_SCHEMA,
    cohorts=v2.COHORT_SCHEMA,
    attribution=v2.ATTRIBUTION_SCHEMA,
)
CATALOG_SCHEMAS = dict(
    websites=v2.WEBSITE_SCHEMA,
    pages=v2.PAGE_SCHEMA,
    campaigns=v2.CAMPAIGN_SCHEMA,
    sample_sessions=v2.SAMPLE_SESSION_SCHEMA,
    sample_events=v2.SAMPLE_EVENT_SCHEMA,
)
V2_KEYS = dict(
    sessions=("row_id",),
    events=("row_id",),
    realtime=("source_record_id",),
    cohorts=("source_record_id",),
    attribution=("path_id", "attribution_model", "touch_position"),
    websites=("website_id",),
    pages=("page_id",),
    campaigns=("campaign_id",),
    sample_sessions=("session_id",),
    sample_events=("event_id",),
)
# These counts have no cross-bucket user identities. A sum cannot establish uniqueness.
USER_METRICS = frozenset(
    (
        "activeUsers",
        "totalUsers",
        "newUsers",
        "returningUsers",
        "averageEngagementTimePerActiveUser",
        "engagedSessionsPerActiveUser",
        "sessionsPerActiveUser",
        "screenPageViewsPerActiveUser",
        "newUserRate",
        "returningUserRate",
    )
)
HARNESS_FIELDS = frozenset(
    (
        "scenario_id",
        "generated_at",
        "generator_version",
        "world_seed",
        "_task_origin",
        "_benchmark_origin",
        "harness_origin",
        "generatorVersion",
        "buildId",
    )
)


def schemas(version):
    if version == 2:
        return {**V2_SCHEMAS, **CATALOG_SCHEMAS}
    from .analytics import SESSION_SCHEMA, EVENT_SCHEMA

    return dict(sessions=SESSION_SCHEMA, events=EVENT_SCHEMA)


def delta_schemas(version):
    allowed = V2_SCHEMAS if version == 2 else schemas(1)
    return {
        t: tuple((k, v) for k, v in schema if k not in HARNESS_FIELDS)
        for t, schema in allowed.items()
    }


def keys(version):
    return (
        V2_KEYS
        if version == 2
        else dict(sessions=("session_id",), events=("event_id",))
    )


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in HARNESS_FIELDS}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def build_delta(seed, destination):
    """Write complete v2 rows only. Task inventory owns add versus replace."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Analytics delta destination already exists")
    with closing(sqlite3.connect(destination)) as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES ('format_version', '2')")
        for table, schema in V2_SCHEMAS.items():
            columns = ",".join(
                '"'
                + k
                + '" '
                + (
                    "REAL"
                    if t == "DOUBLE"
                    else "INTEGER" if t in ("INTEGER", "BIGINT", "BOOLEAN") else "TEXT"
                )
                for k, t in schema
            )
            key = ",".join('"' + k + '"' for k in keys(2)[table])
            db.execute(f'CREATE TABLE "{table}" ({columns}, PRIMARY KEY ({key}))')
            rows = seed.get(table, [])
            if rows:
                db.executemany(
                    f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in schema)})',
                    [tuple(row.get(k) for k, _ in schema) for row in rows],
                )
        db.commit()


def cutoff(tools):
    value = tools.config.snapshot_time
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            ZoneInfo(tools._property["timeZone"])
        )
        if value
        else None
    )


def observed_where(tools, table, alias="r"):
    """Native timestamps are property-local wall time in the saved v2 schemas."""
    schema = dict(schemas(2 if tools._aggregate_mode else 1)[table])
    parts, params = [], []

    def add(sql, value):
        parts.append(sql)
        params.append(value)

    if "property_id" in schema:
        add(f"{alias}.property_id=?", tools._property["id"])
    if tools.config.snapshot_time and "is_privacy_suppressed" in schema:
        parts.append(f"NOT coalesce({alias}.is_privacy_suppressed, true)")
    now = cutoff(tools)
    if now:
        local = now.replace(tzinfo=None)
        if table in ("sessions", "events"):
            if tools._aggregate_mode:
                # Exclude an unfinished hour rather than reveal later observations.
                add(
                    f"CAST({alias}.date AS TIMESTAMP) + {alias}.hour * INTERVAL 1 HOUR + INTERVAL 1 HOUR <= ?",
                    local,
                )
            elif table == "events":
                add(f"TRY_CAST({alias}.event_timestamp AS TIMESTAMPTZ)<=?", now)
            else:
                add(f"{alias}.date<=?", now.date())
        elif table == "realtime":
            add(f"{alias}.minute_timestamp + INTERVAL 1 MINUTE<=?", local)
        elif table == "cohorts":
            add(
                f"{alias}.cohort_start_date + {alias}.activity_week * INTERVAL 7 DAY<=?",
                local,
            )
        elif table == "attribution":
            add(f"{alias}.touch_timestamp<=?", local)
            # Conversion time is not stored. Only completed conversion dates are safe.
            add(f"{alias}.key_event_date<?", now.date())
        elif table == "pages":
            add(f"{alias}.created_at<=?", now.date())
        elif table in ("sample_sessions", "sample_events"):
            field = (
                "session_end_timestamp"
                if table == "sample_sessions"
                else "event_timestamp"
            )
            add(f"{alias}.{field}<=?", local)
    return " AND ".join(parts) or "true", params


def ensure(tools, table):
    from .analytics import AnalyticsToolError, _sql_path

    if table in tools._loaded_families:
        return
    version = 2 if tools._aggregate_mode else 1
    if table not in schemas(version):
        raise AnalyticsToolError(
            "not_available", "Stored Analytics family is unavailable: " + table
        )
    if tools._acquire_family:
        tools._acquire_family(table)
    family = (
        ("samples/" + table.removeprefix("sample_"))
        if table.startswith("sample_")
        else table
    )
    files = sorted((tools.config.baseline_dir / family).rglob("*.parquet"))
    direct = tools.config.baseline_dir / (table + ".parquet")
    if direct.is_file():
        files.append(direct)
    if not files:
        raise AnalyticsToolError(
            "not_available", "Stored Analytics family is unavailable: " + table
        )
    schema = schemas(version)[table]
    columns = ",".join('"' + k + '"' for k, _ in schema)
    paths = "[" + ",".join("'" + _sql_path(p) + "'" for p in files) + "]"
    tools._duck.execute(
        f"CREATE VIEW native_baseline_{table} AS SELECT {columns} FROM read_parquet({paths}, union_by_name=true, hive_partitioning=false)"
    )
    overlay = "native_overlay_" + table
    tools._duck.execute(
        f"CREATE TABLE {overlay} ("
        + ",".join('"' + k + '" ' + t for k, t in schema)
        + ")"
    )
    path = tools.config.overlay_dir / "overlay.sqlite"
    if tools._aggregate_mode and path.is_file():
        with closing(
            sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        ) as db:
            version_row = db.execute(
                "SELECT value FROM metadata WHERE key='format_version'"
            ).fetchone()
            if version_row != ("2",):
                raise AnalyticsToolError(
                    "invalid_delta", "v2 Analytics requires a native v2 delta"
                )
            rows = (
                db.execute(f'SELECT {columns} FROM "{table}"').fetchall()
                if table in V2_SCHEMAS
                else []
            )
            if rows:
                tools._duck.executemany(
                    f'INSERT INTO {overlay} VALUES ({",".join("?" for _ in schema)})',
                    rows,
                )
    predicate = " AND ".join(f'o."{k}"=b."{k}"' for k in keys(version)[table])
    tools._duck.execute(
        f"CREATE VIEW native_effective_{table} AS SELECT * FROM {overlay} UNION ALL SELECT b.* FROM native_baseline_{table} b WHERE NOT EXISTS (SELECT 1 FROM {overlay} o WHERE {predicate})"
    )
    if tools._aggregate_mode and table in ("sessions", "events"):
        tools._duck.execute(
            f"CREATE VIEW aggregate_{table} AS SELECT * FROM native_effective_{table}"
        )
    tools._loaded_families.add(table)


def metadata(tools, property_id):
    from .analytics import (
        DIMENSION_SQL,
        METRIC_SQL,
        AGGREGATE_SESSION_DIMENSION_SQL,
        AGGREGATE_SESSION_METRIC_SQL,
        AGGREGATE_EVENT_DIMENSION_SQL,
        AGGREGATE_EVENT_METRIC_SQL,
    )

    tools._check_access(property_id)
    contracts = (
        dict(
            session=(AGGREGATE_SESSION_DIMENSION_SQL, AGGREGATE_SESSION_METRIC_SQL),
            event=(AGGREGATE_EVENT_DIMENSION_SQL, AGGREGATE_EVENT_METRIC_SQL),
        )
        if tools._aggregate_mode
        else dict(event=(DIMENSION_SQL, METRIC_SQL))
    )
    return public(
        dict(
            ok=True,
            kind="analytics#metadata",
            property=tools._property,
            schema_version=2 if tools._aggregate_mode else 1,
            dimensions=[v for k, v in tools._dimensions.items() if k != "scenarioId"],
            metrics=list(tools._metrics.values()),
            report_grains={
                k: dict(
                    dimensions=[
                        d for d in dims if d in tools._dimensions and d != "scenarioId"
                    ],
                    metrics=[
                        m
                        for m in mets
                        if m in tools._metrics and m != "userKeyEventRate"
                    ],
                )
                for k, (dims, mets) in contracts.items()
            },
            stored_schemas={
                t: {k: v for k, v in s if k not in HARNESS_FIELDS}
                for t, s in schemas(2 if tools._aggregate_mode else 1).items()
            },
            evidence_keys=keys(2 if tools._aggregate_mode else 1),
            non_additive_user_metrics=sorted(USER_METRICS),
            snapshot_time=tools.config.snapshot_time,
        )
    )


def records(
    tools,
    property_id,
    table,
    *,
    filters=None,
    start_date=None,
    end_date=None,
    page_size=100,
    start_cursor=None,
    view="effective",
):
    from .analytics import AnalyticsToolError, _json

    tools._check_access(property_id)
    version = 2 if tools._aggregate_mode else 1
    if table not in schemas(version) or view not in ("baseline", "effective"):
        raise AnalyticsToolError("invalid_argument", "Unknown Analytics table or view")
    if (
        table == "cohorts"
        and cutoff(tools)
        and tools._property.get("dataAsOfTimestamp")
    ):
        saved = datetime.fromisoformat(tools._property["dataAsOfTimestamp"]).replace(
            tzinfo=None
        )
        if saved > cutoff(tools).replace(tzinfo=None):
            raise AnalyticsToolError(
                "not_available",
                "Stored cohort observations are after the task snapshot",
            )
    if not tools._aggregate_mode:
        tools._ensure_report()
        relation = ("baseline_" if view == "baseline" else "effective_") + table
    else:
        ensure(tools, table)
        relation = "native_" + view + "_" + table
    where, params = observed_where(tools, table)
    fields = dict(schemas(version)[table])
    if filters is None:
        filters = {}
    if (
        not isinstance(filters, dict)
        or len(filters) > 20
        or set(filters) - fields.keys()
        or set(filters) & HARNESS_FIELDS
    ):
        raise AnalyticsToolError(
            "invalid_argument", "Use stored business fields for equality filters"
        )
    for field, value in filters.items():
        if isinstance(value, (dict, list)):
            raise AnalyticsToolError(
                "invalid_argument", "Equality filters require scalar values"
            )
        where += f' AND r."{field}" IS NOT DISTINCT FROM ?'
        params.append(value)
    date_field = {
        "sessions": "date",
        "events": "date",
        "realtime": "minute_timestamp",
        "cohorts": "cohort_start_date",
        "attribution": "key_event_date",
        "sample_sessions": "date",
        "sample_events": "date",
    }.get(table)
    if start_date or end_date:
        if not date_field:
            raise AnalyticsToolError(
                "invalid_argument", "Date filters are unavailable for this catalog"
            )
        for value, op in ((start_date, ">="), (end_date, "<=")):
            if value:
                parsed = tools._parse_date(value)
                where += f" AND CAST(r.{date_field} AS DATE){op}?"
                params.append(parsed)
    limit = tools._page_size(page_size)
    scope = hashlib.sha256(
        _json(
            [
                property_id,
                table,
                filters,
                start_date,
                end_date,
                view,
                tools.config.snapshot_time,
            ]
        ).encode()
    ).hexdigest()[:24]
    offset = tools._cursor_offset(start_cursor, scope)
    count = tools._query(
        f"SELECT count(*) AS n FROM {relation} r WHERE {where}", params
    )[0]["n"]
    order = ",".join('r."' + k + '"' for k in keys(version)[table])
    rows = tools._query(
        f"SELECT r.* FROM {relation} r WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    result = dict(
        ok=True,
        kind="analytics#storedRows",
        table=table,
        records=rows,
        rowCount=count,
        hasMore=offset + len(rows) < count,
        nextCursor=(
            tools._encode_cursor(offset + len(rows), scope)
            if offset + len(rows) < count
            else None
        ),
        metadata=dict(
            timeZone=tools._property["timeZone"],
            snapshotTime=tools.config.snapshot_time,
            exact=True,
            grain=table,
            sampled=table.startswith("sample_"),
        ),
    )
    if table == "realtime":
        bounds = tools._query(
            f"SELECT min(minute_timestamp) AS start, max(minute_timestamp) AS end FROM {relation} r WHERE "
            + observed_where(tools, table)[0],
            observed_where(tools, table)[1],
        )[0]
        now = cutoff(tools)
        result["metadata"].update(
            savedWindow=bounds,
            savedDataAsOf=tools._property.get("dataAsOfTimestamp"),
            ageSeconds=(
                (now.replace(tzinfo=None) - bounds["end"]).total_seconds()
                if now and bounds["end"]
                else None
            ),
            currentTraffic=False,
        )
    if table == "cohorts":
        result["metadata"].update(
            numerator="returning_user_count",
            denominator="cohort_user_count",
            periodUnit="week",
            periodAnchor="cohort_start_date + activity_week * 7 days",
            rateUnit="fraction",
        )
    if table == "attribution":
        result["metadata"].update(
            creditUnit="fractional key events", modelField="attribution_model"
        )
    return public(result)
