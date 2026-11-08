"""Exact native insight calculations with disk-backed relational intermediates."""

from __future__ import annotations

from .posthog import PostHogToolError


def compute(tools, definition, start, end, filters):
    kind = definition["type"]
    if kind not in (
        "trends",
        "funnels",
        "retention",
        "stickiness",
        "lifecycle",
        "paths",
    ):
        raise PostHogToolError("unsupported_insight", "Unsupported insight type")
    breakdowns = definition.get("breakdowns") or []
    if breakdowns and kind != "trends":
        raise PostHogToolError(
            "unsupported_insight",
            "Only trends support breakdowns in this native analysis contract",
        )
    if len(breakdowns) > 1 or (
        breakdowns
        and breakdowns[0]
        not in {
            "workspace_id",
            "website_id",
            "person_id",
            "distinct_id",
            "session_id",
            "event_name",
            "event_family",
            "event_kind",
        }
    ):
        raise PostHogToolError(
            "unsupported_insight", "Unsupported breakdown definition"
        )
    if definition.get("interval", "day") not in ("day", "month"):
        raise PostHogToolError("unsupported_insight", "Unsupported insight interval")
    if definition.get("counting_unit", "events") != "events":
        raise PostHogToolError("unsupported_insight", "Unsupported counting unit")
    clauses = ["project_id=?", "timestamp>=?", "timestamp<?"]
    params = [tools.config.project_id, start, end]
    names = definition.get("event_names", [])
    if names:
        clauses.append("event_name IN (" + ",".join("?" for _ in names) + ")")
        params.extend(names)
    extra, values = tools._filters(filters, "properties_json")
    clauses.extend(extra)
    params.extend(values)
    # Only columns used by the calculation survive the filter. DuckDB owns all
    # large intermediates and spills under the connection's existing budgets.
    columns = {
        "event_uuid",
        "timestamp",
        "person_id",
        "distinct_id",
        "event_name",
        "session_id",
        *breakdowns,
    }
    tools._duck.execute("DROP TABLE IF EXISTS analysis_events")
    tools._duck.execute(
        "CREATE TEMP TABLE analysis_events AS SELECT strftime(timezone(?,timestamp::TIMESTAMPTZ),'%Y-%m-%d') AS analysis_day,"
        + ",".join(sorted(columns))
        + " FROM effective_events WHERE "
        + " AND ".join(clauses),
        [tools.project.get("timezone", "UTC"), *params],
    )
    try:
        if kind == "trends":
            length = 7 if definition.get("interval") == "month" else 10
            group = (
                "COALESCE(NULLIF(" + breakdowns[0] + ",''),'all')"
                if breakdowns
                else "'all'"
            )
            rows = tools._bounded_rows(
                f"SELECT substr(analysis_day,1,{length}) AS period, {group} AS breakdown, "
                "count(*) AS count FROM analysis_events GROUP BY 1,2 ORDER BY 1,2"
            )
            return dict(type=kind, series=rows, total=sum(row["count"] for row in rows))
        if kind == "paths":
            rows = tools._bounded_rows(
                "SELECT path,count(*) AS count FROM (SELECT session_id, "
                "string_agg(event_name,' > ' ORDER BY timestamp,event_uuid) AS path FROM analysis_events "
                "WHERE session_id IS NOT NULL AND session_id<>'' GROUP BY session_id) "
                "GROUP BY path ORDER BY count DESC,path LIMIT 20"
            )
            return dict(type=kind, paths=rows)
        if kind == "funnels":
            stages = definition.get("stages", [])
            if (
                not stages
                or len(stages) > 50
                or any(not isinstance(s, str) or not s for s in stages)
            ):
                raise PostHogToolError(
                    "unsupported_insight", "Funnel requires 1 to 50 ordered event names"
                )
            counts = []
            tools._duck.execute("DROP TABLE IF EXISTS analysis_steps")
            try:
                for index, stage in enumerate(stages):
                    previous = (
                        ""
                        if index == 0
                        else (
                            " JOIN analysis_steps p ON "
                            "coalesce(e.person_id,e.distinct_id)=p.actor AND "
                            "(e.timestamp>p.timestamp OR (e.timestamp=p.timestamp AND e.event_uuid>p.event_uuid))"
                        )
                    )
                    tools._duck.execute(
                        "CREATE TEMP TABLE analysis_next AS SELECT "
                        "coalesce(e.person_id,e.distinct_id) AS actor,e.timestamp,e.event_uuid "
                        "FROM analysis_events e" + previous + " WHERE e.event_name=? "
                        "QUALIFY row_number() OVER (PARTITION BY coalesce(e.person_id,e.distinct_id) "
                        "ORDER BY e.timestamp,e.event_uuid)=1",
                        [stage],
                    )
                    counts.append(
                        tools._duck.execute(
                            "SELECT count(*) FROM analysis_next"
                        ).fetchone()[0]
                    )
                    tools._duck.execute("DROP TABLE IF EXISTS analysis_steps")
                    tools._duck.execute(
                        "ALTER TABLE analysis_next RENAME TO analysis_steps"
                    )
                return dict(
                    type=kind,
                    steps=[dict(event=s, count=c) for s, c in zip(stages, counts)],
                    conversion_rate=(
                        round(counts[-1] / counts[0], 4) if counts[0] else 0.0
                    ),
                )
            finally:
                tools._duck.execute("DROP TABLE IF EXISTS analysis_next")
                tools._duck.execute("DROP TABLE IF EXISTS analysis_steps")
        if kind not in ("retention", "stickiness", "lifecycle"):
            raise PostHogToolError("unsupported_insight", "Unsupported insight type")
        active = (
            "WITH active AS (SELECT DISTINCT coalesce(person_id,distinct_id) AS actor, "
            "analysis_day AS day FROM analysis_events), "
            "firsts AS (SELECT actor,min(day) AS first_day FROM active GROUP BY actor), "
            "months AS (SELECT DISTINCT substr(day,1,7) AS period FROM active) "
        )
        if kind == "stickiness":
            rows = tools._bounded_rows(
                active + "SELECT active_days,count(*) AS persons FROM "
                "(SELECT actor,count(*) AS active_days FROM active GROUP BY actor) GROUP BY active_days ORDER BY active_days"
            )
            return dict(type=kind, distribution=rows)
        if kind == "lifecycle":
            rows = tools._bounded_rows(
                active + "SELECT m.period, "
                "(SELECT count(*) FROM firsts f WHERE substr(f.first_day,1,7)=m.period) AS new, "
                "(SELECT count(DISTINCT a.actor) FROM active a JOIN firsts f USING(actor) "
                "WHERE substr(a.day,1,7)=m.period AND substr(f.first_day,1,7)<m.period) AS returning "
                "FROM months m ORDER BY m.period"
            )
            return dict(type=kind, series=rows)
        rows = tools._bounded_rows(
            active + "SELECT m.period, "
            "(SELECT count(*) FROM firsts f WHERE substr(f.first_day,1,7)<=m.period) AS cohort, "
            "(SELECT count(DISTINCT a.actor) FROM active a WHERE substr(a.day,1,7)=m.period) AS retained "
            "FROM months m ORDER BY m.period"
        )
        for row in rows:
            row["rate"] = (
                round(row["retained"] / row["cohort"], 4) if row["cohort"] else 0.0
            )
        return dict(type=kind, series=rows)
    finally:
        tools._duck.execute("DROP TABLE analysis_events")
