"""Paged native catalogs and exact privacy-safe Search Console records."""

from datetime import date, datetime
from .search_console import SearchConsoleToolError

KEYS = {
    "properties": ("id",),
    "pages": ("property_id", "page_id"),
    "queries": ("property_id", "query_id"),
    "performance_rows": ("property_id", "source_event_id"),
    "inspection_overrides": ("property_id", "inspection_url"),
}
PERFORMANCE_COLUMNS = (
    "source_event_id",
    "property_id",
    "date",
    "query_id",
    "page_id",
    "country_code",
    "device",
    "search_type",
    "search_appearance",
    "clicks",
    "impressions",
    "position_sum",
)


def clean(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "source"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def rows(
    tools,
    table,
    key=None,
    *,
    baseline=False,
    limit=100,
    start_row=0,
    snapshot_date=None
):
    key = key or {}
    property_id = tools.config.property_id
    if key.get("property_id", property_id) != property_id:
        raise SearchConsoleToolError(
            "permission_denied", "Property is outside the configured grant"
        )
    if table == "properties":
        return (
            [clean(tools._property)]
            if start_row == 0 and key.get("id", property_id) == property_id
            else []
        )
    tools._ensure("pages")
    params = [property_id]
    where = []
    if table == "pages":
        source, prefix = "pages r", "r"
        select = "r.* EXCLUDE (inspection_bucket)"
        where = ["r.property_id=?"]
        if snapshot_date:
            where.append("r.created_date<=?")
            params.append(snapshot_date)
        order = "r.page_id"
    elif table in ("queries", "performance_rows"):
        tools._ensure("queries")
        tools._performance_view()
        perf = "baseline_performance" if baseline else "effective_performance"
        if table == "queries":
            source, prefix, order = "queries r", "r", "r.query_id"
            select = "?::VARCHAR AS property_id, r.*"
            params = [property_id, property_id]
            where = [
                "NOT r.privacy_hidden",
                "EXISTS (SELECT 1 FROM "
                + perf
                + " p WHERE p.query_id=r.query_id AND p.property_id=?"
                + (" AND p.date<=?" if snapshot_date else "")
                + ")",
            ]
            if snapshot_date:
                params.append(snapshot_date)
        else:
            source, prefix, order = (
                perf
                + " r JOIN queries q ON q.query_id=r.query_id JOIN pages pg ON pg.page_id=r.page_id AND pg.property_id=r.property_id",
                "r",
                "r.source_event_id",
            )
            select = (
                ",".join("r." + c for c in PERFORMANCE_COLUMNS)
                + ",pg.url AS page_url,q.query_text"
            )
            where = ["r.property_id=?", "NOT q.privacy_hidden"]
            if snapshot_date:
                where.append("r.date<=?")
                params.append(snapshot_date)
    else:
        raise SearchConsoleToolError("invalid_argument", "Unknown native catalog")
    for field, value in key.items():
        if field == "property_id":
            continue
        if field not in KEYS[table]:
            raise SearchConsoleToolError("invalid_argument", "Unknown native key")
        where.append(prefix + "." + field + "=?")
        params.append(value)
    return clean(
        tools._execute_rows(
            "SELECT "
            + select
            + " FROM "
            + source
            + " WHERE "
            + " AND ".join(where)
            + " ORDER BY "
            + order
            + " LIMIT ? OFFSET ?",
            [*params, limit, start_row],
        )
    )


def catalog(
    tools, table, *, row_limit=100, start_row=0, site_url=None, snapshot_date=None
):
    tools._site_url(site_url)
    if (
        type(row_limit) is not int
        or not 1 <= row_limit <= 1000
        or type(start_row) is not int
        or start_row < 0
    ):
        raise SearchConsoleToolError(
            "invalid_argument", "Use row_limit 1..1000 and a nonnegative start_row"
        )
    result = rows(
        tools,
        table,
        limit=row_limit + 1,
        start_row=start_row,
        snapshot_date=snapshot_date,
    )
    more = len(result) > row_limit
    return dict(
        ok=True,
        siteUrl=tools.config.property_id,
        records=result[:row_limit],
        startRow=start_row,
        rowLimit=row_limit,
        hasMore=more,
        nextStartRow=start_row + row_limit if more else None,
    )
