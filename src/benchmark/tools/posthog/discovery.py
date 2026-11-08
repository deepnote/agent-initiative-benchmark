"""Small native catalog and history reads. No inferred records or identity joins."""

from .posthog import PostHogToolError, _read_json

CATALOGS = (
    "event_definitions",
    "workspace_groups",
    "website_groups",
    "insights",
    "feature_flags",
)
KEYS = {
    "events": ("event_uuid",),
    "person_changes": ("person_id", "valid_from"),
    "person_history": ("person_id", "valid_from"),
    "distinct_ids": ("distinct_id",),
    "workspace_groups": ("id",),
    "website_groups": ("id",),
    "event_definitions": ("name",),
    "insights": ("id",),
    "feature_flags": ("id",),
}


def rows(tools, table, keys=None, baseline=False, limit=None, offset=0):
    keys = keys or {}
    family = "person_history" if table == "person_changes" else table
    tools._ensure(family)
    clauses, params = ["project_id=?"], [tools.config.project_id]
    if table in ("insights", "feature_flags", "event_definitions"):
        if table == "event_definitions":
            path = tools.config.baseline_dir / "event_definitions.json"
            if not path.is_file():
                raise PostHogToolError(
                    "not_available", "Event definitions are not stored in this source"
                )
            objects = _read_json(path)
        elif baseline:
            objects = _read_json(tools.config.baseline_dir / (table + ".json"))
        else:
            objects = list(getattr(tools, "_" + table).values())
        selected = [
            r
            for r in objects
            if r.get("project_id", tools.config.project_id) == tools.config.project_id
            and all(r.get(k) == v for k, v in keys.items())
        ]
        selected.sort(key=lambda r: tuple(str(r[k]) for k in KEYS[table]))
        return selected[offset : offset + limit] if limit else selected
    if table == "events":
        source = "baseline_events" if baseline else "effective_events"
    elif table in ("person_changes", "person_history"):
        source = "baseline_person_history" if baseline else "effective_person_history"
    elif table in ("distinct_ids", "workspace_groups", "website_groups"):
        source = (
            "read_parquet('"
            + tools._sql_path(tools.config.baseline_dir / (table + ".parquet"))
            + "')"
        )
    else:
        raise PostHogToolError("invalid_input", "Unknown native record type")
    for key, value in keys.items():
        if key not in KEYS[table] and key not in (
            "person_id",
            "workspace_id",
            "distinct_id",
        ):
            raise PostHogToolError("invalid_input", "Unsupported native record key")
        clauses.append('"' + key + '"=?')
        params.append(value)
    from .posthog import EVENT_COLUMNS

    selected = ",".join(EVENT_COLUMNS) if table == "events" else "*"
    query = "SELECT " + selected + " FROM " + source + " WHERE " + " AND ".join(clauses)
    query += " ORDER BY " + ",".join(KEYS[table]) + " LIMIT ? OFFSET ?"
    return tools._bounded_rows(
        query, [*params, limit or tools.config.max_output_rows, offset]
    )


def catalog(
    tools, kind, limit=None, start_cursor=None, projection="full", insight_id=None
):
    """Full is the default projection. Summary omits only insight results."""
    if kind not in CATALOGS:
        raise PostHogToolError("invalid_input", "Unknown catalog kind")
    if projection not in ("full", "summary"):
        raise PostHogToolError("invalid_input", "Unknown catalog projection")
    if insight_id is not None and (
        kind != "insights" or not isinstance(insight_id, str) or not insight_id
    ):
        raise PostHogToolError(
            "invalid_input", "insight_id requires an exact nonempty insight ID"
        )
    size = tools._page_size(limit)
    # Keep old unfiltered cursors byte-compatible. Projection does not change
    # selection or order, so it may change between pages. An ID filter may not.
    scope = tools._scope(
        "catalog",
        kind,
        tools.config.project_id,
        *([insight_id] if insight_id is not None else [])
    )
    offset = tools._cursor_offset(start_cursor, scope)
    page = rows(
        tools,
        kind,
        {"id": insight_id} if insight_id is not None else None,
        limit=size + 1,
        offset=offset,
    )
    more = len(page) > size
    result = dict(
        ok=True,
        object="catalog",
        kind=kind,
        records=page[:size],
        has_more=more,
        next_cursor=tools._encode_cursor(offset + size, scope) if more else None,
    )
    if projection == "summary" and kind == "insights":
        result["records"] = [
            {k: v for k, v in row.items() if k != "result"} for row in result["records"]
        ]
        result.update(
            projection="summary",
            omitted_fields=["result"],
            full_record_read=dict(
                tool="posthog_list_catalog",
                kind="insights",
                projection="full",
                id_argument="insight_id",
            ),
        )
    return result


def history(
    tools,
    kind,
    person_id=None,
    distinct_id=None,
    flag_id=None,
    limit=None,
    start_cursor=None,
):
    if kind not in ("person_history", "distinct_ids", "feature_flags"):
        raise PostHogToolError("invalid_input", "Unknown history kind")
    if (
        kind == "person_history"
        and (not person_id or distinct_id or flag_id)
        or kind == "distinct_ids"
        and (not (person_id or distinct_id) or flag_id)
        or kind == "feature_flags"
        and (not flag_id or person_id or distinct_id)
    ):
        raise PostHogToolError(
            "invalid_input", "History requires matching native identity filters"
        )
    size = tools._page_size(limit)
    scope = tools._scope("history", kind, person_id, distinct_id, flag_id)
    offset = tools._cursor_offset(start_cursor, scope)
    if kind == "feature_flags":
        flag = tools.posthog_get_feature_flag(flag_id)
        stored = flag.get("history")
        if stored is None:
            return dict(
                ok=True,
                object="history",
                kind=kind,
                records=[],
                available=False,
                has_more=False,
                next_cursor=None,
            )
        page = stored[offset : offset + size + 1]
    else:
        filters = {
            k: v
            for k, v in dict(person_id=person_id, distinct_id=distinct_id).items()
            if v
        }
        page = rows(tools, kind, keys=filters, limit=size + 1, offset=offset)
    more = len(page) > size
    return dict(
        ok=True,
        object="history",
        kind=kind,
        records=page[:size],
        available=True,
        has_more=more,
        next_cursor=tools._encode_cursor(offset + size, scope) if more else None,
    )
