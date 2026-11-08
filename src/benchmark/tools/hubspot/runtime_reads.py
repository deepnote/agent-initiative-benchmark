"""Bounded native CRM reads and row overlays for the benchmark runtime.

No metric calculations live here. The optional native metrics catalog owns
definitions. Observations are the same stored property-history rows as detail.
"""

from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime

from benchmark.prompts import load_text


from .hubspot import (
    HubSpotTools,
    HubSpotToolError,
    OBJECT_TYPES,
    CORE_COLUMNS,
    OVERLAY_SQL,
    _new_database,
    _write_rows,
    _finish_database,
)


def message(name):
    return json.loads(load_text("tools/hubspot/messages.json"))[name]


TABLES = {
    **{name: CORE_COLUMNS for name in OBJECT_TYPES},
    "property_values": (
        "object_type",
        "record_id",
        "property_name",
        "value",
        "updated_at",
    ),
    "associations": (
        "id",
        "from_object_type",
        "from_record_id",
        "to_object_type",
        "to_record_id",
        "association_type",
        "created_at",
    ),
    "history": (
        "id",
        "object_type",
        "record_id",
        "property_name",
        "value",
        "timestamp",
        "source_type",
        "source_id",
    ),
    "activities": (
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
    "activity_associations": (
        "activity_id",
        "object_type",
        "record_id",
        "association_type",
    ),
}
KEYS = {name: ("id",) for name in TABLES}
KEYS.update(
    property_values=("object_type", "record_id", "property_name"),
    activity_associations=(
        "activity_id",
        "object_type",
        "record_id",
        "association_type",
    ),
)
CATALOGS = {
    "property_definitions": ("object_type", "name"),
    "owners": ("id",),
    "teams": ("id",),
    "pipelines": ("object_type", "pipeline_id"),
    "pipeline_stages": ("pipeline_id", "stage_id"),
}


def build_runtime_overlay(rows, path):
    """Complete native rows, using the existing schema and database writer."""
    db = _new_database(path, OVERLAY_SQL)
    try:
        for table, columns in TABLES.items():
            _write_rows(
                db,
                table,
                (*columns, "operation"),
                [dict(row, operation="upsert") for row in rows.get(table, ())],
            )
        _finish_database(db, path)
    except BaseException:
        db.close()
        raise


COMPACT_PROPERTIES = {
    "companies": ("name", "domain"),
    "contacts": ("firstname", "lastname", "email"),
    "deals": ("dealname", "dealstage", "amount"),
}


class RuntimeHubSpotTools(HubSpotTools):
    def _cursor_scope(self, scope):
        return hashlib.sha256(
            json.dumps(
                [
                    scope,
                    self.config.agent_owner_id,
                    sorted(self.config.visible_team_ids),
                    self.config.allow_restricted_properties,
                    getattr(self, "source_identity", None),
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _encode_cursor(self, offset, scope):
        return super()._encode_cursor(offset, self._cursor_scope(scope))

    def _decode_cursor(self, after, scope):
        return super()._decode_cursor(after, self._cursor_scope(scope))

    def _selected_properties(self, object_type, properties, **kwargs):
        if properties is None:
            properties = [
                name
                for name in COMPACT_PROPERTIES[object_type]
                if (object_type, name) in self._definitions
                and (
                    self.config.allow_restricted_properties
                    or self._definitions[object_type, name]["access_level"]
                    == "standard"
                )
            ]
        return super()._selected_properties(object_type, properties, **kwargs)

    def _property_definition(self, object_type, name):
        try:
            return super()._property_definition(object_type, name)
        except HubSpotToolError as error:
            if error.code == "invalid_property" and isinstance(name, str) and name:
                raise HubSpotToolError(
                    "invalid_property",
                    message("unknown_property").format(property_name=name),
                ) from None
            raise

    def _filters(self, object_type, filters):
        # Validate native scalar types without inventing enum choices or aliases.
        if isinstance(filters, (list, tuple)):
            for item in filters:
                if not isinstance(item, dict):
                    continue
                definition = self._property_definition(
                    object_type, item.get("property_name")
                )
                values = item.get("values", [item["value"]] if "value" in item else [])
                if isinstance(values, (list, tuple)):
                    for value in values:
                        valid = True
                        if definition["type"] == "number":
                            try:
                                valid = not isinstance(value, bool) and math.isfinite(
                                    float(value)
                                )
                            except (TypeError, ValueError, OverflowError):
                                valid = False
                        elif definition["type"] == "enumeration":
                            valid = isinstance(value, str)
                        elif definition["type"] == "bool":
                            valid = value in ("true", "false", "0", "1")
                        if not valid:
                            raise HubSpotToolError(
                                "invalid_input", message("invalid_filter_type")
                            )
        return super()._filters(object_type, filters)

    def hubspot_search_companies(self, query="", *, page_size=10, **kwargs):
        return super().hubspot_search_companies(query, page_size=page_size, **kwargs)

    def hubspot_search_contacts(self, query="", *, page_size=10, **kwargs):
        return super().hubspot_search_contacts(query, page_size=page_size, **kwargs)

    def hubspot_search_deals(self, query="", *, page_size=10, **kwargs):
        return super().hubspot_search_deals(query, page_size=page_size, **kwargs)

    def readable(self, object_type, record_id):
        kind = self._object_type(object_type)
        rid = self._record_id(record_id)
        record = self._record(kind, rid)
        if record is None or not self._can_read(record):
            raise HubSpotToolError("not_found", "CRM record is unavailable")
        return kind, rid

    def page(self, sql, params, scope, page_size, after):
        size = self._page_size(page_size)
        # Bind continuation to identity/grants as well as the complete query.
        scope = json.dumps(
            [
                scope,
                self.config.agent_owner_id,
                self.config.visible_team_ids,
                self.config.allow_restricted_properties,
            ],
            sort_keys=True,
        )
        offset = self._decode_cursor(after, scope)
        total = self._connection.execute(
            "SELECT COUNT(*) FROM (" + sql + ")", params
        ).fetchone()[0]
        rows = [
            dict(row)
            for row in self._connection.execute(
                sql + " LIMIT ? OFFSET ?", (*params, size, offset)
            )
        ]
        result = {"results": rows, "total": total}
        if offset + len(rows) < total:
            result["paging"] = {
                "next": {"after": self._encode_cursor(offset + len(rows), scope)}
            }
        return result

    def hubspot_list_metadata(
        self,
        catalog="property_definitions",
        object_type=None,
        page_size=50,
        after=None,
        property_name=None,
        archived=False,
        owner_id=None,
    ):
        if owner_id is not None and catalog != "owners":
            raise HubSpotToolError("invalid_input", message("owner_catalog"))
        if catalog == "property_values":
            return self._property_choices(
                object_type, property_name, archived, page_size, after
            )
        if property_name is not None:
            raise HubSpotToolError(
                "invalid_input", "property_name requires the property_values catalog"
            )
        if catalog not in CATALOGS:
            raise HubSpotToolError("invalid_input", "Unknown CRM catalog")
        where, args = [], []
        if object_type is not None:
            if catalog not in {"property_definitions", "pipelines"}:
                raise HubSpotToolError(
                    "invalid_input", "This catalog has no object_type filter"
                )
            where.append("object_type=?")
            args.append(self._object_type(object_type))
        if (
            catalog == "property_definitions"
            and not self.config.allow_restricted_properties
        ):
            where.append("access_level='standard'")
        if catalog in {"teams", "owners"}:
            where.append(
                ("id" if catalog == "teams" else "team_id")
                + " IN ("
                + ",".join("?" for _ in self.config.visible_team_ids)
                + ")"
            )
            args.extend(self.config.visible_team_ids)
        if owner_id is not None:
            where.append("id=?")
            args.append(self._record_id(owner_id))
        sql = "SELECT * FROM baseline." + catalog
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + ",".join(CATALOGS[catalog])
        scope = [catalog, object_type]
        if owner_id is not None:
            scope.append(owner_id)
        return self.page(sql, args, scope, page_size, after)

    def _property_choices(self, object_type, property_name, archived, page_size, after):
        kind = self._object_type(object_type)
        definition = self._property_definition(kind, property_name)
        scope = ["property_values", kind, property_name, archived]
        # Some native schemas supply choices. The pinned schema does not.
        options = definition.get("options")
        if options is not None:
            try:
                choices = json.loads(options) if isinstance(options, str) else options
                if not isinstance(choices, list) or len(choices) > 10000:
                    raise ValueError()
                rows = [
                    (
                        dict(
                            value=v["value"],
                            **({"label": v["label"]} if "label" in v else {}),
                        )
                        if isinstance(v, dict)
                        else dict(value=v)
                    )
                    for v in choices
                ]
            except (ValueError, KeyError, TypeError):
                raise HubSpotToolError(
                    "invalid_input", message("invalid_options")
                ) from None
            if rows:
                size = self._page_size(page_size)
                offset = self._decode_cursor(after, json.dumps(scope))
                result = dict(
                    results=rows[offset : offset + size],
                    total=len(rows),
                    origin="authoritative_options",
                    coverage=dict(allowed_schema=True, observations=False),
                )
                if offset + size < len(rows):
                    result["paging"] = dict(
                        next=dict(
                            after=self._encode_cursor(offset + size, json.dumps(scope))
                        )
                    )
                return result
        teams = ",".join("?" for _ in self.config.visible_team_ids)
        sql = (
            "SELECT p.value,COUNT(*) record_count FROM effective_property_values p "
            "JOIN effective_"
            + kind
            + " r ON r.id=p.record_id WHERE p.object_type=? AND p.property_name=? "
            "AND r.access_team_id IN ("
            + teams
            + ") AND r.archived=? GROUP BY p.value ORDER BY p.value"
        )
        result = self.page(
            sql,
            [kind, property_name, *self.config.visible_team_ids, int(archived)],
            scope,
            page_size,
            after,
        )
        result.update(
            origin="observed_values",
            coverage=dict(
                allowed_schema=False,
                records="readable_effective",
                archived=archived,
                missing_properties="not_counted",
            ),
        )
        return result

    def hubspot_list_metrics(self, object_type=None, page_size=50, after=None):
        try:
            from .metrics import metric_definitions
        except ImportError:
            return {
                "available": False,
                "results": [],
                "reason": "Native metric catalog is unavailable in this checkout",
            }
        # Native catalog is small. Its function consumes the dataset catalog and
        # supplies meanings/units, including company versus contact CSAT.
        definitions = [
            r
            for r in self._definitions.values()
            if (
                object_type is None
                or r["object_type"] == self._object_type(object_type)
            )
            and (
                r["access_level"] == "standard"
                or self.config.allow_restricted_properties
            )
        ]
        catalog = metric_definitions(definitions)
        size = self._page_size(page_size)
        scope = json.dumps(["metrics", object_type, catalog], sort_keys=True)
        offset = self._decode_cursor(after, scope)
        result = {
            "available": True,
            "results": catalog[offset : offset + size],
            "total": len(catalog),
        }
        if offset + size < len(catalog):
            result["paging"] = {
                "next": {"after": self._encode_cursor(offset + size, scope)}
            }
        return result

    def hubspot_read_history(
        self,
        object_type,
        record_id,
        property_name,
        start_time=None,
        end_time=None,
        page_size=50,
        after=None,
        history_id=None,
        order="asc",
    ):
        kind, rid = self.readable(object_type, record_id)
        self._property_definition(kind, property_name)
        where, args = ["object_type=?", "record_id=?", "property_name=?"], [
            kind,
            rid,
            property_name,
        ]
        instants = []
        for key, value, operator in (
            ("start", start_time, ">="),
            ("end", end_time, "<="),
        ):
            if value is not None:
                try:
                    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if instant.tzinfo is None:
                        raise ValueError()
                except (ValueError, AttributeError):
                    raise HubSpotToolError(
                        "invalid_input",
                        "History dates require timestamps with time zones",
                    ) from None
                instants.append(instant)
                where.append("julianday(timestamp)" + operator + "julianday(?)")
                args.append(value)
        if len(instants) == 2 and instants[0] > instants[1]:
            raise HubSpotToolError("invalid_input", "History start is after end")
        if history_id is not None:
            where.append("id=?")
            args.append(self._record_id(history_id))
        direction = self._order(order)
        sql = (
            "SELECT * FROM effective_history WHERE "
            + " AND ".join(where)
            + " ORDER BY julianday(timestamp) "
            + direction
            + ",id "
            + direction
        )
        return self.page(
            sql,
            args,
            [
                "history",
                kind,
                rid,
                property_name,
                start_time,
                end_time,
                history_id,
                order,
            ],
            page_size,
            after,
        )

    @staticmethod
    def _order(order):
        if order not in {"asc", "desc"}:
            raise HubSpotToolError("invalid_input", message("invalid_order"))
        return order.upper()

    def hubspot_list_activities(
        self,
        object_type,
        record_id,
        page_size=20,
        after=None,
        activity_id=None,
        start_time=None,
        end_time=None,
        order="desc",
        query="",
        activity_type=None,
        detail=None,
        body_chars=800,
    ):
        kind, rid = self.readable(object_type, record_id)
        detail = (
            ("full" if activity_id is not None else "compact")
            if detail is None
            else detail
        )
        if (
            not isinstance(detail, str)
            or detail not in {"compact", "full"}
            or type(body_chars) is not int
            or not 0 <= body_chars <= 4000
            or not isinstance(query, str)
            or len(query) > 200
        ):
            raise HubSpotToolError(
                "invalid_input", message("invalid_activity_projection")
            )
        sql = (
            "SELECT a.* FROM effective_activities a WHERE EXISTS (SELECT 1 FROM effective_activity_associations aa "
            "WHERE aa.activity_id=a.id AND aa.object_type=? AND aa.record_id=?)"
        )
        args = [kind, rid]
        if activity_id is not None:
            sql += " AND a.id=?"
            args.append(self._record_id(activity_id))
        if query:
            sql += " AND instr(lower(coalesce(a.title,'') || ' ' || coalesce(a.body,'')),lower(?))>0"
            args.append(query)
        if activity_type is not None:
            sql += " AND a.activity_type=?"
            args.append(self._record_id(activity_type))
        instants = []
        for value, operator in ((start_time, ">="), (end_time, "<=")):
            if value is None:
                continue
            try:
                instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if instant.tzinfo is None:
                    raise ValueError()
            except (ValueError, AttributeError):
                raise HubSpotToolError(
                    "invalid_input", "History dates require timestamps with time zones"
                ) from None
            instants.append(instant)
            sql += " AND julianday(a.occurred_at)" + operator + "julianday(?)"
            args.append(value)
        if len(instants) == 2 and instants[0] > instants[1]:
            raise HubSpotToolError("invalid_input", "History start is after end")
        direction = self._order(order)
        sql += " ORDER BY julianday(a.occurred_at) " + direction + ",a.id " + direction
        result = self.page(
            sql,
            args,
            [
                "activities-v2",
                kind,
                rid,
                activity_id,
                start_time,
                end_time,
                order,
                query,
                activity_type,
                detail,
                body_chars,
            ],
            page_size,
            after,
        )
        if detail == "compact":
            fields = (
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
                "source_type",
                "source_id",
            )
            result["results"] = [
                dict(
                    {key: row[key] for key in fields},
                    body=row["body"][:body_chars],
                    body_truncated=len(row["body"]) > body_chars,
                )
                for row in result["results"]
            ]
            result["omitted_fields"] = ["metadata_json", "created_at", "updated_at"]
            result["detail"] = "compact"
        return result

    def hubspot_list_associations(
        self,
        object_type,
        record_id,
        page_size=50,
        after=None,
        association_id=None,
        object_types=None,
    ):
        kind, rid = self.readable(object_type, record_id)

        # Filter both ends before counts and pagination. Unreadable associations
        # cannot hide readable ones beyond the old detail cap.
        def visible_end(side):
            return (
                "("
                + " OR ".join(
                    "(a."
                    + side
                    + "_object_type='"
                    + t
                    + "' AND EXISTS (SELECT 1 FROM effective_"
                    + t
                    + " r WHERE r.id=a."
                    + side
                    + "_record_id AND r.access_team_id IN ("
                    + ",".join("?" for _ in self.config.visible_team_ids)
                    + ")))"
                    for t in OBJECT_TYPES
                )
                + ")"
            )

        sql = (
            "SELECT a.* FROM effective_associations a WHERE "
            "((a.from_object_type=? AND a.from_record_id=?) OR (a.to_object_type=? AND a.to_record_id=?)) "
            "AND " + visible_end("from") + " AND " + visible_end("to")
        )
        args = [kind, rid, kind, rid, *self.config.visible_team_ids * 6]
        if association_id is not None:
            sql += " AND a.id=?"
            args.append(self._record_id(association_id))
        if object_types is not None:
            targets = self._association_types(kind, object_types)
            if not targets:
                sql += " AND 0"
            else:
                placeholders = ",".join("?" for _ in targets)
                sql += (
                    " AND CASE WHEN a.from_object_type=? AND a.from_record_id=? THEN a.to_object_type "
                    "ELSE a.from_object_type END IN (" + placeholders + ")"
                )
                args.extend([kind, rid, *targets])
        sql += " ORDER BY julianday(a.created_at),a.id"
        return self.page(
            sql,
            args,
            ["associations", kind, rid, association_id, object_types],
            page_size,
            after,
        )

    def hubspot_get_record(
        self,
        object_type,
        record_id,
        *,
        properties=None,
        associations=None,
        properties_with_history=None,
        archived=False,
    ):
        kind, rid = self.readable(object_type, record_id)
        record = self._record(kind, rid)
        if record["archived"] and not archived:
            raise HubSpotToolError("not_found", "CRM record is unavailable")
        result = self._record_result(
            kind, record, self._selected_properties(kind, properties)
        )
        result["omittedRelated"] = ["activities", "associations", "property_history"]
        if associations:
            targets = self._association_types(kind, associations)
            # Explicit compatibility reads are bounded native pages, never full payloads.
            page = self.hubspot_list_associations(
                kind, rid, page_size=20, object_types=targets
            )
            result["associationPage"] = page
            result["associationObjectTypes"] = list(targets)
        if properties_with_history:
            selected = self._selected_properties(
                kind, properties_with_history, allow_empty=True
            )
            result["historyPages"] = {
                name: dict(
                    self.hubspot_read_history(
                        kind, rid, name, page_size=20, order="desc"
                    ),
                    order="desc",
                )
                for name in selected
            }
        return result
