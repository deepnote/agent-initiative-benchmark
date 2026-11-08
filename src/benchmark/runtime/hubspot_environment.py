"""Read-only HubSpot adapter in the shared worker. No source or cache owner."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

from benchmark.prompts import load_text
from benchmark.tools.hubspot.hubspot import (
    HubSpotConfig,
    HubSpotToolError,
    OBJECT_TYPES,
)
from benchmark.tools.hubspot.runtime_reads import (
    RuntimeHubSpotTools,
    TABLES,
    KEYS,
    CATALOGS,
    build_runtime_overlay,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/hubspot/seed.json"
DOCUMENT_COLUMNS = (
    "id",
    "object_type",
    "record_id",
    "property_name",
    "sha256",
    "bytes",
    "media_type",
    "created_at",
)
ALL_KEYS = {
    **KEYS,
    **CATALOGS,
    "documents": ("id", "object_type", "record_id", "property_name"),
}
NULLABLE = {"archived_at", "value", "owner_id", "outcome", "due_at"}


class HubSpotEnvironmentError(ValueError):
    def __init__(self, reason, code="invalid_hubspot_environment"):
        self.code, self.source, self.validation_reason = code, "hubspot", reason
        super().__init__(reason)


def check(value, reason, code="invalid_hubspot_environment"):
    if not value:
        raise HubSpotEnvironmentError(reason, code)


def settings(config):
    value = config.get("service_settings", {}).get("hubspot")
    if not value and not any(t.startswith("hubspot_") for t in config["tool_scope"]):
        return None
    required = {
        "agent_owner_id",
        "visible_team_ids",
        "allow_restricted_properties",
        "max_document_bytes",
        "max_document_pages",
        "max_page_text_bytes",
    }
    check(
        isinstance(value, dict) and set(value) == required,
        "HubSpot requires explicit pinned identity, team/property grants and document limits",
        "invalid_service_settings",
    )
    check(
        value["agent_owner_id"] is None
        or isinstance(value["agent_owner_id"], str)
        and value["agent_owner_id"].strip(),
        "HubSpot owner must be null for technical reads or an explicit source agent owner",
        "invalid_service_settings",
    )
    teams = value["visible_team_ids"]
    check(
        isinstance(teams, list)
        and teams
        and all(isinstance(t, str) and t.strip() for t in teams)
        and len(set(teams)) == len(teams),
        "HubSpot requires explicit distinct source team grants",
        "invalid_service_settings",
    )
    check(
        type(value["allow_restricted_properties"]) is bool,
        "HubSpot property access must be explicit",
        "invalid_service_settings",
    )
    for name in required - {
        "agent_owner_id",
        "visible_team_ids",
        "allow_restricted_properties",
    }:
        check(
            type(value[name]) is int and value[name] > 0,
            "HubSpot document limits must be positive integers",
            "invalid_service_settings",
        )
    return value


def document_path(value):
    check(
        isinstance(value, str)
        and value.startswith("contracts/")
        and "\\" not in value
        and ":" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and not any(ord(c) < 32 or ord(c) == 127 for c in value)
        and PurePosixPath(value).suffix.lower() in {".pdf", ".md"},
        "Unsafe or unsupported linked contract path",
    )
    return value


def record_table(record):
    table = record.get("table") if isinstance(record, dict) else None
    check(
        table in ALL_KEYS and set(record) == {"table", *ALL_KEYS[table]},
        "Exact HubSpot table and complete native key required",
    )
    check(
        all(isinstance(record[k], str) and record[k] for k in ALL_KEYS[table]),
        "HubSpot keys must be nonempty strings",
    )
    return table


def key(table, row):
    return (table, *(row[k] for k in ALL_KEYS[table]))


def validate_row(table, row, snapshot):
    columns = DOCUMENT_COLUMNS if table == "documents" else TABLES[table]
    check(
        isinstance(row, dict) and set(row) == set(columns),
        "HubSpot delta requires complete rows with exact native fields",
    )
    for field, value in row.items():
        if (
            value is None
            and field in NULLABLE
            and not (table in OBJECT_TYPES and field == "owner_id")
        ):
            continue
        if field in {"archived", "bytes"}:
            check(
                type(value) is int
                and value >= 0
                and (field != "archived" or value in (0, 1)),
                "Invalid HubSpot integer",
            )
        else:
            check(isinstance(value, str), "Invalid HubSpot native field type")
        if field in {
            "id",
            "record_id",
            "activity_id",
            "owner_id",
            "access_team_id",
            "from_record_id",
            "to_record_id",
        }:
            check(
                bool(value) and len(value) <= (512 if table == "documents" else 128),
                "Invalid HubSpot native ID",
            )
        if field in {"object_type", "from_object_type", "to_object_type"}:
            check(value in OBJECT_TYPES, "Invalid HubSpot object type")
        if field in {
            "created_at",
            "updated_at",
            "occurred_at",
            "archived_at",
            "timestamp",
        }:
            instant = timestamp(value, "hubspot", field)
            check(
                instant <= snapshot,
                "HubSpot observed state is after snapshot",
                "state_after_snapshot",
            )
        if field == "due_at":
            timestamp(value, "hubspot", field)  # scheduled work may be later
    if row.get("updated_at") and row.get("created_at"):
        check(
            timestamp(row["updated_at"], "hubspot", "updated_at")
            >= timestamp(row["created_at"], "hubspot", "created_at"),
            "HubSpot update precedes creation",
        )
    if table == "activities":
        check(
            isinstance(json.loads(row["metadata_json"]), dict),
            "Activity metadata must be an object",
        )
    if table == "documents":
        document_path(row["id"])
        check(
            re.fullmatch("[0-9a-f]{64}", row["sha256"]) and row["bytes"] > 0,
            "Invalid document hash or size",
        )
        check(
            row["media_type"]
            == ("application/pdf" if row["id"].endswith(".pdf") else "text/markdown"),
            "Document media type does not match path",
        )


STRING = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "items": {"type": "string"}, "maxItems": 50}
PAGING = {
    "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
    "after": {"type": ["string", "null"]},
}
ORDER = {"order": {"type": "string", "enum": ["asc", "desc"]}}
PARENT = {
    "object_type": {"type": "string", "enum": list(OBJECT_TYPES)},
    "record_id": STRING,
}
SEARCH = {
    "query": {"type": "string"},
    "properties": STRINGS,
    "sort": STRING,
    "filters": {
        "type": "array",
        "maxItems": 12,
        "items": {
            "type": "object",
            "properties": {
                "property_name": STRING,
                "operator": STRING,
                "value": {"type": ["string", "number", "boolean", "null"]},
                "values": STRINGS,
            },
            "required": ["property_name", "operator"],
            "additionalProperties": False,
        },
    },
    "archived": {"type": "boolean"},
    **PAGING,
}
SPECS = {
    **{f"hubspot_search_{t}": ({}, SEARCH, "search") for t in OBJECT_TYPES},
    "hubspot_get_record": (
        PARENT,
        {
            "properties": STRINGS,
            "associations": STRINGS,
            "properties_with_history": STRINGS,
            "archived": {"type": "boolean"},
        },
        "record",
    ),
    "hubspot_list_metadata": (
        {},
        {
            "catalog": {"type": "string", "enum": [*CATALOGS, "property_values"]},
            "object_type": PARENT["object_type"],
            "property_name": STRING,
            "archived": {"type": "boolean"},
            "owner_id": STRING,
            **PAGING,
        },
        "metadata",
    ),
    "hubspot_list_metrics": (
        {},
        {"object_type": PARENT["object_type"], **PAGING},
        "metrics",
    ),
    "hubspot_read_history": (
        {**PARENT, "property_name": STRING},
        {
            **PAGING,
            **ORDER,
            **{k: STRING for k in ("start_time", "end_time", "history_id")},
        },
        "history",
    ),
    "hubspot_list_activities": (
        PARENT,
        {
            **PAGING,
            **ORDER,
            **{
                k: STRING
                for k in ("activity_id", "start_time", "end_time", "activity_type")
            },
            "query": {"type": "string", "maxLength": 200},
            "detail": {"type": "string", "enum": ["compact", "full"]},
            "body_chars": {"type": "integer", "minimum": 0, "maximum": 4000},
        },
        "activities",
    ),
    "hubspot_list_associations": (
        PARENT,
        {**PAGING, "association_id": STRING, "object_types": STRINGS},
        "associations",
    ),
    "hubspot_read_contract": (
        PARENT,
        {
            "document_id": STRING,
            "property_name": STRING,
            "page": {"type": "integer", "minimum": 1, "maximum": 10000},
            "offset": {"type": "integer", "minimum": 0, "maximum": 10000000},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 64000},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "contract",
    ),
    "hubspot_search_contracts": (
        {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
        {
            "object_type": PARENT["object_type"],
            "record_id": STRING,
            "archived": {"type": "boolean"},
            "after": {"type": ["string", "null"]},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 10},
            "match_mode": {"type": "string", "enum": ["all", "any", "phrase"]},
            "context_chars": {"type": "integer", "minimum": 0, "maximum": 3000},
        },
        "contract_search",
    ),
}


def tool_definitions(scope, cap):
    return [
        {
            "name": name,
            "description": load_text("tools/hubspot/" + help_name + ".txt"),
            "parameters": {
                "type": "object",
                "properties": {**required, **optional},
                "required": list(required),
                "additionalProperties": False,
            },
        }
        for name, (required, optional, help_name) in SPECS.items()
        if name in scope
    ]


def valid_arguments(value, schema):
    if "anyOf" in schema:
        return any(valid_arguments(value, option) for option in schema["anyOf"])
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    kind = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        type(None): "null",
    }.get(type(value))
    if kind not in types and not (kind == "integer" and "number" in types):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind in {"integer", "number"} and not schema.get(
        "minimum", -1e300
    ) <= value <= schema.get("maximum", 1e300):
        return False
    if kind == "string" and (
        len(value) < schema.get("minLength", 0)
        or len(value) > min(10000, schema.get("maxLength", 10000))
    ):
        return False
    if kind == "array" and (
        len(value) > schema.get("maxItems", 100)
        or not all(valid_arguments(v, schema["items"]) for v in value)
    ):
        return False
    if kind == "object":
        props = schema.get("properties", {})
        return (
            not (set(value) - props.keys())
            and set(schema.get("required", [])) <= value.keys()
            and all(valid_arguments(v, props[k]) for k, v in value.items())
        )
    return True


class HubSpotEnvironment:
    def __init__(
        self, package, manifest, task, config, work_dir, max_result_bytes, acquire
    ):
        self.scope = set(config["tool_scope"]) & SPECS.keys()
        self.settings = settings(config)
        self.package, self.task, self.acquire = Path(package), task, acquire
        self.work, self.limit = Path(work_dir) / "hubspot", max_result_bytes
        self.snapshot = timestamp(config["snapshot_time"], "config", "snapshot_time")
        self.tools = None
        self.document_cache = self.work / "text-cache"
        self.document_source = config["build_ref"]
        self.manifest = {
            f["path"]: f
            for f in manifest["files"]
            if f["path"].startswith("delta/hubspot/")
        }
        if FILE in self.manifest:
            check(
                self.manifest[FILE]["bytes"] <= 8_388_608, "HubSpot seed exceeds 8 MiB"
            )
            self.delta = load_json(self.package / FILE)
        else:
            self.delta = {}
        check(
            isinstance(self.delta, dict)
            and not self.delta.keys() - {*TABLES, "documents"},
            "Unsupported HubSpot delta tables",
        )
        payload = set()
        for table, rows in self.delta.items():
            check(isinstance(rows, list), "HubSpot delta table must be an array")
            for row in rows:
                validate_row(table, row, self.snapshot)
                k = key(table, row)
                check(k not in payload, "Duplicate HubSpot delta key")
                payload.add(k)
        inventory = set()
        for change in task["changes"]:
            if change["service"] != "hubspot":
                continue
            record = change["source"]["record"]
            table = record_table(record)
            check(
                table in {*TABLES, "documents"},
                "HubSpot tasks cannot alter catalogs or grants",
            )
            k = key(table, record)
            expected = (
                "delta/hubspot/" + document_path(record["id"])
                if table == "documents"
                else FILE
            )
            check(
                change["file"] == expected and k not in inventory,
                "Invalid HubSpot change inventory",
            )
            inventory.add(k)
        check(
            payload == inventory,
            "HubSpot payload and change inventory must match exactly",
        )
        paths = {FILE} if FILE in self.manifest else set()
        for doc in self.delta.get("documents", []):
            path = "delta/hubspot/" + doc["id"]
            check(
                path in self.manifest
                and self.manifest[path]["bytes"] == doc["bytes"]
                and self.manifest[path]["sha256"] == doc["sha256"],
                "Document must match its pinned package artifact",
            )
            paths.add(path)
        check(paths == set(self.manifest), "Unlisted HubSpot document artifacts")
        check(
            not (
                self.manifest
                or inventory
                or any(e["service"] == "hubspot" for e in task["evidence"])
            )
            or self.scope,
            "HubSpot task data requires enabled reads",
        )

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def open(self):
        if self.tools is not None:
            return
        if not self.scope:
            raise HubSpotEnvironmentError("HubSpot reads are not configured")
        baseline = self.acquire()
        self.work.mkdir(parents=True, exist_ok=True)
        build_runtime_overlay(self.delta, self.work / "overlay.sqlite")
        value = self.settings
        try:
            self.tools = RuntimeHubSpotTools(
                HubSpotConfig(
                    baseline_dir=baseline,
                    overlay_dir=self.work,
                    agent_owner_id=value["agent_owner_id"],
                    visible_team_ids=tuple(value["visible_team_ids"]),
                    allow_restricted_properties=value["allow_restricted_properties"],
                    max_page_size=100,
                )
            )
            self.tools.source_identity = hashlib.sha256(
                json.dumps([self.document_source, self.delta], sort_keys=True).encode()
            ).hexdigest()
            if value["agent_owner_id"] is not None:
                owner = self.tools._connection.execute(
                    "SELECT user_id FROM baseline.owners WHERE id=?",
                    (value["agent_owner_id"],),
                ).fetchone()
                check(
                    owner is not None and owner["user_id"] == "U_AGENT",
                    "HubSpot owner must identify the benchmark agent, not an employee",
                    "invalid_service_settings",
                )
            for team in value["visible_team_ids"]:
                check(
                    self.tools._connection.execute(
                        "SELECT 1 FROM baseline.teams WHERE id=?", (team,)
                    ).fetchone(),
                    "Pinned HubSpot team grant is absent from source",
                )
            self._validate_effective()
            for evidence in self.task["evidence"]:
                if evidence["service"] == "hubspot":
                    check(
                        "record" in evidence["source"],
                        "HubSpot evidence requires an exact native record",
                    )
                    check(
                        self.exact(evidence["source"]["record"], "effective")[0]
                        is not None,
                        "HubSpot evidence is missing",
                    )
        except HubSpotToolError as error:
            self.close()
            raise HubSpotEnvironmentError(
                "HubSpot source identity or native schema is invalid: " + error.message
            ) from error
        except BaseException:
            self.close()
            raise

    def _row(self, table, fields, view="effective"):
        db = self.tools._connection
        selected = (
            "baseline." + table
            if view == "baseline" or table in CATALOGS
            else "effective_" + table
        )
        return db.execute(
            "SELECT * FROM "
            + selected
            + " WHERE "
            + " AND ".join(k + "=?" for k in fields),
            tuple(fields.values()),
        ).fetchone()

    def _visible(self, table, row, view):
        needed = "hubspot_get_record"
        if table in OBJECT_TYPES:
            parent = row
        elif table in {
            "property_values",
            "history",
            "activity_associations",
            "documents",
        }:
            parent = self._row(row["object_type"], {"id": row["record_id"]}, view)
            if table in {"property_values", "history", "documents"}:
                self.tools._property_definition(
                    row["object_type"], row["property_name"]
                )
            needed = {
                "history": "hubspot_read_history",
                "documents": "hubspot_read_contract",
                "activity_associations": "hubspot_list_activities",
            }.get(table, needed)
            if table == "documents" and "hubspot_search_contracts" in self.scope:
                needed = "hubspot_search_contracts"
        elif table == "activities":
            selected = (
                "baseline.activity_associations"
                if view == "baseline"
                else "effective_activity_associations"
            )
            links = self.tools._connection.execute(
                "SELECT * FROM " + selected + " WHERE activity_id=?", (row["id"],)
            )
            parent = next(
                (
                    p
                    for link in links
                    if (
                        p := self._row(
                            link["object_type"], {"id": link["record_id"]}, view
                        )
                    )
                    is not None
                    and self.tools._can_read(p)
                ),
                None,
            )
            needed = "hubspot_list_activities"
        elif table == "associations":
            parent = self._row(
                row["from_object_type"], {"id": row["from_record_id"]}, view
            )
            target = self._row(row["to_object_type"], {"id": row["to_record_id"]}, view)
            check(
                target is not None and self.tools._can_read(target),
                "HubSpot association target is inaccessible",
            )
            needed = "hubspot_list_associations"
        else:
            check(
                "hubspot_list_metadata" in self.scope,
                "HubSpot catalog read is not configured",
            )
            if table == "property_definitions":
                self.tools._property_definition(row["object_type"], row["name"])
            if table in {"teams", "owners"}:
                check(
                    row["id" if table == "teams" else "team_id"]
                    in self.settings["visible_team_ids"],
                    "HubSpot catalog row is inaccessible",
                )
            return
        check(
            needed in self.scope
            and parent is not None
            and self.tools._can_read(parent),
            "HubSpot evidence is inaccessible through configured reads",
        )

    def _validate_effective(self):
        for change in self.task["changes"]:
            if change["service"] != "hubspot":
                continue
            record = change["source"]["record"]
            table = record["table"]
            if table == "documents":
                # New document versions use distinct paths. Never shadow old bytes.
                check(
                    change["operation"] == "add",
                    "Contract changes require a new immutable document version",
                )
                self._document_link(record, "effective")
                check(
                    record["id"] not in self._linked_paths(record, "baseline"),
                    "Contract version already exists in baseline",
                )
                continue
            fields = {k: record[k] for k in KEYS[table]}
            original = self._row(table, fields, "baseline")
            check(
                (original is not None) == (change["operation"] == "replace"),
                "HubSpot add/replace conflicts with baseline existence",
            )
            effective = self._row(table, fields)
            stable = {
                "id",
                "record_id",
                "object_type",
                "from_record_id",
                "from_object_type",
                "to_record_id",
                "to_object_type",
                "activity_id",
                "created_at",
            } & set(effective.keys())
            if original:
                check(
                    all(original[k] == effective[k] for k in stable),
                    "HubSpot replacement changes native parent or creation identity",
                )
                if table in OBJECT_TYPES:
                    check(
                        all(
                            original[k] == effective[k]
                            for k in ("owner_id", "access_team_id")
                        ),
                        "HubSpot task replacements cannot change access grants",
                    )
        self._validate_modified_records()
        self._validate_activity_effects()
        # Only identity keys are needed for referential checks. Reuse those
        # sets instead of re-reading a full parent for each CRM history row.
        # Effective views retain exactly the same add/replace semantics.
        db = self.tools._connection
        identities = {
            table: {
                r["id"]
                for r in db.execute(
                    "SELECT id FROM "
                    + (
                        "baseline." + table
                        if table in CATALOGS
                        else "effective_" + table
                    )
                )
            }
            for table in (*OBJECT_TYPES, "owners", "teams", "activities")
        }
        # Native rows stay on disk. Validate references and timestamps by streaming.
        for table in TABLES:
            for raw in self.tools._connection.execute(
                "SELECT * FROM effective_" + table
            ):
                row = dict(raw)
                validate_row(table, row, self.snapshot)
                if table in OBJECT_TYPES:
                    check(
                        row["owner_id"] in identities["owners"]
                        and row["access_team_id"] in identities["teams"],
                        "HubSpot record has missing owner/team",
                    )
                if "object_type" in row:
                    check(
                        row["record_id"] in identities[row["object_type"]],
                        "HubSpot row has missing parent",
                    )
                if "property_name" in row:
                    check(
                        (row["object_type"], row["property_name"])
                        in self.tools._definitions,
                        "HubSpot row has undefined property",
                    )
                    definition = self.tools._definitions[
                        row["object_type"], row["property_name"]
                    ]
                    # Native CRM exports use both null and an empty string for
                    # an unset or cleared property. Neither is a numeric zero.
                    if row["value"] not in (None, ""):
                        if definition["type"] == "number":
                            try:
                                valid = math.isfinite(float(row["value"]))
                            except ValueError:
                                valid = False
                            check(valid, "HubSpot number property is invalid")
                        if definition["type"] == "bool":
                            check(
                                row["value"] in {"true", "false", "0", "1"},
                                "HubSpot boolean property is invalid",
                            )
                        if row["property_name"] in {
                            "hs_lastmodifieddate",
                            "last_contacted",
                            "csat_response_date",
                            "createdate",
                        }:
                            check(
                                timestamp(row["value"], "hubspot", "value")
                                <= self.snapshot,
                                "HubSpot observed property is after snapshot",
                                "state_after_snapshot",
                            )
                if table == "associations":
                    check(
                        all(
                            row[side + "_record_id"]
                            in identities[row[side + "_object_type"]]
                            for side in ("from", "to")
                        ),
                        "HubSpot association has missing endpoint",
                    )
                    check(
                        (row["from_object_type"], row["from_record_id"])
                        != (row["to_object_type"], row["to_record_id"]),
                        "HubSpot self association",
                    )
                if table == "activity_associations":
                    check(
                        row["activity_id"] in identities["activities"],
                        "HubSpot activity link has missing activity",
                    )
                if table == "activities" and row["owner_id"] is not None:
                    check(
                        row["owner_id"] in identities["owners"],
                        "HubSpot activity owner is absent",
                    )
        db = self.tools._connection
        for row in db.execute("SELECT id FROM effective_deals"):
            fields = {
                p["property_name"]: p["value"]
                for p in db.execute(
                    "SELECT property_name,value FROM effective_property_values WHERE object_type='deals' AND record_id=? AND property_name IN ('pipeline','dealstage')",
                    (row["id"],),
                )
            }
            if fields:
                check(
                    set(fields) == {"pipeline", "dealstage"}
                    and db.execute(
                        "SELECT 1 FROM baseline.pipelines WHERE pipeline_id=?",
                        (fields["pipeline"],),
                    ).fetchone()
                    and db.execute(
                        "SELECT 1 FROM baseline.pipeline_stages WHERE pipeline_id=? AND stage_id=?",
                        (fields["pipeline"], fields["dealstage"]),
                    ).fetchone(),
                    "HubSpot deal pipeline/stage is invalid",
                )
        check(
            not db.execute(
                "SELECT 1 FROM effective_associations GROUP BY from_object_type,from_record_id,to_object_type,to_record_id,association_type HAVING COUNT(*)>1 LIMIT 1"
            ).fetchone(),
            "Duplicate effective HubSpot association",
        )

    def _validate_modified_records(self):
        """Task changes must preserve the native core/property modification clock."""
        affected = {
            (table, row["id"])
            for table in OBJECT_TYPES
            for row in self.delta.get(table, [])
        }
        affected.update(
            (r["object_type"], r["record_id"])
            for table in ("property_values", "history")
            for r in self.delta.get(table, [])
        )
        faults = []
        for kind, rid in sorted(affected):
            name = "lastmodifieddate" if kind == "contacts" else "hs_lastmodifieddate"
            if (kind, name) not in self.tools._definitions:
                continue
            row = self._row(kind, dict(id=rid))
            check(
                row is not None and self.tools._can_read(row),
                "Modified CRM record is absent or inaccessible",
            )
            self.tools._property_definition(kind, name)
            prop = self._row(
                "property_values",
                dict(object_type=kind, record_id=rid, property_name=name),
            )
            at = timestamp(row["updated_at"], "hubspot", "updated_at")
            if (
                prop is None
                or not prop["value"]
                or timestamp(prop["value"], "hubspot", name) != at
            ):
                faults.append(f"{kind}/{rid} updated_at must agree with {name}")
            stamps = [
                r["updated_at"]
                for r in self.delta.get("property_values", [])
                if (r["object_type"], r["record_id"]) == (kind, rid)
            ]
            stamps += [
                r["timestamp"]
                for r in self.delta.get("history", [])
                if (r["object_type"], r["record_id"]) == (kind, rid)
            ]
            if any(timestamp(s, "hubspot", "modified") > at for s in stamps):
                faults.append(
                    f"{kind}/{rid} updated_at predates an authored property change or observation"
                )
        check(
            not faults,
            "HubSpot task modification consistency: " + "; ".join(faults[:24]),
        )

    def _validate_activity_effects(self):
        """Check task-touched parents only. Never repair or reinterpret source rows."""
        from benchmark.tools.hubspot.metrics import activity_summary

        db = self.tools._connection

        def agrees(name, actual, expected):
            if name in {"num_notes", "num_contacted_notes", "crm_touch_count"}:
                try:
                    return actual is not None and float(actual) == int(expected)
                except (ValueError, TypeError):
                    return False
            return (
                (not actual and not expected)
                or bool(actual and expected)
                and timestamp(actual, "hubspot", name)
                == timestamp(expected, "hubspot", name)
            )

        def activities(kind, rid):
            return db.execute(
                "SELECT DISTINCT a.* FROM effective_activities a JOIN effective_activity_associations l ON l.activity_id=a.id WHERE l.object_type=? AND l.record_id=?",
                (kind, rid),
            )

        affected, faults = set(), []
        for activity in self.delta.get("activities", []):
            owner = (
                self._row("owners", {"id": activity["owner_id"]})
                if activity["owner_id"]
                else None
            )
            if not (
                owner
                and owner["active"]
                and owner["user_id"] != "U_AGENT"
                and owner["team_id"] in self.settings["visible_team_ids"]
                and activity["source_type"].strip()
                and activity["source_id"].strip()
            ):
                faults.append(
                    "activity "
                    + activity["id"]
                    + " needs valid active native employee attribution and source provenance"
                )
            links = list(
                db.execute(
                    "SELECT object_type,record_id FROM effective_activity_associations WHERE activity_id=?",
                    (activity["id"],),
                )
            )
            if not links:
                faults.append(
                    "activity "
                    + activity["id"]
                    + " needs an explicit record association"
                )
            affected.update(tuple(r) for r in links)
            affected.update(
                tuple(r)
                for r in db.execute(
                    "SELECT object_type,record_id FROM baseline.activity_associations WHERE activity_id=?",
                    (activity["id"],),
                )
            )
        affected.update(
            (r["object_type"], r["record_id"])
            for r in self.delta.get("activity_associations", [])
        )
        summary_names = {
            "num_notes",
            "num_contacted_notes",
            "crm_touch_count",
            "last_activity_date",
            "notes_last_updated",
            "hs_last_sales_activity_timestamp",
        }
        affected.update(
            (r["object_type"], r["record_id"])
            for t in ("property_values", "history")
            for r in self.delta.get(t, [])
            if r["property_name"] in summary_names
        )
        for kind, rid in sorted(affected):
            parent = self._row(kind, {"id": rid})
            check(
                parent is not None and self.tools._can_read(parent),
                "Task activity parent is absent or inaccessible",
            )
            expected = activity_summary(activities(kind, rid), kind)
            for name, value in expected.items():
                if (kind, name) not in self.tools._definitions:
                    continue
                self.tools._property_definition(kind, name)  # retain property grants
                current = self._row(
                    "property_values",
                    dict(object_type=kind, record_id=rid, property_name=name),
                )
                same = current is not None and agrees(name, current["value"], value)
                if not same:
                    faults.append(
                        f"{kind}/{rid} {name} must match linked activities (expected {value!r})"
                    )
                # When this metric has history, a changed current value needs a
                # matching new observation. Original observations stay intact.
                original = self._row(
                    "property_values",
                    dict(object_type=kind, record_id=rid, property_name=name),
                    "baseline",
                )
                changed = original is None or not agrees(name, original["value"], value)
                tracked = db.execute(
                    "SELECT 1 FROM baseline.history WHERE object_type=? AND record_id=? AND property_name=? LIMIT 1",
                    (kind, rid, name),
                ).fetchone()
                observations = [
                    r
                    for r in self.delta.get("history", [])
                    if (r["object_type"], r["record_id"], r["property_name"])
                    == (kind, rid, name)
                ]
                if (
                    changed
                    and tracked
                    and not any(agrees(name, r["value"], value) for r in observations)
                ):
                    faults.append(
                        f"{kind}/{rid} {name} needs a matching task history observation"
                    )
                for observation in observations:
                    at = timestamp(observation["timestamp"], "hubspot", "timestamp")
                    dated = activity_summary(
                        (
                            a
                            for a in activities(kind, rid)
                            if timestamp(a["occurred_at"], "hubspot", "occurred_at")
                            <= at
                        ),
                        kind,
                    )[name]
                    if not agrees(name, observation["value"], dated) or not all(
                        observation[k].strip() for k in ("source_type", "source_id")
                    ):
                        faults.append(
                            f"{kind}/{rid} {name} history must match dated activities with source provenance"
                        )
        check(
            not faults, "HubSpot task activity consistency: " + "; ".join(faults[:24])
        )

    def _linked_paths(self, record, view):
        fields = (record["object_type"], record["record_id"], record["property_name"])
        db = self.tools._connection
        prefix = "baseline." if view == "baseline" else "effective_"
        return {
            r[0]
            for r in db.execute(
                "SELECT value FROM "
                + prefix
                + "property_values WHERE object_type=? AND record_id=? AND property_name=? UNION SELECT value FROM "
                + prefix
                + "history WHERE object_type=? AND record_id=? AND property_name=?",
                (*fields, *fields),
            )
            if r[0] is not None
        }

    def _document_link(self, record, view):
        self._visible("documents", record, view)
        check(
            record["id"] in self._linked_paths(record, view),
            "Contract is not linked by the readable CRM record",
        )

    def exact(self, record, view):
        table = record_table(record)
        self.open()
        if table == "documents":
            from .hubspot_documents import linked_contract_paths

            if record["id"] not in linked_contract_paths(self, record, view):
                return (
                    None,
                    list(DOCUMENT_COLUMNS),
                    {k: "TEXT" for k in DOCUMENT_COLUMNS},
                )
            result = self.read_contract(
                record["object_type"],
                record["record_id"],
                property_name=record["property_name"],
                document_id=record["id"],
                view=view,
            )
            return result, list(result), {k: "TEXT" for k in result}
        row = self._row(table, {k: record[k] for k in ALL_KEYS[table]}, view)
        types = {
            r["name"]: r["type"]
            for r in self.tools._connection.execute(
                "PRAGMA baseline.table_info(" + table + ")"
            )
        }
        if row is not None:
            self._visible(table, row, view)
            row = dict(row)
            if table == "activities":
                row["metadata_json"] = json.dumps(
                    public_metadata(json.loads(row["metadata_json"]))
                )
        return row, list(types), types

    def read_contract(
        self,
        object_type,
        record_id,
        property_name="contract_document_path",
        document_id=None,
        page=1,
        offset=0,
        max_chars=12000,
        view="effective",
        max_pages=1,
    ):
        from .hubspot_documents import read_contract

        return read_contract(
            self,
            object_type,
            record_id,
            property_name,
            document_id,
            page,
            offset,
            max_chars,
            view,
            max_pages,
        )

    def configure_document_cache(self, root, source_identity=None):
        self.document_cache = Path(root)
        if source_identity is not None:
            self.document_source = source_identity

    def prepare_contract_search(self, **arguments):
        from .hubspot_documents import prepare_contract_search

        required, optional, _ = SPECS["hubspot_search_contracts"]
        optional = {
            k: optional[k] for k in ("after", "archived", "object_type", "record_id")
        }
        optional.update(
            max_documents=dict(type="integer", minimum=1, maximum=10),
            max_pages=dict(type="integer", minimum=1, maximum=100),
        )
        check(
            valid_arguments(
                dict(query="\0", **arguments),
                dict(
                    type="object",
                    properties={**required, **optional},
                    required=list(required),
                ),
            ),
            "Invalid HubSpot arguments",
        )
        self.open()
        return prepare_contract_search(self, **arguments)

    def call(self, name, arguments):
        if name not in self.scope:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": "Unknown or disallowed tool",
            }
        required, optional, _ = SPECS[name]
        schema = {
            "type": "object",
            "properties": {**required, **optional},
            "required": list(required),
        }
        if not valid_arguments(arguments, schema):
            return {
                "ok": False,
                "error": "invalid_arguments",
                "message": "Invalid HubSpot arguments",
            }
        self.open()
        try:
            if name == "hubspot_read_contract":
                result = self.read_contract(**arguments)
            elif name == "hubspot_search_contracts":
                from .hubspot_documents import search_contracts

                result = search_contracts(self, **arguments)
            else:
                result = getattr(self.tools, name)(**arguments)
            result = public_metadata(result)
            if len(_encoded(result)) > self.limit:
                return {
                    "ok": False,
                    "error": "result_too_large",
                    "message": "CRM result exceeds the result byte limit",
                }
            return result
        except HubSpotToolError as error:
            return error.as_dict()

    def close(self):
        if self.tools is not None:
            self.tools.close()
            self.tools = None


def public_metadata(value):
    # Harness markers are metadata keys, not business history source fields.
    if isinstance(value, dict):
        return {
            k: (
                json.dumps(public_metadata(json.loads(v)))
                if k == "metadata_json" and isinstance(v, str)
                else public_metadata(v)
            )
            for k, v in value.items()
            if k not in {"_benchmark_origin", "_task_origin", "_harness_origin"}
        }
    if isinstance(value, list):
        return [public_metadata(v) for v in value]
    return value
