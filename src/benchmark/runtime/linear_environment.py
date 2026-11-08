"""Two bounded native Linear reads inside the existing attempt worker."""

from __future__ import annotations

from benchmark.prompts import load_text

import math
import re
from datetime import date
from pathlib import Path

from .contracts import load_json, timestamp
from .slack_environment import _encoded
from benchmark.tools.linear.linear import (
    BASELINE_TABLE_COLUMNS,
    OVERLAY_TABLE_COLUMNS,
    PRIMARY_KEYS,
    EFFECTIVE_ISSUES_CTE,
    EFFECTIVE_LABELS_CTE,
    EFFECTIVE_COMMENTS_CTE,
    EFFECTIVE_RELATIONS_CTE,
    EFFECTIVE_HISTORY_CTE,
    LinearConfig,
    LinearTools,
    LinearToolError,
    build_linear_overlay_from_seed,
)

# Sole Linear definition, argument validation and dispatch registry.
LINEAR_READS = {
    "linear_search_issues": (
        {},
        {
            "query": {"type": "string", "default": ""},
            **{
                key: {"type": ["string", "null"]}
                for key in (
                    "team_id",
                    "project_id",
                    "assignee_id",
                    "state",
                    "start_cursor",
                )
            },
            "label_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            "include_archived": {"type": "boolean", "default": False},
            "page_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
            },
        },
        load_text("tools/linear/linear_search_issues.txt"),
    ),
    "linear_get_issue": (
        {"issue_id": {"type": "string", "minLength": 1}},
        {},
        load_text("tools/linear/linear_get_issue.txt"),
    ),
}
LINEAR_GUIDE = load_text("guides/linear/data_overview.txt")
FILE = "delta/linear/seed.json"
CTES = dict(
    zip(
        OVERLAY_TABLE_COLUMNS,
        (
            EFFECTIVE_ISSUES_CTE,
            EFFECTIVE_LABELS_CTE,
            EFFECTIVE_COMMENTS_CTE,
            EFFECTIVE_RELATIONS_CTE,
            EFFECTIVE_HISTORY_CTE,
        ),
    )
)
NULLABLE = {
    "issues": {"project_id", "assignee_id", "estimate", "due_date", "completed_at"},
    "projects": {"lead_id", "start_date", "target_date"},
    "labels": {"team_id"},
    "history": {"from_value", "to_value"},
}
INTEGERS = {"number", "priority", "estimate", "archived", "active", "private"}
ENUMS = {
    ("team_members", "role"): {"member", "lead", "guest"},
    ("projects", "status"): {"planned", "started", "paused", "completed", "canceled"},
    ("workflow_states", "type"): {
        "backlog",
        "unstarted",
        "started",
        "completed",
        "canceled",
    },
    ("relations", "type"): {"blocks", "blocked_by", "duplicate_of", "related"},
}
REFERENCES = {
    "team_id": "teams",
    "user_id": "users",
    "lead_id": "users",
    "state_id": "workflow_states",
    "project_id": "projects",
    "label_id": "labels",
    "creator_id": "users",
    "actor_id": "users",
    "assignee_id": "users",
    "issue_id": "issues",
    "related_issue_id": "issues",
}


class LinearEnvironmentError(ValueError):
    def __init__(self, reason, *, record=None, code="invalid_linear_environment"):
        self.code, self.record = code, record
        self.source, self.validation_reason = "linear", reason
        super().__init__(reason)


def _fail(table, row, reason, code="invalid_linear_environment"):
    record = {"table": table, **{k: row.get(k) for k in PRIMARY_KEYS[table]}}
    raise LinearEnvironmentError(f"Linear {record}: {reason}", record=record, code=code)


def _key(table, row):
    return (table, *(row[k] for k in PRIMARY_KEYS[table]))


def _record(value):
    table = value.get("table")
    if (
        not isinstance(table, str)
        or table not in OVERLAY_TABLE_COLUMNS
        or set(value) != {"table", *PRIMARY_KEYS[table]}
    ):
        raise LinearEnvironmentError(
            "Linear record must name a supported table and exact native primary key"
        )
    if any(not isinstance(value[k], str) or not value[k] for k in PRIMARY_KEYS[table]):
        raise LinearEnvironmentError("Linear record keys must be nonempty strings")
    return table


def _validate_row(table, row, snapshot):
    """Types before compilation, and again for streamed effective baseline rows."""
    for field in BASELINE_TABLE_COLUMNS[table]:
        value = row[field]
        if value is None and field in NULLABLE.get(table, ()):
            continue
        if field in INTEGERS:
            valid = type(value) is int
        elif field == "position":
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = isinstance(value, str)
        if not valid:
            _fail(table, row, f"{field}: invalid type")
        if field == "id" or field.endswith("_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
                _fail(table, row, f"{field}: invalid native ID")
        if field in {"active", "archived", "private"} and value not in (0, 1):
            _fail(table, row, f"{field}: expected 0 or 1")
        if field == "number" and value <= 0:
            _fail(table, row, "number must be positive")
        if field == "priority" and not 0 <= value <= 4:
            _fail(table, row, "priority must be 0 to 4")
        if field == "estimate" and not 0 <= value <= LinearConfig.max_estimate:
            _fail(table, row, "estimate outside native range")
        if (table, field) in ENUMS and value not in ENUMS[table, field]:
            _fail(table, row, f"{field}: invalid enum")
        if (
            field
            in {"title", "body", "field", "identifier", "key", "name", "email", "slug"}
            and not value.strip()
        ):
            _fail(table, row, f"{field}: empty text")
        if field.endswith("_date"):
            try:
                date.fromisoformat(value)
            except ValueError:
                _fail(table, row, f"{field}: invalid date")
    observed = {}
    for field in ("created_at", "updated_at", "completed_at"):
        if row.get(field) is not None:
            try:
                observed[field] = timestamp(row[field], "linear", field)
            except ValueError:
                _fail(table, row, f"{field}: expected timezone-aware timestamp")
            if observed[field] > snapshot:
                _fail(
                    table, row, f"{field}: state after snapshot", "state_after_snapshot"
                )
    created = observed.get("created_at")
    updated = observed.get("updated_at", created)
    if created and updated < created:
        _fail(table, row, "updated_at precedes created_at")
    completed = observed.get("completed_at")
    if completed and not created <= completed <= updated:
        _fail(table, row, "completed_at outside created/updated interval")


def _arguments_valid(name, arguments):
    required, optional, _ = LINEAR_READS[name]
    fields = {**required, **optional}
    if (
        not isinstance(arguments, dict)
        or set(arguments) - fields.keys()
        or not required.keys() <= arguments.keys()
    ):
        return False
    for key, value in arguments.items():
        schema = fields[key]
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        kind = {
            str: "string",
            int: "integer",
            bool: "boolean",
            list: "array",
            type(None): "null",
        }.get(type(value))
        if kind not in types:
            return False
        if kind == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
            return False
        if kind == "string" and key == "issue_id" and not value.strip():
            return False
        if kind == "array" and any(type(item) is not str for item in value):
            return False
    return True


def _public_issue(issue):
    issue.pop("source", None)
    for collection in ("comments", "relations", "history"):
        for row in issue.get(collection, ()):
            row.pop("source", None)
    return issue


def tool_definitions(scope, max_result_bytes):
    return [
        {
            "name": name,
            "description": description
            + load_text("tools/linear/result_cap.txt").format(
                max_result_bytes=max_result_bytes
            ),
            "parameters": {
                "type": "object",
                "properties": {**required, **optional},
                "required": list(required),
                "additionalProperties": False,
            },
        }
        for name, (required, optional, description) in LINEAR_READS.items()
        if name in scope
    ]


class LinearEnvironment:
    def __init__(
        self, package, manifest, task, config, work_dir, max_result_bytes, acquire
    ):
        self.scope = set(config["tool_scope"]) & LINEAR_READS.keys()
        self.task, self.limit, self.acquire = task, max_result_bytes, acquire
        self.snapshot = timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.work = Path(work_dir) / "linear"
        self.tools = None
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/linear/")
        ]
        if any(f["path"] != FILE or f["bytes"] > 8_388_608 for f in entries):
            raise LinearEnvironmentError("Linear accepts only seed.json, at most 8 MiB")
        self.delta = load_json(Path(package) / FILE) if entries else {}
        if (
            not isinstance(self.delta, dict)
            or self.delta.keys() - OVERLAY_TABLE_COLUMNS.keys()
        ):
            raise LinearEnvironmentError("Unsupported Linear delta tables")
        payload = set()
        for table, rows in self.delta.items():
            if not isinstance(rows, list):
                raise LinearEnvironmentError("Linear table payload must be an array")
            identifiers = set()
            for row in rows:
                if not isinstance(row, dict) or set(row) - {"operation"} != set(
                    BASELINE_TABLE_COLUMNS[table]
                ):
                    raise LinearEnvironmentError(
                        f"Linear {table}: complete rows with exact native fields required"
                    )
                if row.get("operation", "upsert") != "upsert":
                    _fail(table, row, "only upsert supported")
                _validate_row(table, row, self.snapshot)
                key = _key(table, row)
                if key in payload:
                    _fail(table, row, "duplicate payload key")
                payload.add(key)
                if table == "issues":
                    identifier = row["identifier"].lower()
                    if identifier in identifiers:
                        _fail(table, row, "duplicate payload identifier")
                    identifiers.add(identifier)
        inventory = set()
        for change in task["changes"]:
            if change["service"] == "linear":
                record = change["source"]["record"]
                table = _record(record)
                key = _key(table, record)
                if change["file"] != FILE or key in inventory:
                    _fail(table, record, "invalid or duplicate change inventory")
                inventory.add(key)
        if payload != inventory:
            raise LinearEnvironmentError(
                "Linear payload and task changes must match exactly"
            )
        if (
            entries
            or inventory
            or any(e["service"] == "linear" for e in task["evidence"])
        ) and not self.scope:
            raise LinearEnvironmentError(
                "Linear task data requires enabled Linear reads"
            )

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def open(self):
        if self.tools is not None:
            return
        baseline_dir = self.acquire()
        self.work.mkdir()
        seed = self.work / "seed.json"
        seed.write_bytes(_encoded(self.delta))
        build_linear_overlay_from_seed(seed, self.work / "overlay.sqlite")
        try:
            self.tools = LinearTools(
                LinearConfig(
                    baseline_dir=baseline_dir,
                    overlay_dir=self.work,
                    agent_user_id="U_AGENT",
                )
            )
            self._validate_effective()
            self.tools.connection.execute("PRAGMA query_only=ON")
            self._validate_evidence()
        except BaseException as error:
            self.close()
            if isinstance(error, LinearEnvironmentError):
                raise
            raise LinearEnvironmentError(
                "Linear native database or U_AGENT identity is invalid"
            ) from error

    def _select(self, table):
        return (
            "effective_issue_labels"
            if table == "issue_labels"
            else f"effective_{table}" if table in CTES else f"baseline.{table}"
        )

    def _one(self, table, fields):
        return self.tools.connection.execute(
            "SELECT * FROM "
            + self._select(table)
            + " WHERE "
            + " AND ".join(f"{k} = ?" for k in fields)
            + " LIMIT 1",
            tuple(fields.values()),
        ).fetchone()

    def _validate_effective(self):
        db = self.tools.connection
        for table, columns in BASELINE_TABLE_COLUMNS.items():
            present = {
                row["name"]
                for row in db.execute(f"PRAGMA baseline.table_info({table})")
            }
            missing = set(columns) - present
            if missing:
                raise LinearEnvironmentError(
                    f"Linear baseline {table}: missing native fields {sorted(missing)}",
                    record={"table": table},
                )
        for table, cte in CTES.items():
            name = self._select(table)
            db.execute(f"CREATE TEMP VIEW {name} AS WITH {cte} SELECT * FROM {name}")
        agent = self._one("users", {"id": "U_AGENT"})
        if agent is None or agent["active"] != 1:
            raise LinearEnvironmentError("Baseline requires active U_AGENT")
        changed_issues = {row["id"] for row in self.delta.get("issues", ())}
        for change in self.task["changes"]:
            if change["service"] != "linear":
                continue
            row = change["source"]["record"]
            table = row["table"]
            fields = {k: row[k] for k in PRIMARY_KEYS[table]}
            original = db.execute(
                f"SELECT * FROM baseline.{table} WHERE "
                + " AND ".join(f"{k} = ?" for k in fields),
                tuple(fields.values()),
            ).fetchone()
            if (original is not None) != (change["operation"] == "replace"):
                _fail(table, row, "add/replace conflicts with baseline existence")
            if original is not None:
                effective = self._one(table, fields)
                stable = (
                    ("id", "team_id", "number", "identifier")
                    if table == "issues"
                    else PRIMARY_KEYS[table]
                    + (
                        ("issue_id", "related_issue_id")
                        if table == "relations"
                        else ("issue_id",) if table != "issue_labels" else ()
                    )
                )
                if any(original[k] != effective[k] for k in stable):
                    _fail(table, row, "replacement changes object or parent identity")
        # Stream one row at a time. Catalog and foreign-key lookups use the same
        # attached database, never a copied baseline or an in-memory catalog.
        for table in BASELINE_TABLE_COLUMNS:
            for raw in db.execute(f"SELECT * FROM {self._select(table)}"):
                row = dict(raw)
                _validate_row(table, row, self.snapshot)
                related = {}
                for field, target in REFERENCES.items():
                    if row.get(field) is not None:
                        related[field] = self._one(target, {"id": row[field]})
                        if related[field] is None:
                            _fail(table, row, f"{field}: missing effective {target}")
                if table == "issues":
                    team = related["team_id"]
                    if (
                        row["identifier"].upper()
                        != f"{team['key']}-{row['number']}".upper()
                    ):
                        _fail(table, row, "identifier does not match team/number")
                    # Stored projects may span teams and stored assignments may
                    # outlive membership or activation. New task rows retain the
                    # native write constraints, including complete replacements.
                    for field in (
                        ("state_id", "project_id")
                        if row["id"] in changed_issues
                        else ("state_id",)
                    ):
                        if (
                            field in related
                            and related[field]["team_id"] != row["team_id"]
                        ):
                            _fail(table, row, f"{field}: incompatible team")
                    if (
                        row["id"] in changed_issues
                        and row["assignee_id"] is not None
                        and (
                            not related["assignee_id"]["active"]
                            or self._one(
                                "team_members",
                                {
                                    "team_id": row["team_id"],
                                    "user_id": row["assignee_id"],
                                },
                            )
                            is None
                        )
                    ):
                        _fail(table, row, "assignee must be active and a team member")
                if table == "issue_labels" and related["label_id"]["team_id"] not in (
                    None,
                    related["issue_id"]["team_id"],
                ):
                    _fail(table, row, "label belongs to another team")
                if table == "relations" and row["issue_id"] == row["related_issue_id"]:
                    _fail(table, row, "self relation")
        unique = {
            "issues": [("identifier COLLATE NOCASE",), ("team_id", "number")],
            "relations": [("issue_id", "related_issue_id", "type")],
            "users": [("email",)],
            "teams": [("key COLLATE NOCASE",)],
            "projects": [("slug COLLATE NOCASE",)],
            "workflow_states": [("team_id", "name")],
        }
        for table in BASELINE_TABLE_COLUMNS:
            for fields in [PRIMARY_KEYS[table], *unique.get(table, ())]:
                duplicate = db.execute(
                    f"SELECT {','.join(PRIMARY_KEYS[table])} FROM {self._select(table)} GROUP BY {','.join(fields)} HAVING COUNT(*) > 1 LIMIT 1"
                ).fetchone()
                if duplicate:
                    _fail(table, dict(duplicate), f"duplicate effective {fields}")

    def _validate_evidence(self):
        for evidence in self.task["evidence"]:
            if evidence["service"] != "linear":
                continue
            record = evidence["source"].get("record")
            if not isinstance(record, dict):
                raise LinearEnvironmentError(
                    "Linear first-slice evidence requires a native record"
                )
            table = _record(record)
            row = self._one(table, {k: record[k] for k in PRIMARY_KEYS[table]})
            if row is None:
                _fail(table, record, "evidence is missing")
            issue_id = row["id"] if table == "issues" else row["issue_id"]
            if "linear_get_issue" in self.scope:
                result = self.call("linear_get_issue", {"issue_id": issue_id})
                available = "error" not in result
                if table != "issues" and available:
                    collection = "labels" if table == "issue_labels" else table
                    key = (
                        record["label_id"] if table == "issue_labels" else record["id"]
                    )
                    available = any(r["id"] == key for r in result[collection])
            elif table == "issues" and "linear_search_issues" in self.scope:
                request = {
                    "query": row["identifier"],
                    "team_id": row["team_id"],
                    "include_archived": True,
                    "page_size": 1,
                }
                available = False
                while True:
                    result = self.call("linear_search_issues", request)
                    if "error" in result:
                        break
                    if any(r["id"] == issue_id for r in result["issues"]):
                        available = True
                        break
                    if not result["has_more"]:
                        break
                    request["start_cursor"] = result["next_cursor"]
            else:
                available = False
            if not available:
                _fail(
                    table,
                    record,
                    "required evidence is inaccessible, truncated or oversized",
                )

    def call(self, name, arguments):
        if name not in self.scope:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": "Unknown or disallowed tool",
            }
        if not _arguments_valid(name, arguments):
            return {
                "ok": False,
                "error": "invalid_arguments",
                "message": "Invalid Linear arguments",
            }
        self.open()
        try:
            result = getattr(self.tools, name)(**arguments)
        except LinearToolError as error:
            return {
                "ok": False,
                "error": error.code,
                "message": (
                    "Issue not found"
                    if error.code == "not_found"
                    else "Linear request is invalid or unavailable"
                ),
            }
        if name == "linear_search_issues":
            for issue in result["issues"]:
                _public_issue(issue)
        else:
            _public_issue(result)
        if len(_encoded(result)) > self.limit:
            return {
                "ok": False,
                "error": "result_too_large",
                "message": "Linear result exceeds the result cap; narrow the request or reduce page size",
            }
        return result

    def close(self):
        if self.tools is not None:
            self.tools.close()
            self.tools = None
