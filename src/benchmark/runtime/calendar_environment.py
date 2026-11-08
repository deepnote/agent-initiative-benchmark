"""Read-only native Calendar inside the shared worker and file lease."""

from pathlib import Path
import re
from zoneinfo import ZoneInfo

from benchmark.prompts import load_text
from benchmark.tools.calendar.calendar import (
    CalendarConfig,
    CalendarTools,
    CalendarToolError,
    BASELINE_TABLE_COLUMNS,
    OVERLAY_TABLE_COLUMNS,
    PRIMARY_KEYS,
    EFFECTIVE_EVENTS_CTE,
    EVENT_STATUSES,
    TRANSPARENCIES,
    VISIBILITIES,
    EVENT_TYPES,
    RESPONSE_STATUSES,
    REMINDER_METHODS,
    _event_time,
    _valid_email,
    build_calendar_overlay_from_seed,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/calendar/seed.json"
CHILDREN = ("attendees", "recurrence_rules", "reminders")
NULLABLE = {"recurring_event_id", "original_start_time", "conference_url"}
INTEGERS = {
    "start_utc",
    "end_utc",
    "all_day",
    "sequence",
    "position",
    "optional",
    "organizer",
    "self",
    "resource",
    "minutes",
}
BOOLS = {"all_day", "optional", "organizer", "self", "resource"}
STRING = {"type": "string"}
OPTIONAL_STRING = {"type": ["string", "null"]}
PAGE = {
    "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
    "start_cursor": OPTIONAL_STRING,
}
CALENDARS = {"type": ["array", "null"], "items": STRING, "maxItems": 50}
READS = {
    "calendar_list_calendars": ({}, PAGE),
    "calendar_search_events": (
        {"time_min": STRING, "time_max": STRING},
        dict(
            PAGE,
            query=STRING,
            calendar_ids=CALENDARS,
            attendee_email=OPTIONAL_STRING,
            include_cancelled={"type": "boolean"},
        ),
    ),
    "calendar_get_event": ({"event_id": STRING}, {"calendar_id": OPTIONAL_STRING}),
    "calendar_query_free_busy": (
        {"time_min": STRING, "time_max": STRING},
        {"calendar_ids": CALENDARS},
    ),
}


class CalendarEnvironmentError(ValueError):
    code = "invalid_calendar_environment"
    source = "calendar"

    def __init__(self, reason):
        self.validation_reason = reason
        super().__init__(reason)


def check(condition, message):
    if not condition:
        raise CalendarEnvironmentError(message)


def key(table, row):
    return (table, *(row[k] for k in PRIMARY_KEYS[table]))


def record_table(record):
    table = record.get("table")
    check(
        table in OVERLAY_TABLE_COLUMNS
        and set(record) == {"table", *PRIMARY_KEYS[table]},
        "Calendar requires an exact supported record key",
    )
    for name in PRIMARY_KEYS[table]:
        check(
            (
                (type(record[name]) is int and record[name] >= 0)
                if name == "position"
                else isinstance(record[name], str) and bool(record[name])
            ),
            "Invalid Calendar record key",
        )
    return table


def tool_definitions(scope, cap):
    return [
        dict(
            name=name,
            description=load_text("tools/calendar/" + name + ".txt"),
            parameters=dict(
                type="object",
                properties={**required, **optional},
                required=list(required),
                additionalProperties=False,
            ),
        )
        for name, (required, optional) in READS.items()
        if name in scope
    ]


def arguments_valid(name, args):
    required, optional = READS[name]
    if (
        not isinstance(args, dict)
        or not required.keys() <= args.keys()
        or args.keys() - (required.keys() | optional.keys())
    ):
        return False
    for field, value in args.items():
        schema = {**required, **optional}[field]
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        kind = {
            str: "string",
            int: "integer",
            bool: "boolean",
            list: "array",
            type(None): "null",
        }.get(type(value))
        if kind not in kinds:
            return False
        if kind == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
            return False
        if kind == "array" and (
            not value
            or len(value) > 50
            or any(type(v) is not str or not v for v in value)
        ):
            return False
    return True


def public(value):
    if isinstance(value, dict):
        # html_link is fabricated by the old native sample implementation.
        return {
            k: public(v) for k, v in value.items() if k not in {"source", "html_link"}
        }
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


class CalendarEnvironment:
    def __init__(
        self, package, manifest, task, config, work_dir, max_result_bytes, acquire
    ):
        self.scope = set(config["tool_scope"]) & READS.keys()
        self.task, self.limit, self.acquire = task, max_result_bytes, acquire
        self.snapshot = timestamp(config["snapshot_time"], "config", "snapshot_time")
        self.work = Path(work_dir) / "calendar"
        self.tools = None
        settings = config.get("service_settings", {}).get("calendar", {})
        check(
            not self.scope
            or (
                set(settings)
                in (
                    {"agent_email"},
                    {
                        "agent_email",
                        "access_mode",
                        "delegated_user_id",
                        "owner_authority_sha256",
                    },
                )
                and _valid_email(settings["agent_email"])
            ),
            "Calendar requires explicit service_settings.calendar.agent_email",
        )
        self.email = settings.get("agent_email")
        self.delegation = (
            settings
            if settings.get("access_mode") == "synthetic_source_delegation"
            else None
        )
        check(
            not self.scope
            or set(settings) == {"agent_email"}
            or (
                self.delegation is not None
                and isinstance(settings.get("delegated_user_id"), str)
                and bool(settings["delegated_user_id"])
                and isinstance(settings.get("owner_authority_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", settings["owner_authority_sha256"])
            ),
            "Calendar delegation requires a pinned source principal and owner authority hash",
        )
        entries = [
            e for e in manifest["files"] if e["path"].startswith("delta/calendar/")
        ]
        check(
            all(e["path"] == FILE and e["bytes"] <= 8_388_608 for e in entries),
            "Calendar accepts only seed.json up to 8 MiB",
        )
        self.delta = load_json(Path(package) / FILE) if entries else {}
        check(
            isinstance(self.delta, dict)
            and (not self.delta or set(self.delta) == set(OVERLAY_TABLE_COLUMNS)),
            "Calendar delta requires events, attendees, recurrence_rules and reminders arrays",
        )
        payload = set()
        parents = set()
        for table, rows in self.delta.items():
            check(isinstance(rows, list), "Calendar table must be an array")
            for row in rows:
                check(
                    isinstance(row, dict)
                    and set(row) == set(BASELINE_TABLE_COLUMNS[table]),
                    "Calendar requires complete native rows without operation fields",
                )
                self.validate_row(table, row)
                ident = key(table, row)
                check(ident not in payload, "Duplicate Calendar payload key")
                payload.add(ident)
                if table == "events":
                    parents.add((row["calendar_id"], row["id"]))
        for table in CHILDREN:
            for row in self.delta.get(table, []):
                check(
                    (row["calendar_id"], row["event_id"]) in parents,
                    "Calendar children require a complete parent event replacement",
                )
        inventory = set()
        for change in task["changes"]:
            if change["service"] == "calendar":
                row = change["source"]["record"]
                table = record_table(row)
                ident = key(table, row)
                check(
                    change["file"] == FILE and ident not in inventory,
                    "Invalid Calendar change inventory",
                )
                inventory.add(ident)
        check(
            payload == inventory,
            "Calendar changes must match every supplied native row",
        )
        check(
            not (
                entries
                or inventory
                or any(e["service"] == "calendar" for e in task["evidence"])
            )
            or self.scope,
            "Calendar task data requires enabled reads",
        )

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def validate_row(self, table, row):
        try:
            self._validate_row(table, row)
        except CalendarEnvironmentError:
            raise
        except (ValueError, KeyError) as error:
            raise CalendarEnvironmentError(
                "Invalid Calendar native row or time zone"
            ) from error

    def _validate_row(self, table, row):
        for field in BASELINE_TABLE_COLUMNS[table]:
            value = row[field]
            if value is None and field in NULLABLE:
                continue
            check(
                type(value) is int if field in INTEGERS else isinstance(value, str),
                "Invalid Calendar native field type: " + field,
            )
            if field in BOOLS:
                check(value in (0, 1), "Invalid Calendar boolean")
            if field in {"sequence", "position", "minutes"}:
                check(value >= 0, "Invalid Calendar nonnegative field")
        if table == "events":
            check(
                bool(row["id"]) and bool(row["calendar_id"]), "Empty Calendar identity"
            )
            check(
                len(row["summary"]) <= 1000
                and len(row["description"]) <= 50000
                and len(row["location"]) <= 2000,
                "Calendar event text exceeds native bounds",
            )
            start, first, day = _event_time(
                row["start_time"], row["timezone"], "start_time"
            )
            end, last, end_day = _event_time(
                row["end_time"], row["timezone"], "end_time"
            )
            check(
                start == row["start_time"]
                and end == row["end_time"]
                and first == row["start_utc"]
                and last == row["end_utc"]
                and day == end_day == bool(row["all_day"])
                and last > first,
                "Inconsistent Calendar time or exclusive end",
            )
            check(
                last - first <= 367 * 86400,
                "Calendar event duration exceeds supported bound",
            )
            check(
                row["status"] in EVENT_STATUSES
                and row["transparency"] in TRANSPARENCIES
                and row["visibility"] in VISIBILITIES
                and row["event_type"] in EVENT_TYPES,
                "Invalid Calendar event enum",
            )
            ZoneInfo(row["timezone"])
            check(
                all(_valid_email(row[f]) for f in ("creator_email", "organizer_email")),
                "Invalid Calendar person email",
            )
            created = timestamp(row["created_at"], "calendar", "created_at")
            updated = timestamp(row["updated_at"], "calendar", "updated_at")
            check(
                created <= updated <= self.snapshot,
                "Calendar observed state is after snapshot or out of order",
            )
            check(
                (row["recurring_event_id"] is None)
                == (row["original_start_time"] is None),
                "Calendar override requires parent and original start",
            )
            if row["original_start_time"] is not None:
                _, _, original_day = _event_time(
                    row["original_start_time"], row["timezone"], "original_start_time"
                )
                check(original_day == day, "Calendar override changes all-day type")
        elif table == "attendees":
            check(
                _valid_email(row["email"])
                and row["response_status"] in RESPONSE_STATUSES,
                "Invalid Calendar attendee",
            )
        elif table == "reminders":
            check(
                row["method"] in REMINDER_METHODS
                and row["minutes"] <= CalendarConfig.max_reminder_minutes,
                "Invalid Calendar reminder",
            )
        elif table == "recurrence_rules":
            check(
                isinstance(row["rule"], str)
                and len(row["rule"]) <= 1000
                and row["rule"].startswith(
                    ("RRULE:", "RDATE:", "EXDATE:", "RDATE;", "EXDATE;")
                ),
                "Invalid Calendar recurrence line",
            )

    def open(self):
        if self.tools is not None:
            return
        root = self.acquire()
        self.work.mkdir(parents=True)
        seed = self.work / "seed.json"
        seed.write_bytes(_encoded(self.delta))
        build_calendar_overlay_from_seed(seed, self.work / "overlay.sqlite")
        try:
            self.tools = CalendarTools(
                CalendarConfig(
                    baseline_dir=Path(root),
                    overlay_dir=self.work,
                    agent_email=self.email,
                    expand_recurring=True,
                )
            )
            db = self.tools.connection
            for table, columns in BASELINE_TABLE_COLUMNS.items():
                present = {
                    r["name"]
                    for r in db.execute(f"PRAGMA baseline.table_info({table})")
                }
                check(set(columns) <= present, "Calendar baseline schema is incomplete")
            identity = db.execute(
                "SELECT * FROM baseline.users WHERE email=? COLLATE NOCASE",
                (self.email,),
            ).fetchone()
            expected = (
                self.delegation["delegated_user_id"] if self.delegation else "U_AGENT"
            )
            check(
                identity is not None
                and identity["user_id"] == expected
                and identity["active"] == 1,
                "Calendar identity must match the active pinned benchmark or delegated source principal",
            )
            if self.delegation:
                check(
                    db.execute(
                        "SELECT 1 FROM baseline.calendar_acl WHERE user_email=? COLLATE NOCASE AND role IN ('freeBusyReader','reader','writer','owner')",
                        (self.email,),
                    ).fetchone(),
                    "Delegated Calendar principal has no source read grants",
                )
            db.execute(
                f"CREATE TEMP VIEW effective_events AS WITH {EFFECTIVE_EVENTS_CTE} SELECT * FROM effective_events"
            )
            for table in CHILDREN:
                columns = ",".join("c." + f for f in BASELINE_TABLE_COLUMNS[table])
                db.execute(
                    f"""CREATE TEMP VIEW effective_{table} AS
                    SELECT {columns} FROM main.{table} c JOIN effective_events e ON e.calendar_id=c.calendar_id AND e.id=c.event_id AND e.source='overlay'
                    UNION ALL SELECT {columns} FROM baseline.{table} c JOIN effective_events e ON e.calendar_id=c.calendar_id AND e.id=c.event_id AND e.source='baseline' """
                )
            for change in self.task["changes"]:
                if change["service"] != "calendar":
                    continue
                record = change["source"]["record"]
                table = record_table(record)
                original = self.one(table, record, "baseline")
                check(
                    (original is not None) == (change["operation"] == "replace"),
                    "Calendar add/replace disagrees with baseline",
                )
            for table in OVERLAY_TABLE_COLUMNS:
                for row in db.execute(f"SELECT * FROM effective_{table}"):
                    row = dict(row)
                    self.validate_row(table, row)
                    if table == "events":
                        check(
                            db.execute(
                                "SELECT 1 FROM baseline.calendars WHERE id=?",
                                (row["calendar_id"],),
                            ).fetchone(),
                            "Unknown Calendar reference",
                        )
                        if row["recurring_event_id"]:
                            parent = self.one(
                                "events",
                                dict(
                                    calendar_id=row["calendar_id"],
                                    id=row["recurring_event_id"],
                                ),
                            )
                            check(
                                parent is not None
                                and parent["recurring_event_id"] is None
                                and parent["id"] != row["id"]
                                and parent["all_day"] == row["all_day"],
                                "Invalid Calendar recurrence parent",
                            )
                    else:
                        check(
                            self.one(
                                "events",
                                dict(
                                    calendar_id=row["calendar_id"], id=row["event_id"]
                                ),
                            )
                            is not None,
                            "Calendar child has no effective event",
                        )
                keys = ",".join(PRIMARY_KEYS[table])
                check(
                    db.execute(
                        f"SELECT 1 FROM effective_{table} GROUP BY {keys} HAVING COUNT(*)>1 LIMIT 1"
                    ).fetchone()
                    is None,
                    "Duplicate effective Calendar key",
                )
            for table, cap in (
                ("attendees", 200),
                ("recurrence_rules", 20),
                ("reminders", 10),
            ):
                check(
                    db.execute(
                        f"SELECT 1 FROM effective_{table} GROUP BY calendar_id,event_id HAVING COUNT(*)>? LIMIT 1",
                        (cap,),
                    ).fetchone()
                    is None,
                    "Calendar child collection exceeds native bound",
                )
            check(
                db.execute(
                    "SELECT 1 FROM effective_attendees GROUP BY calendar_id,event_id,email COLLATE NOCASE HAVING COUNT(*)>1 LIMIT 1"
                ).fetchone()
                is None,
                "Duplicate Calendar attendee email",
            )
            check(
                db.execute(
                    "SELECT 1 FROM effective_events WHERE recurring_event_id IS NOT NULL GROUP BY calendar_id,recurring_event_id,original_start_time HAVING COUNT(*)>1 LIMIT 1"
                ).fetchone()
                is None,
                "Duplicate Calendar original occurrence",
            )
            check(
                db.execute(
                    "SELECT 1 FROM effective_events GROUP BY id HAVING COUNT(*)>1 LIMIT 1"
                ).fetchone()
                is None,
                "Calendar event IDs must be globally unique",
            )
            db.execute("PRAGMA query_only=ON")
            for evidence in self.task["evidence"]:
                if evidence["service"] == "calendar":
                    self.exact(evidence["source"]["record"], "effective")
        except BaseException as error:
            self.close()
            if isinstance(error, (ValueError, KeyError)) and not isinstance(
                error, CalendarEnvironmentError
            ):
                raise CalendarEnvironmentError(
                    "Invalid Calendar database or identity"
                ) from error
            raise

    def one(self, table, record, view="effective"):
        selected = f"baseline.{table}" if view == "baseline" else f"effective_{table}"
        return self.tools.connection.execute(
            f"SELECT * FROM {selected} WHERE "
            + " AND ".join(f"{k}=?" for k in PRIMARY_KEYS[table]),
            [record[k] for k in PRIMARY_KEYS[table]],
        ).fetchone()

    def exact(self, record, view="effective"):
        table = record_table(record)
        self.open()
        row = self.one(table, record, view)
        if row is None:
            return None
        event = (
            dict(row)
            if table == "events"
            else dict(
                self.one(
                    "events",
                    dict(calendar_id=row["calendar_id"], id=row["event_id"]),
                    view,
                )
            )
        )
        calendar = self.tools._resolve_calendar(
            event["calendar_id"], minimum_role="reader"
        )
        event.update(
            source=(
                "baseline"
                if view == "baseline"
                else self.one("events", event)["source"]
            ),
            access_role=calendar["access_role"],
        )
        check(
            "calendar_get_event" in self.scope and self.tools._can_see_details(event),
            "Exact Calendar evidence is inaccessible or private",
        )
        detail = public(self.tools._event_result(event))
        check(
            len(_encoded(detail)) <= self.limit,
            "Exact Calendar detail exceeds result limit",
        )
        return {f: row[f] for f in BASELINE_TABLE_COLUMNS[table]}

    def call(self, name, arguments):
        if name not in self.scope:
            return dict(
                ok=False, error="unknown_tool", message="Unknown or disallowed tool"
            )
        if not arguments_valid(name, arguments):
            return dict(
                ok=False,
                error="invalid_arguments",
                message="Invalid Calendar arguments",
            )
        self.open()
        try:
            result = public(getattr(self.tools, name)(**arguments))
        except CalendarToolError as error:
            return dict(
                ok=False,
                error=error.code,
                message="Calendar read failed: " + error.code,
            )
        if len(_encoded(result)) > self.limit:
            return dict(
                ok=False,
                error="result_too_large",
                message="Calendar result exceeds the result byte limit",
            )
        return result

    def close(self):
        if self.tools is not None:
            self.tools.close()
            self.tools = None
