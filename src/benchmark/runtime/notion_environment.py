"""Read-only native Notion state in the shared worker."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from fractions import Fraction
from pathlib import Path, PurePosixPath

from benchmark.prompts import load_text
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/notion/seed.json"
PAGE_FIELDS = (
    "id",
    "title",
    "parent_page_id",
    "database_id",
    "content_path",
    "properties_json",
    "created_time",
    "last_edited_time",
    "archived",
    "content",
)
REVISION_FIELDS = (
    "id",
    "page_id",
    "edited_time",
    "editor_id",
    "summary",
    "reason",
    "before_ref",
    "after_ref",
)
VERSION_FIELDS = ("page_id", "version", "edited_time", "content")
KEYS = {"pages": ("id",), "revisions": ("id",), "versions": ("page_id", "version")}
S = {"type": "string"}
N = {"type": ["string", "null"]}
PAGING = {
    "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
    "start_cursor": N,
}
BODY = {
    "start_cursor": N,
    "max_chars": {"type": "integer", "minimum": 4, "maximum": 65536},
}
NOTION_READS = {
    "notion_search": (
        {"query": S},
        {"page_size": PAGING["page_size"], "page_id": N, "database_id": N},
    ),
    "notion_fetch_page": ({"page_id": S}, BODY),
    "notion_query_database": (
        {"database_id": S},
        {
            **PAGING,
            "filter": {"type": ["object", "null"]},
            "sorts": {"type": ["array", "null"], "items": {"type": "object"}},
        },
    ),
    "notion_list_catalog": (
        {},
        {
            **PAGING,
            "kind": {"type": "string", "enum": ["page", "database"]},
            "parent_page_id": N,
        },
    ),
    "notion_list_history": (
        {"page_id": S},
        {**PAGING, "kind": {"type": "string", "enum": ["revisions", "versions"]}},
    ),
    "notion_fetch_version": ({"page_id": S, "version": S}, BODY),
}


class NotionEnvironmentError(ValueError):
    def __init__(self, reason, *, code="invalid_notion_environment", record=None):
        self.code, self.record, self.source, self.validation_reason = (
            code,
            record,
            "notion",
            reason,
        )
        super().__init__(reason)


def check(condition, reason, code="invalid_notion_environment"):
    if not condition:
        raise NotionEnvironmentError(reason, code=code)


def instant(value, file="notion", field="time"):
    """Exact observed instants, including fractions smaller than a microsecond."""
    try:
        check(
            isinstance(value, str)
            and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})",
                value,
            ),
            "Explicit ISO timestamp with timezone required",
        )
        parsed = timestamp(value, file, field)
        seconds = (
            parsed.toordinal() * 86400
            + parsed.hour * 3600
            + parsed.minute * 60
            + parsed.second
        )
        fraction = re.match(r"[.,](\d+)", value[19:])
        offset = parsed.utcoffset()
        return (
            Fraction(seconds)
            + (Fraction("0." + fraction[1]) if fraction else 0)
            - (offset.days * 86400 + offset.seconds)
        )
    except (ValueError, TypeError):
        raise NotionEnvironmentError(
            "Explicit valid ISO timestamp with timezone required"
        ) from None


def properties(value):
    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate property key")
            result[key] = item
        return result

    def nonfinite(_):
        raise ValueError("Nonfinite property value")

    try:
        result = json.loads(
            value, object_pairs_hook=object_pairs, parse_constant=nonfinite
        )
    except (ValueError, TypeError):
        raise NotionEnvironmentError("Invalid native properties_json") from None
    check(isinstance(result, dict), "Native properties_json must contain an object")
    return result


def native_id(value):
    try:
        check(
            isinstance(value, str) and str(uuid.UUID(value)) == value,
            "Canonical native UUID required",
        )
    except (ValueError, AttributeError):
        raise NotionEnvironmentError("Canonical native UUID required") from None


def version_path(page_id, version):
    native_id(page_id)
    check(
        isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", version),
        "Invalid version",
    )
    return f"versions/{page_id}/{version}.md"


def relative_path(value):
    check(
        isinstance(value, str)
        and value
        and "\\" not in value
        and not PurePosixPath(value).is_absolute()
        and all(p not in ("", ".", "..") for p in value.split("/")),
        "Unsafe Notion path",
    )
    return value


def record_key(record):
    check(
        isinstance(record, dict) and record.get("table") in KEYS,
        "Exact Notion record key required",
    )
    table = record["table"]
    check(set(record) == {"table", *KEYS[table]}, "Exact Notion record fields required")
    for field in KEYS[table]:
        if field == "version":
            version_path(record["page_id"], record[field])
        else:
            native_id(record[field])
    return table


def definitions(scope, cap):
    return [
        dict(
            name=name,
            description=load_text("tools/notion/" + name + ".txt")
            + load_text("tools/notion/result_cap.txt").format(max_result_bytes=cap),
            parameters=dict(
                type="object",
                properties={**required, **optional},
                required=list(required),
                additionalProperties=False,
            ),
        )
        for name, (required, optional) in NOTION_READS.items()
        if name in scope
    ]


def arguments_valid(name, args):
    required, optional = NOTION_READS[name]
    fields = {**required, **optional}
    if (
        not isinstance(args, dict)
        or not required.keys() <= args.keys()
        or args.keys() - fields.keys()
    ):
        return False
    for key, value in args.items():
        schema = fields[key]
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        kind = {
            str: "string",
            int: "integer",
            list: "array",
            dict: "object",
            type(None): "null",
        }.get(type(value))
        if kind not in kinds or ("enum" in schema and value not in schema["enum"]):
            return False
        if kind == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
            return False
        if kind == "array" and any(not isinstance(v, dict) for v in value):
            return False
    return True


class NotionEnvironment:
    def __init__(self, package, manifest, task, config, work_dir, cap, acquire):
        self.scope = set(config["tool_scope"]) & NOTION_READS.keys()
        self.task, self.limit, self.acquire = task, cap, acquire
        self.snapshot = instant(config["snapshot_time"], "config.json", "snapshot_time")
        self.work = Path(work_dir).resolve() / "notion"
        self.tools = None
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/notion/")
        ]
        check(
            all(f["path"] == FILE and f["bytes"] <= 8388608 for f in entries),
            "Only a small Notion seed.json is supported",
        )
        self.delta = load_json(Path(package) / FILE) if entries else {}
        check(
            isinstance(self.delta, dict) and not self.delta.keys() - KEYS.keys(),
            "Unsupported Notion delta collections",
        )
        self.rows = {}
        for table, rows in self.delta.items():
            check(isinstance(rows, list), "Notion collection must be an array")
            for row in rows:
                fields = dict(
                    pages=PAGE_FIELDS,
                    revisions=REVISION_FIELDS,
                    versions=VERSION_FIELDS,
                )[table]
                check(
                    isinstance(row, dict) and set(row) == set(fields),
                    "Complete native Notion rows required",
                )
                self.validate_row(table, row)
                key = (table, *(row[k] for k in KEYS[table]))
                check(key not in self.rows, "Duplicate Notion delta identity")
                self.rows[key] = row
        inventory = {}
        for change in task["changes"]:
            if change["service"] != "notion":
                continue
            record = change["source"]["record"]
            table = record_key(record)
            key = (table, *(record[k] for k in KEYS[table]))
            check(
                change["file"] == FILE and key not in inventory,
                "Invalid Notion change inventory",
            )
            inventory[key] = change
        check(
            set(inventory) == set(self.rows),
            "Notion payload and task changes must match exactly",
        )
        self.inventory = inventory
        check(
            self.scope
            or not (
                entries
                or inventory
                or any(e["service"] == "notion" for e in task["evidence"])
            ),
            "Notion task data requires enabled reads",
        )

    @property
    def tool_definitions(self):
        return definitions(self.scope, self.limit)

    def validate_row(self, table, row):
        for field in KEYS[table]:
            if field != "version":
                native_id(row[field])
        if table == "versions":
            version_path(row["page_id"], row["version"])
            check(
                isinstance(row["content"], str),
                "Version requires full Markdown content",
            )
            check(
                instant(row["edited_time"], "notion", "edited_time") <= self.snapshot,
                "Version is after snapshot",
                "state_after_snapshot",
            )
            return
        if table == "pages":
            check(
                isinstance(row["title"], str) and row["title"].strip(),
                "Page title required",
            )
            check(
                isinstance(row["content"], str)
                and isinstance(row["properties_json"], str),
                "Full body and native properties required",
            )
            properties(row["properties_json"])
            check(row["archived"] is False, "Task deltas cannot archive pages")
            for field in ("parent_page_id", "database_id"):
                if row[field] is not None:
                    native_id(row[field])
            check(
                not (row["parent_page_id"] and row["database_id"]),
                "Page has two parents",
            )
            relative_path(row["content_path"])
            check(
                row["content_path"] == f"pages/{row['id']}.md",
                "Task body must use pages/<native-id>.md",
            )
            times = [
                instant(row[k], "notion", k)
                for k in ("created_time", "last_edited_time")
            ]
            check(
                times[0] <= times[1] <= self.snapshot,
                "Page times must be explicit and at or before snapshot",
                "state_after_snapshot",
            )
        else:
            native_id(row["page_id"])
            check(
                row["editor_id"] is None
                or isinstance(row["editor_id"], str)
                and row["editor_id"],
                "Invalid editor identity",
            )
            check(
                all(isinstance(row[k], str) for k in ("summary", "reason")),
                "Revision summary and reason required",
            )
            check(
                instant(row["edited_time"], "notion", "edited_time") <= self.snapshot,
                "Revision is after snapshot",
                "state_after_snapshot",
            )
            for field in ("before_ref", "after_ref"):
                if row[field] is not None:
                    relative_path(row[field])
                    check(
                        re.fullmatch(
                            r"versions/"
                            + re.escape(row["page_id"])
                            + r"/[A-Za-z0-9_-]{1,128}\.md",
                            row[field],
                        ),
                        "Revision version reference has a different page or unsafe path",
                    )

    def files(self, source):
        if source == "overlay":
            return [
                p.relative_to(self.work).as_posix()
                for p in self.work.rglob("*")
                if p.is_file()
            ]
        return self.acquire(inventory=True)

    def acquire_file(self, source, relative):
        relative_path(relative)
        root = (
            self.work
            if source == "overlay"
            else Path(self.acquire(("notion/" + relative,))).resolve()
        )
        path = root / relative
        check(
            path.is_file()
            and path.resolve().is_relative_to(root.resolve())
            and not any(p.is_symlink() for p in (path, *path.parents))
            and path.stat().st_nlink == 1,
            "Missing or unsafe linked Notion file",
        )
        return path

    def open(self):
        if self.tools is not None:
            return
        from benchmark.tools.notion.notion import (
            NotionConfig,
            NotionTools,
            NotionToolError,
            build_overlay_catalog,
        )

        baseline = self.acquire()
        self.work.mkdir(parents=True)
        config = replace(
            NotionConfig.from_env(),
            baseline_dir=baseline,
            overlay_dir=self.work,
            overlay_runtime_dir=self.work / "indexes",
            embedding_local_files_only=True,
        )
        from .shared_source import derived_directory

        from benchmark.dataset import cache_at
        from benchmark.search_assets import check_search, search_pin

        with cache_at(derived_directory("asset-verification")) as cache:
            model_path = check_search(baseline.parent, cache=cache)
        config = replace(
            config,
            runtime_dir=derived_directory("notion-indexes"),
            embedding_model=search_pin()["model"],
            embedding_model_path=model_path,
        )
        for row in self.delta.get("pages", []):
            path = self.work / row["content_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row["content"], encoding="utf-8")
        seed = self.work / "seed.json"
        seed.write_bytes(
            _encoded(
                {
                    "pages": [
                        {
                            **{
                                k: v
                                for k, v in r.items()
                                if k not in ("content", "properties_json")
                            },
                            "properties": properties(r["properties_json"]),
                        }
                        for r in self.delta.get("pages", [])
                    ]
                }
            )
        )
        try:
            build_overlay_catalog(
                seed,
                self.work / "overlay.sqlite",
                self.work,
                config,
                read_only=False,
                index_content=True,
            )
        except NotionToolError as error:
            raise NotionEnvironmentError(error.message, code=error.code) from error
        for row in self.delta.get("versions", []):
            path = self.work / version_path(row["page_id"], row["version"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row["content"], encoding="utf-8")
        if self.delta.get("revisions"):
            (self.work / "revisions.jsonl").write_text(
                "".join(
                    json.dumps(r, ensure_ascii=False) + "\n"
                    for r in self.delta["revisions"]
                ),
                encoding="utf-8",
            )
        try:
            self.tools = NotionTools(
                config, acquire_file=self.acquire_file, file_inventory=self.files
            )
            self.tools._connection.execute("PRAGMA query_only=ON")
            self.validate_effective()
        except BaseException:
            self.close()
            raise

    def page_row(self, page_id, view="effective"):
        if view == "baseline":
            return self.tools._connection.execute(
                "SELECT *, 'baseline' AS source FROM baseline.pages WHERE id=? AND operation='upsert' AND archived=0",
                (page_id,),
            ).fetchone()
        return self.tools._effective_page(page_id)

    def validate_effective(self):
        db = self.tools._connection
        for key, row in self.rows.items():
            table = key[0]
            if table == "pages":
                old = db.execute(
                    "SELECT * FROM baseline.pages WHERE id=?", (row["id"],)
                ).fetchone()
                if old:
                    check(
                        old["operation"] == "upsert" and not old["archived"],
                        "Task changes cannot restore archived pages",
                    )
                    check(
                        all(
                            old[k] == row[k]
                            for k in (
                                "id",
                                "parent_page_id",
                                "database_id",
                                "created_time",
                            )
                        ),
                        "Replacement changes native identity or parent",
                    )
                    check(
                        instant(row["last_edited_time"], "notion", "last_edited_time")
                        >= instant(
                            old["last_edited_time"], "notion", "last_edited_time"
                        ),
                        "Replacement moves page backwards in time",
                    )
                for target in re.findall(
                    r"notion://page/([A-Za-z0-9-]+)", row["content"]
                ):
                    check(
                        self.page_row(target) is not None, "Missing linked Notion page"
                    )
                if row["database_id"]:
                    check(
                        self.tools._database_exists(row["database_id"]),
                        "Missing database parent",
                    )
                visited = {row["id"]}
                parent = row["parent_page_id"]
                while parent:
                    check(parent not in visited, "Notion hierarchy cycle")
                    visited.add(parent)
                    p = self.page_row(parent)
                    check(p is not None, "Missing page parent")
                    parent = p["parent_page_id"]
                for prop in properties(row["properties_json"]).values():
                    if isinstance(prop, dict) and prop.get("type") == "relation":
                        check(
                            isinstance(prop.get("relation"), list),
                            "Invalid relation property",
                        )
                        for target in prop["relation"]:
                            check(
                                isinstance(target, dict)
                                and self.page_row(target.get("id")) is not None,
                                "Missing relation target",
                            )
            elif table == "revisions":
                check(self.page_row(row["page_id"]) is not None, "Revision has no page")
                old = next(
                    (
                        r
                        for r in self.tools.history_rows(None, "baseline")
                        if r.get("id") == row["id"]
                    ),
                    None,
                )
                if old:
                    check(
                        old["page_id"] == row["page_id"],
                        "Revision replacement changes page",
                    )
                page = self.page_row(row["page_id"])
                check(
                    instant(page["created_time"], "notion", "created_time")
                    <= instant(row["edited_time"], "notion", "edited_time")
                    <= instant(page["last_edited_time"], "notion", "last_edited_time"),
                    "Revision time outside page lifetime",
                )
                for field in ("before_ref", "after_ref"):
                    if row[field]:
                        check(
                            row[field]
                            in set(self.files("baseline")) | set(self.files("overlay")),
                            "Missing referenced stored version",
                        )
                current = self.rows.get(("pages", row["page_id"]))
                if (
                    current
                    and row["after_ref"]
                    and instant(row["edited_time"])
                    == instant(current["last_edited_time"])
                ):
                    version = row["after_ref"].rsplit("/", 1)[-1][:-3]
                    stored = self.rows.get(("versions", row["page_id"], version))
                    if stored:
                        check(
                            stored["content"] == current["content"]
                            and instant(stored["edited_time"])
                            == instant(row["edited_time"]),
                            "Current revision, stored version and page body disagree",
                        )
            else:
                check(self.page_row(row["page_id"]) is not None, "Version has no page")
                p = self.page_row(row["page_id"])
                check(
                    instant(p["created_time"], "notion", "created_time")
                    <= instant(row["edited_time"], "notion", "edited_time")
                    <= instant(p["last_edited_time"], "notion", "last_edited_time"),
                    "Version time outside page lifetime",
                )
                old = version_path(row["page_id"], row["version"]) in self.files(
                    "baseline"
                )
            check(
                bool(old) == (self.inventory[key]["operation"] == "replace"),
                "Add/replace conflicts with baseline existence",
            )
        # References and observed times are checked without loading any page body.
        from benchmark.tools.notion.notion import EFFECTIVE_CTES

        for row in db.execute("SELECT * FROM baseline.databases"):
            times = [
                instant(row[k], "notion", k)
                for k in ("created_time", "last_edited_time")
            ]
            check(
                times[0] <= times[1] <= self.snapshot,
                "Stored database state is after snapshot",
                "state_after_snapshot",
            )
            if row["parent_page_id"]:
                check(
                    self.page_row(row["parent_page_id"]) is not None,
                    "Stored database parent is missing",
                )
        for row in db.execute(EFFECTIVE_CTES + " SELECT * FROM effective_pages"):
            times = [
                instant(row[k], "notion", k)
                for k in ("created_time", "last_edited_time")
            ]
            check(
                times[0] <= times[1] <= self.snapshot,
                "Stored page state is after snapshot",
                "state_after_snapshot",
            )
            if row["content_path"]:
                relative_path(row["content_path"])
        for e in self.task["evidence"]:
            if e["service"] == "notion":
                record = e["source"].get("record")
                table = record_key(record)
                needed = dict(
                    pages="notion_fetch_page",
                    revisions="notion_list_history",
                    versions="notion_fetch_version",
                )[table]
                check(
                    needed in self.scope, "Declared evidence requires its public read"
                )
                if table == "pages":
                    row = self.page_row(record["id"])
                    check(row is not None, "Declared page evidence is missing")
                    if row["content_path"]:
                        check(
                            row["content_path"] in self.files(row["source"]),
                            "Declared page body is missing",
                        )
                elif table == "versions":
                    check(
                        version_path(record["page_id"], record["version"])
                        in set(self.files("baseline")) | set(self.files("overlay")),
                        "Declared version is missing",
                    )
                else:
                    check(
                        "revisions.jsonl"
                        in set(self.files("baseline")) | set(self.files("overlay")),
                        "Declared history is missing",
                    )
                # Exact preflight validates the row/body when it is inspected.
                # Hidden evidence keys must not cause speculative body reads.

    def exact(self, record, view):
        table = record_key(record)
        needed = dict(
            pages="notion_fetch_page",
            revisions="notion_list_history",
            versions="notion_fetch_version",
        )[table]
        check(needed in self.scope, "Exact evidence needs the configured public read")
        self.open()
        if table == "pages":
            row = self.page_row(record["id"], view)
            if row is None:
                return None
            result = self.tools._page_from_row(row, include_content=False)
            if row["content_path"]:
                result.update(
                    self.tools.read_body(
                        row["source"],
                        row["content_path"],
                        row["id"],
                        max_chars=min(65536, max(4, self.limit // 8)),
                    )
                )
            else:
                result.update(content="", has_more=False, next_cursor=None)
            return result
        if table == "revisions":
            for row in self.tools.history_rows(None, view):
                if row.get("id") == record["id"]:
                    self.validate_history_row(row)
                    return row
            return None
        relative = version_path(record["page_id"], record["version"])
        source = (
            "overlay"
            if view == "effective" and (self.work / relative).is_file()
            else "baseline"
        )
        if relative not in self.files(source):
            return None
        return dict(
            page_id=record["page_id"],
            version=record["version"],
            **self.tools.read_body(
                source,
                relative,
                record["page_id"] + "/" + record["version"],
                max_chars=min(65536, max(4, self.limit // 8)),
            ),
        )

    def validate_history_row(self, row):
        check(
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and isinstance(row.get("page_id"), str),
            "Malformed stored revision",
        )
        check(
            instant(row["edited_time"], "notion", "edited_time") <= self.snapshot,
            "Stored revision is after snapshot",
            "state_after_snapshot",
        )
        for field in ("before_ref", "after_ref"):
            if row.get(field):
                relative_path(row[field])
                check(
                    row[field].startswith("versions/" + row["page_id"] + "/"),
                    "Stored revision references another page",
                )

    def call(self, name, args):
        if name not in self.scope:
            return dict(
                ok=False, error="unknown_tool", message="Unknown or disallowed tool"
            )
        if not arguments_valid(name, args):
            return dict(
                ok=False, error="invalid_arguments", message="Invalid Notion arguments"
            )
        self.open()
        from benchmark.tools.notion.notion import NotionToolError, _normalize_id

        try:
            args = dict(args)
            if name == "notion_fetch_page":
                page_id = _normalize_id(args.pop("page_id"))
                row = self.page_row(page_id)
                if row is None:
                    raise NotionToolError("not_found", "Page not found")
                result = self.tools._page_from_row(row, include_content=False)
                if row["content_path"]:
                    args["max_chars"] = min(
                        args.get("max_chars", 16000), max(4, self.limit // 8)
                    )
                    result.update(
                        self.tools.read_body(
                            row["source"], row["content_path"], page_id, **args
                        )
                    )
                else:
                    result.update(content="", has_more=False, next_cursor=None)
            else:
                if name == "notion_fetch_version":
                    args["max_chars"] = min(
                        args.get("max_chars", 16000), max(4, self.limit // 8)
                    )
                result = getattr(self.tools, name)(**args)
            if (
                name == "notion_list_history"
                and args.get("kind", "revisions") == "revisions"
            ):
                for row in result["results"]:
                    self.validate_history_row(row)
        except NotionToolError as error:
            return error.as_dict()
        except (TypeError, KeyError, ValueError) as error:
            if isinstance(error, NotionEnvironmentError):
                raise
            return dict(
                object="error",
                code="invalid_request",
                message="Invalid Notion request or stored content",
            )
        if len(_encoded(result)) > self.limit:
            return dict(
                ok=False,
                error="result_too_large",
                message="Notion result exceeds the cap. Reduce page_size or max_chars.",
            )
        return result

    def close(self):
        if self.tools is not None:
            self.tools.close()
            self.tools = None
