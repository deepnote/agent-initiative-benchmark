"""Read-only Stripe task state, compiled by the native SQLite builder."""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmark.prompts import load_text
from benchmark.tools.stripe.stripe import (
    TABLE_COLUMNS,
    BOOLEAN_COLUMNS,
    JSON_COLUMNS,
    SUBSCRIPTION_STATUSES,
    INVOICE_STATUSES,
    PAYMENT_STATUSES,
    StripeConfig,
    StripeTools,
    StripeToolError,
    build_stripe_overlay_from_seed,
    _effective_cte,
)
from .contracts import load_json, timestamp
from .slack_environment import _encoded

FILE = "delta/stripe/seed.json"
NULLABLE = {
    "customers": {"hubspot_company_id"},
    "subscriptions": {"trial_end", "canceled_at", "ended_at", "latest_invoice_id"},
    "invoices": {
        "subscription_id",
        "due_date",
        "paid_at",
        "next_payment_attempt",
        "hosted_invoice_url",
    },
    "payments": {
        "customer_id",
        "invoice_id",
        "failure_code",
        "failure_message",
        "receipt_email",
        "latest_charge_id",
        "canceled_at",
    },
    "refunds": {"reason"},
}
INTEGERS = BOOLEAN_COLUMNS | {
    "balance",
    "unit_amount",
    "recurring_interval_count",
    "quantity",
    "subtotal",
    "total",
    "amount_due",
    "amount_paid",
    "amount_remaining",
    "attempt_count",
    "amount",
    "amount_received",
}
REFERENCES = {
    "product_id": "products",
    "price_id": "prices",
    "customer_id": "customers",
    "subscription_id": "subscriptions",
    "invoice_id": "invoices",
    "latest_invoice_id": "invoices",
    "payment_id": "payments",
}
OBJECT_TABLES = {
    "customer": "customers",
    "product": "products",
    "price": "prices",
    "subscription": "subscriptions",
    "invoice": "invoices",
    "payment_intent": "payments",
    "refund": "refunds",
}
OBSERVED = {
    "created_at",
    "updated_at",
    "paid_at",
    "canceled_at",
    "ended_at",
    "current_period_start",
    "period_start",
}
TIMES = OBSERVED | {
    "current_period_end",
    "trial_end",
    "due_date",
    "next_payment_attempt",
    "period_end",
}
TEXT = lambda: {"type": ["string", "null"]}
PAGE = {
    "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
    "start_cursor": TEXT(),
}
DATES = {"created_after": TEXT(), "created_before": TEXT()}
READS = {
    "stripe_search_customers": ({}, {"query": {"type": "string"}, **PAGE}),
    "stripe_list_subscriptions": (
        {},
        {**{k: TEXT() for k in ("customer_id", "status", "price_id")}, **PAGE},
    ),
    "stripe_list_invoices": (
        {},
        {
            **{k: TEXT() for k in ("customer_id", "subscription_id", "status")},
            **DATES,
            **PAGE,
        },
    ),
    "stripe_list_payments": (
        {},
        {
            **{k: TEXT() for k in ("customer_id", "invoice_id", "status")},
            **DATES,
            **PAGE,
        },
    ),
    "stripe_list_events": (
        {},
        {
            **{k: TEXT() for k in ("event_type", "object_type", "object_id")},
            **DATES,
            **PAGE,
        },
    ),
    "stripe_list_objects": (
        {"table": {"type": "string", "enum": ["products", "prices", "refunds"]}},
        {"product_id": TEXT(), "payment_id": TEXT(), **PAGE},
    ),
    "stripe_get_object": (
        {
            "table": {"type": "string", "enum": list(TABLE_COLUMNS)},
            "object_id": {"type": "string", "minLength": 1},
        },
        {},
    ),
}


class StripeEnvironmentError(ValueError):
    def __init__(self, reason, *, record=None, code="invalid_stripe_environment"):
        self.code, self.record = code, record
        self.source, self.validation_reason = "stripe", reason
        super().__init__(reason)


def record_table(record):
    if (
        not isinstance(record, dict)
        or set(record) != {"table", "id"}
        or not isinstance(record["table"], str)
        or record["table"] not in TABLE_COLUMNS
        or not isinstance(record["id"], str)
        or not record["id"]
    ):
        raise StripeEnvironmentError("Stripe evidence requires an exact table and id")
    return record["table"]


def public(value):
    """Remove known harness markers, including markers inside historical bodies."""
    if isinstance(value, list):
        return [public(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {
        k: public(v)
        for k, v in value.items()
        if k
        not in {
            "source_event_id",
            "generation_event_id",
            "benchmark_task_id",
            "benchmark_origin",
            "harness_origin",
            "record_type",
        }
        and not (
            k == "source"
            and isinstance(v, str)
            and v in {"baseline", "overlay"}
            and "id" in value
        )
    }


def public_row(row):
    result = dict(row)
    for key in JSON_COLUMNS & result.keys():
        result[key] = _encoded(public(json.loads(result[key]))).decode()
    return {k: v for k, v in result.items() if k != "source"}


def tool_definitions(scope, cap):
    return [
        {
            "name": name,
            "description": load_text("tools/stripe/" + name + ".txt")
            + load_text("tools/stripe/result_cap.txt").format(max_result_bytes=cap),
            "parameters": {
                "type": "object",
                "properties": {**required, **optional},
                "required": list(required),
                "additionalProperties": False,
            },
        }
        for name, (required, optional) in READS.items()
        if name in scope
    ]


def valid_arguments(name, args):
    required, optional = READS[name]
    fields = {**required, **optional}
    if (
        not isinstance(args, dict)
        or set(args) - fields.keys()
        or not required.keys() <= args.keys()
    ):
        return False
    for key, value in args.items():
        schema = fields[key]
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        kind = {str: "string", int: "integer", type(None): "null"}.get(type(value))
        if kind not in kinds or ("enum" in schema and value not in schema["enum"]):
            return False
        if kind == "integer" and not schema["minimum"] <= value <= schema["maximum"]:
            return False
    return True


def validate_row(table, row, snapshot):
    def fail(reason):
        raise StripeEnvironmentError(
            f"Stripe {table}/{row.get('id')}: {reason}",
            record={"table": table, "id": row.get("id")},
        )

    for key in TABLE_COLUMNS[table]:
        value = row[key]
        if value is None and key in NULLABLE.get(table, ()):
            continue
        if key in INTEGERS:
            if type(value) is not int or not -(2**63) <= value < 2**63:
                fail(f"{key} requires a native integer")
            if key in BOOLEAN_COLUMNS and value not in (0, 1):
                fail(f"{key} requires 0 or 1")
            if key != "balance" and value < (
                1
                if key in {"quantity", "recurring_interval_count"}
                or (table == "refunds" and key == "amount")
                else 0
            ):
                fail(f"{key} outside native range")
        elif not isinstance(value, str):
            fail(f"{key} requires text")
        if key == "id" or key.endswith("_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}", value):
                fail(f"{key} is invalid")
        if key == "currency" and not re.fullmatch(r"[a-z]{3}", value):
            fail("currency requires its native lowercase code")
        if key in JSON_COLUMNS:
            try:
                parsed = json.loads(
                    value, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))
                )
                if not isinstance(parsed, dict):
                    fail(f"{key} requires a JSON object")
            except (ValueError, TypeError):
                fail(f"{key} requires valid JSON")
        if key in TIMES:
            try:
                instant = timestamp(value, "stripe", key)
            except ValueError:
                fail(f"{key} requires a timestamp with timezone")
            if key in OBSERVED and instant > snapshot:
                fail(f"{key} is after snapshot")
    if row.get("updated_at") and timestamp(
        row["updated_at"], "stripe", "updated_at"
    ) < timestamp(row["created_at"], "stripe", "created_at"):
        fail("updated_at precedes created_at")
    enums = {
        "subscriptions": SUBSCRIPTION_STATUSES,
        "invoices": INVOICE_STATUSES,
        "payments": PAYMENT_STATUSES,
        "refunds": {"pending", "requires_action", "succeeded", "failed", "canceled"},
    }
    if table in enums and row["status"] not in enums[table]:
        fail("invalid native status")
    for key, values in {
        "tax_exempt": {"none", "exempt", "reverse"},
        "recurring_interval": {"day", "week", "month", "year"},
        "usage_type": {"licensed", "metered"},
    }.items():
        if key in row and row[key] not in values:
            fail(f"invalid {key}")


class StripeEnvironment:
    def __init__(
        self, package, manifest, task, config, work_dir, max_result_bytes, acquire
    ):
        self.scope = set(config["tool_scope"]) & READS.keys()
        self.task, self.limit, self.acquire = task, max_result_bytes, acquire
        self.snapshot = timestamp(
            config["snapshot_time"], "config.json", "snapshot_time"
        )
        self.work, self.tools = Path(work_dir) / "stripe", None
        settings = config.get("service_settings", {}).get("stripe", {})
        if not isinstance(settings, dict):
            raise StripeEnvironmentError(
                "service_settings.stripe requires an account settings object"
            )
        self.account = settings.get("account_id")
        if self.scope and (
            set(settings) != {"account_id"}
            or not isinstance(self.account, str)
            or not re.fullmatch(r"acct_[A-Za-z0-9_-]+", self.account)
        ):
            raise StripeEnvironmentError(
                "Enabled Stripe reads require pinned service_settings.stripe.account_id"
            )
        entries = [
            f for f in manifest["files"] if f["path"].startswith("delta/stripe/")
        ]
        if any(f["path"] != FILE or f["bytes"] > 8_388_608 for f in entries):
            raise StripeEnvironmentError("Stripe accepts only seed.json, at most 8 MiB")
        self.delta = load_json(Path(package) / FILE) if entries else {}
        if not isinstance(self.delta, dict) or self.delta.keys() - TABLE_COLUMNS.keys():
            raise StripeEnvironmentError("Unsupported Stripe delta tables")
        payload = set()
        for table, rows in self.delta.items():
            if not isinstance(rows, list):
                raise StripeEnvironmentError("Stripe table payload requires an array")
            for row in rows:
                if not isinstance(row, dict) or set(row) != set(TABLE_COLUMNS[table]):
                    raise StripeEnvironmentError(
                        "Stripe additions and replacements require complete native rows"
                    )
                validate_row(table, row, self.snapshot)
                if row["account_id"] != self.account or (table, row["id"]) in payload:
                    raise StripeEnvironmentError(
                        "Stripe delta has an invalid account or duplicate key"
                    )
                payload.add((table, row["id"]))
        inventory = set()
        for change in task["changes"]:
            if change["service"] != "stripe":
                continue
            record = change["source"]["record"]
            table = record_table(record)
            key = table, record["id"]
            if change["file"] != FILE or key in inventory:
                raise StripeEnvironmentError(
                    "Invalid or duplicate Stripe task inventory"
                )
            inventory.add(key)
        if payload != inventory:
            raise StripeEnvironmentError(
                "Stripe payload and task inventory must match exactly"
            )
        if (
            entries
            or inventory
            or any(e["service"] == "stripe" for e in task["evidence"])
        ) and not self.scope:
            raise StripeEnvironmentError("Stripe task data requires enabled reads")

    @property
    def tool_definitions(self):
        return tool_definitions(self.scope, self.limit)

    def open(self):
        if self.tools is not None:
            return
        baseline_dir = self.acquire()
        self.work.mkdir()
        seed = {"metadata": {}, **{table: [] for table in TABLE_COLUMNS}}
        for table, rows in self.delta.items():
            seed[table] = [
                {
                    **row,
                    **{k: json.loads(row[k]) for k in JSON_COLUMNS & row.keys()},
                    "operation": "upsert",
                }
                for row in rows
            ]
        path = self.work / "seed.json"
        path.write_bytes(_encoded(seed))
        build_stripe_overlay_from_seed(path, self.work / "overlay.sqlite")
        try:
            self.tools = StripeTools(
                StripeConfig(
                    baseline_dir=Path(baseline_dir),
                    overlay_dir=self.work,
                    account_id=self.account,
                    max_result_bytes=self.limit,
                )
            )
            self._validate_effective()
            self._validate_evidence()
        except BaseException as error:
            self.close()
            if isinstance(error, StripeEnvironmentError):
                raise
            raise StripeEnvironmentError("Stripe native database is invalid") from error

    def row(self, table, id, view="effective"):
        select = (
            f"SELECT * FROM baseline.{table}"
            if view == "baseline"
            else f"WITH {_effective_cte(table)} SELECT * FROM effective_{table}"
        )
        raw = self.tools._connection.execute(
            select + " WHERE id=? AND account_id=?", (id, self.account)
        ).fetchone()
        return dict(raw) if raw else None

    def _validate_effective(self):
        db = self.tools._connection
        primary = db.execute(
            "SELECT value FROM baseline.metadata WHERE key='primary_account_id'"
        ).fetchone()
        if primary is not None and json.loads(primary[0]) != self.account:
            raise StripeEnvironmentError(
                "Pinned Stripe account differs from source primary_account_id"
            )
        if not any(
            db.execute(
                f"SELECT 1 FROM baseline.{table} WHERE account_id=? LIMIT 1",
                (self.account,),
            ).fetchone()
            for table in TABLE_COLUMNS
        ):
            raise StripeEnvironmentError(
                "Pinned Stripe account does not exist in the source"
            )
        for change in self.task["changes"]:
            if change["service"] != "stripe":
                continue
            record = change["source"]["record"]
            table, id = record["table"], record["id"]
            original = db.execute(
                f"SELECT * FROM baseline.{table} WHERE id=?", (id,)
            ).fetchone()
            current = self.row(table, id)
            if (original is not None) != (change["operation"] == "replace"):
                raise StripeEnvironmentError(
                    "Stripe add/replace conflicts with source existence"
                )
            stable = {"id", "account_id"} | (
                (REFERENCES.keys() - {"latest_invoice_id"}) & current.keys()
            )
            stable.add("created_at")
            if table == "events":
                stable.update(("object_type", "object_id", "type"))
            if table in {"prices", "invoices", "payments", "refunds"}:
                stable.add("currency")
            if original and any(original[k] != current[k] for k in stable):
                raise StripeEnvironmentError(
                    "Stripe replacement changes account or parent identity"
                )
        self.event_links = {}
        for table in TABLE_COLUMNS:
            for raw in db.execute(
                f"WITH {_effective_cte(table)} SELECT * FROM effective_{table} WHERE account_id=?",
                (self.account,),
            ):
                row = dict(raw)
                validate_row(table, row, self.snapshot)
                for field, parent_table in REFERENCES.items():
                    if row.get(field) is None:
                        continue
                    parent = self.row(parent_table, row[field])
                    if parent is None:
                        raise StripeEnvironmentError(
                            f"Stripe {table}/{row['id']}: missing same-account {field}"
                        )
                    if field == "payment_id" and parent["currency"] != row["currency"]:
                        raise StripeEnvironmentError(
                            "Stripe refund and payment currencies differ"
                        )
                    if (
                        field in {"invoice_id", "subscription_id"}
                        and row.get("customer_id")
                        and parent.get("customer_id") != row["customer_id"]
                    ):
                        raise StripeEnvironmentError(
                            "Stripe linked objects have different customers"
                        )
                    if field == "invoice_id" and parent["currency"] != row["currency"]:
                        raise StripeEnvironmentError(
                            "Stripe invoice and payment currencies differ"
                        )
                if table == "events":
                    link = self._validate_event_reference(row)
                    if link is not None:
                        self.event_links[row["id"]] = link
        duplicate = db.execute(
            f"WITH {_effective_cte('invoices')} SELECT number FROM effective_invoices "
            "WHERE account_id=? GROUP BY number HAVING COUNT(*)>1 LIMIT 1",
            (self.account,),
        ).fetchone()
        if duplicate:
            raise StripeEnvironmentError(
                "Stripe effective invoice numbers must be unique within the account"
            )

    def _validate_event_reference(self, row):
        # Keep accepted historical bodies as stored. Some source events name
        # their event family rather than the current object's table.
        body = json.loads(row["data_json"])
        obj = body.get("object", body)
        target = OBJECT_TABLES.get(row["object_type"])
        object_id = row["object_id"]
        baseline_event = row["id"] not in {
            r["id"] for r in self.delta.get("events", [])
        }
        historical_refund = False
        historical_link = False
        if baseline_event and isinstance(obj, dict):
            if row["object_type"] == "customer" and row["type"].startswith(
                "customer.subscription."
            ):
                target = "subscriptions"
                historical_link = True
            elif row["object_type"] == "charge" and row["type"] == "charge.refunded":
                target = "payments"
                historical_link = True
                historical_refund = obj.get("payment_id") == object_id and str(
                    obj.get("id", "")
                ).startswith("re_")
                if not historical_refund:
                    raise StripeEnvironmentError(
                        "Historical charge refund has no matching payment reference"
                    )
            elif row["object_type"] == "credit_note" and row["type"].startswith(
                "credit_note."
            ):
                target, object_id = "invoices", obj.get("invoice_id")
                historical_link = True
        parent = (
            self.row(target, object_id) if target is not None and object_id else None
        )
        unresolved_allowed = (
            baseline_event
            and row["object_type"] in ("charge", "credit_note")
            and historical_link
            and bool(object_id)
        )
        if target is None or not object_id or parent is None and not unresolved_allowed:
            raise StripeEnvironmentError(
                "Stripe event requires a supported same-account object reference",
                record=dict(
                    table="events",
                    id=row["id"],
                    object_type=row["object_type"],
                    object_id=row["object_id"],
                    target=target,
                    target_id=object_id,
                ),
            )
        if (
            parent is None
            and unresolved_allowed
            and getattr(self, "tools", None) is not None
        ):
            foreign = self.tools._connection.execute(
                f"SELECT 1 FROM baseline.{target} WHERE id=? AND account_id<>? UNION ALL SELECT 1 FROM overlay.{target} WHERE id=? AND account_id<>? LIMIT 1",
                (object_id, self.account, object_id, self.account),
            ).fetchone()
            if foreign:
                raise StripeEnvironmentError(
                    "Stripe historical event reference crosses accounts"
                )
        if (
            isinstance(obj, dict)
            and not historical_refund
            and obj.get("id", row["object_id"]) != row["object_id"]
        ):
            raise StripeEnvironmentError(
                "Stripe historical object ID differs from its event reference"
            )
        self._validate_event_body(body)
        if historical_link:
            return dict(
                table=target,
                id=object_id,
                status="resolved" if parent is not None else "unresolved",
            )

    def _validate_event_body(self, value):
        if isinstance(value, list):
            for item in value:
                self._validate_event_body(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if (
                    key in {"account_id", "account"}
                    and isinstance(item, str)
                    and item.startswith("acct_")
                    and item != self.account
                ):
                    raise StripeEnvironmentError(
                        "Stripe historical event body crosses accounts"
                    )
                if key != "metadata":
                    self._validate_event_body(item)

    def _validate_evidence(self):
        for evidence in self.task["evidence"]:
            if evidence["service"] != "stripe":
                continue
            record = evidence["source"].get("record")
            table = record_table(record)
            if "stripe_get_object" not in self.scope:
                raise StripeEnvironmentError(
                    "Exact Stripe evidence requires stripe_get_object access"
                )
            result = self.call(
                "stripe_get_object", {"table": table, "object_id": record["id"]}
            )
            if not result.get("ok"):
                raise StripeEnvironmentError(
                    "Stripe evidence is missing, inaccessible or oversized"
                )

    def call(self, name, arguments):
        if name not in self.scope:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": "Unknown or disallowed tool",
            }
        if not valid_arguments(name, arguments):
            return {
                "ok": False,
                "error": "invalid_arguments",
                "message": "Invalid Stripe arguments",
            }
        self.open()
        try:
            result = public(getattr(self.tools, name)(**arguments))
            if name == "stripe_list_events":
                for event in result.get("events", []):
                    if event.get("id") in self.event_links:
                        event["linked_record"] = self.event_links[event["id"]]
            if (
                name == "stripe_get_object"
                and arguments.get("table") == "events"
                and result.get("record", {}).get("id") in self.event_links
            ):
                result["record"]["linked_record"] = self.event_links[
                    result["record"]["id"]
                ]
        except StripeToolError as error:
            return error.as_dict()
        if len(_encoded(result)) > self.limit:
            return {
                "ok": False,
                "error": "result_too_large",
                "message": "Stripe result exceeds the result cap",
            }
        return result

    def close(self):
        if self.tools is not None:
            self.tools.close()
            self.tools = None
