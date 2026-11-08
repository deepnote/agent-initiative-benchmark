"""Local Stripe benchmark wrappers backed by two SQLite databases."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import tomllib

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")

    configured: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        configured.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return configured[0], configured[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "stripe"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "stripe"


class StripeToolError(ValueError):
    """A predictable error that can be returned to an agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class StripeConfig:
    """Paths, account scope, and limits for Stripe."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    account_id: str = "acct_pagemantle"
    default_page_size: int = 20
    max_page_size: int = 100
    max_search_characters: int = 500
    sqlite_timeout_seconds: float = 30.0
    max_result_bytes: int | None = None

    @classmethod
    def from_env(cls) -> StripeConfig:
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_STRIPE_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_STRIPE_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            account_id=os.getenv("BENCHMARK_STRIPE_ACCOUNT_ID", cls.account_id),
            default_page_size=int(
                os.getenv("STRIPE_DEFAULT_PAGE_SIZE", cls.default_page_size)
            ),
            max_page_size=int(os.getenv("STRIPE_MAX_PAGE_SIZE", cls.max_page_size)),
            max_search_characters=int(
                os.getenv("STRIPE_MAX_SEARCH_CHARACTERS", cls.max_search_characters)
            ),
            sqlite_timeout_seconds=float(
                os.getenv("STRIPE_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
        )

    def validate(self) -> None:
        if not self.account_id.strip():
            raise ValueError("account_id cannot be empty")
        if self.default_page_size < 1:
            raise ValueError("default_page_size must be positive")
        if self.max_page_size < self.default_page_size:
            raise ValueError("max_page_size must be at least default_page_size")
        if self.max_search_characters < 1:
            raise ValueError("max_search_characters must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")
        if self.max_result_bytes is not None and (
            type(self.max_result_bytes) is not int or self.max_result_bytes < 1
        ):
            raise ValueError("max_result_bytes must be a positive integer")


TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "customers": (
        "id",
        "account_id",
        "name",
        "email",
        "description",
        "currency",
        "balance",
        "delinquent",
        "tax_exempt",
        "hubspot_company_id",
        "created_at",
        "updated_at",
        "metadata_json",
    ),
    "products": (
        "id",
        "account_id",
        "name",
        "description",
        "active",
        "created_at",
        "metadata_json",
    ),
    "prices": (
        "id",
        "account_id",
        "product_id",
        "active",
        "currency",
        "unit_amount",
        "recurring_interval",
        "recurring_interval_count",
        "usage_type",
        "created_at",
        "metadata_json",
    ),
    "subscriptions": (
        "id",
        "account_id",
        "customer_id",
        "status",
        "price_id",
        "quantity",
        "cancel_at_period_end",
        "current_period_start",
        "current_period_end",
        "trial_end",
        "canceled_at",
        "ended_at",
        "latest_invoice_id",
        "created_at",
        "updated_at",
        "metadata_json",
    ),
    "invoices": (
        "id",
        "account_id",
        "customer_id",
        "subscription_id",
        "number",
        "status",
        "currency",
        "subtotal",
        "total",
        "amount_due",
        "amount_paid",
        "amount_remaining",
        "billing_reason",
        "due_date",
        "paid_at",
        "attempt_count",
        "next_payment_attempt",
        "created_at",
        "period_start",
        "period_end",
        "hosted_invoice_url",
        "metadata_json",
    ),
    "payments": (
        "id",
        "account_id",
        "customer_id",
        "invoice_id",
        "status",
        "currency",
        "amount",
        "amount_received",
        "payment_method_type",
        "description",
        "failure_code",
        "failure_message",
        "receipt_email",
        "latest_charge_id",
        "created_at",
        "canceled_at",
        "metadata_json",
    ),
    "refunds": (
        "id",
        "account_id",
        "payment_id",
        "status",
        "currency",
        "amount",
        "reason",
        "created_at",
        "metadata_json",
    ),
    "events": (
        "id",
        "account_id",
        "type",
        "object_type",
        "object_id",
        "api_version",
        "created_at",
        "data_json",
    ),
}

JSON_COLUMNS = {"metadata_json", "data_json"}
BOOLEAN_COLUMNS = {"active", "cancel_at_period_end", "delinquent"}

SUBSCRIPTION_STATUSES = {
    "active",
    "past_due",
    "canceled",
    "incomplete",
    "incomplete_expired",
    "trialing",
    "unpaid",
    "paused",
}
INVOICE_STATUSES = {"draft", "open", "paid", "uncollectible", "void"}
PAYMENT_STATUSES = {
    "requires_payment_method",
    "requires_confirmation",
    "requires_action",
    "processing",
    "requires_capture",
    "canceled",
    "succeeded",
}


def _overlay_sql() -> str:
    statements = [
        "PRAGMA foreign_keys = OFF;",
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    ]
    for table, columns in TABLE_COLUMNS.items():
        definitions = [f'"{column}"' for column in columns]
        definitions[0] += " TEXT PRIMARY KEY"
        for index in range(1, len(definitions)):
            definitions[index] += ""
        definitions.append(
            "operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete'))"
        )
        statements.append(f"CREATE TABLE {table} ({', '.join(definitions)});")
        statements.append(
            f"CREATE INDEX {table}_account_idx ON {table}(account_id, operation, id);"
        )
    return "\n".join(statements)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _seed_value(column: str, value: Any) -> Any:
    if column in JSON_COLUMNS:
        return _json(value if value is not None else {})
    if column in BOOLEAN_COLUMNS and isinstance(value, bool):
        return int(value)
    return value


def _validate_timestamp(value: Any, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")


def _validate_seed(seed: Mapping[str, Any], *, overlay: bool) -> None:
    if not isinstance(seed, Mapping):
        raise TypeError("Stripe seed must be an object")
    expected = set(TABLE_COLUMNS)
    missing = expected - set(seed)
    if missing:
        raise ValueError(f"Stripe seed is missing tables: {', '.join(sorted(missing))}")
    if not isinstance(seed.get("metadata"), dict):
        raise TypeError("Stripe seed metadata must be an object")

    seen: set[str] = set()
    for table, columns in TABLE_COLUMNS.items():
        rows = seed.get(table)
        if not isinstance(rows, list):
            raise TypeError(f"Stripe seed {table} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"Stripe seed {table} rows must be objects")
            missing_columns = set(columns) - set(row)
            if missing_columns:
                raise ValueError(
                    f"Stripe {table} row is missing: {', '.join(sorted(missing_columns))}"
                )
            record_id = row["id"]
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Stripe {table} id must be a non-empty string")
            identity = f"{table}:{record_id}"
            if identity in seen:
                raise ValueError(f"Duplicate Stripe record: {identity}")
            seen.add(identity)
            if overlay and row.get("operation") not in {"upsert", "delete"}:
                raise ValueError(f"Stripe overlay {identity} has an invalid operation")
            for key, value in row.items():
                if key.endswith("_at") and key not in {"tax_exempt"}:
                    _validate_timestamp(value, f"{table}.{key}", optional=True)


def _insert_rows(
    connection: sqlite3.Connection,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    overlay: bool,
    incremental: bool = False,
) -> None:
    columns = list(TABLE_COLUMNS[table]) + (["operation"] if overlay else [])
    placeholders = ",".join("?" for _ in columns)
    names = ",".join(f'"{column}"' for column in columns)
    values = (
        tuple(_seed_value(column, row.get(column)) for column in columns)
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    if incremental:
        from benchmark.sqlite_updates import sync_rows

        sync_rows(connection, table, columns, values)
    else:
        connection.executemany(
            f"INSERT INTO {table} ({names}) VALUES ({placeholders})", values
        )


def build_stripe_overlay_from_seed(
    seed_path: Path, overlay_path: Path
) -> dict[str, int]:
    """Build a small task overlay from a readable seed."""

    seed = json.loads(seed_path.read_text())
    _validate_seed(seed, overlay=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = overlay_path.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA page_size = 4096")
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.executescript(_overlay_sql())
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [(key, _json(value)) for key, value in sorted(seed["metadata"].items())],
        )
        for table in TABLE_COLUMNS:
            _insert_rows(connection, table, seed[table], overlay=True)
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    os.replace(temporary, overlay_path)
    return {table: len(seed[table]) for table in TABLE_COLUMNS}


def _effective_cte(table: str) -> str:
    columns = TABLE_COLUMNS[table]
    selected = ", ".join(f'b."{column}"' for column in columns)
    overlaid = ", ".join(f'o."{column}"' for column in columns)
    return f"""
    effective_{table} AS (
        SELECT {selected}, 'baseline' AS source
        FROM baseline.{table} AS b
        WHERE NOT EXISTS (
            SELECT 1 FROM overlay.{table} AS o WHERE o.id = b.id
        )
        UNION ALL
        SELECT {overlaid}, 'overlay' AS source
        FROM overlay.{table} AS o
        WHERE o.operation = 'upsert'
    )
    """


def _instant(value: str) -> int:
    """Exact UTC microseconds, without float or timestamp-text ordering."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Stripe timestamp requires a timezone")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


class StripeTools:
    """Read the effective Stripe state for one configured account."""

    def __init__(self, config: StripeConfig | None = None):
        self.config = config or StripeConfig.from_env()
        self.config.validate()
        baseline_path = self.config.baseline_dir / "stripe.sqlite"
        overlay_path = self.config.overlay_dir / "overlay.sqlite"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing Stripe baseline: {baseline_path}")
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Missing Stripe overlay: {overlay_path}")

        self._connection = sqlite3.connect(
            ":memory:", timeout=self.config.sqlite_timeout_seconds, uri=True
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.create_function(
            "stripe_instant", 1, _instant, deterministic=True
        )
        self._connection.execute(
            "ATTACH DATABASE ? AS baseline",
            (f"file:{baseline_path.resolve()}?mode=ro&immutable=1",),
        )
        self._connection.execute(
            "ATTACH DATABASE ? AS overlay",
            (f"file:{overlay_path.resolve()}?mode=ro",),
        )
        self._connection.execute("PRAGMA query_only = ON")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def stripe_search_customers(
        self,
        query: str = "",
        *,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search customers by ID, name, email, description, or HubSpot ID."""

        query = self._search_query(query)
        limit = self._page_size(page_size)
        scope = self._scope("customers", query, limit, self.config.account_id)
        offset = self._cursor_offset(start_cursor, scope)
        conditions = ["account_id = ?"]
        params: list[Any] = [self.config.account_id]
        if query:
            pattern = f"%{self._like(query.lower())}%"
            conditions.append(
                "(lower(id) LIKE ? ESCAPE '$' OR lower(name) LIKE ? ESCAPE '$' "
                "OR lower(email) LIKE ? ESCAPE '$' OR lower(description) LIKE ? ESCAPE '$' "
                "OR lower(coalesce(hubspot_company_id, '')) LIKE ? ESCAPE '$')"
            )
            params.extend([pattern] * 5)
        rows, total = self._page(
            "customers",
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        customers = [self._customer_result(row) for row in rows]
        return self._page_result("customers", customers, total, offset, limit, scope)

    def stripe_list_subscriptions(
        self,
        *,
        customer_id: str | None = None,
        status: str = "all",
        price_id: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List subscriptions in stable newest-first order."""

        customer_id = self._optional_id(customer_id, "customer_id")
        price_id = self._optional_id(price_id, "price_id")
        status = self._status(status, SUBSCRIPTION_STATUSES, allow_all=True) or "all"
        if customer_id:
            self._require_customer(customer_id)
        if price_id:
            self._require_record("prices", price_id, "price")
        limit = self._page_size(page_size)
        scope = self._scope(
            "subscriptions",
            customer_id,
            status,
            price_id,
            limit,
            self.config.account_id,
        )
        offset = self._cursor_offset(start_cursor, scope)
        conditions = ["account_id = ?"]
        params: list[Any] = [self.config.account_id]
        if customer_id:
            conditions.append("customer_id = ?")
            params.append(customer_id)
        if status != "all":
            conditions.append("status = ?")
            params.append(status)
        if price_id:
            conditions.append("price_id = ?")
            params.append(price_id)
        rows, total = self._page(
            "subscriptions",
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        subscriptions = [self._subscription_result(row) for row in rows]
        return self._page_result(
            "subscriptions", subscriptions, total, offset, limit, scope
        )

    def stripe_list_invoices(
        self,
        *,
        customer_id: str | None = None,
        subscription_id: str | None = None,
        status: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List invoices with optional customer, subscription, status, and time filters."""

        customer_id = self._optional_id(customer_id, "customer_id")
        subscription_id = self._optional_id(subscription_id, "subscription_id")
        status = self._status(status, INVOICE_STATUSES, allow_all=False)
        created_after = self._timestamp(created_after, "created_after")
        created_before = self._timestamp(created_before, "created_before")
        self._validate_range(created_after, created_before)
        if customer_id:
            self._require_customer(customer_id)
        if subscription_id:
            self._require_record("subscriptions", subscription_id, "subscription")
        limit = self._page_size(page_size)
        scope = self._scope(
            "invoices",
            customer_id,
            subscription_id,
            status,
            created_after,
            created_before,
            limit,
            self.config.account_id,
        )
        offset = self._cursor_offset(start_cursor, scope)
        conditions = ["account_id = ?"]
        params: list[Any] = [self.config.account_id]
        for column, value in (
            ("customer_id", customer_id),
            ("subscription_id", subscription_id),
            ("status", status),
        ):
            if value:
                conditions.append(f"{column} = ?")
                params.append(value)
        self._add_time_conditions(conditions, params, created_after, created_before)
        rows, total = self._page(
            "invoices",
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        invoices = [self._invoice_result(row) for row in rows]
        return self._page_result("invoices", invoices, total, offset, limit, scope)

    def stripe_list_payments(
        self,
        *,
        customer_id: str | None = None,
        invoice_id: str | None = None,
        status: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        page_size: int | None = None,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """List PaymentIntents, exposed by the benchmark as payments."""

        customer_id = self._optional_id(customer_id, "customer_id")
        invoice_id = self._optional_id(invoice_id, "invoice_id")
        status = self._status(status, PAYMENT_STATUSES, allow_all=False)
        created_after = self._timestamp(created_after, "created_after")
        created_before = self._timestamp(created_before, "created_before")
        self._validate_range(created_after, created_before)
        if customer_id:
            self._require_customer(customer_id)
        if invoice_id:
            self._require_record("invoices", invoice_id, "invoice")
        limit = self._page_size(page_size)
        scope = self._scope(
            "payments",
            customer_id,
            invoice_id,
            status,
            created_after,
            created_before,
            limit,
            self.config.account_id,
        )
        offset = self._cursor_offset(start_cursor, scope)
        conditions = ["account_id = ?"]
        params: list[Any] = [self.config.account_id]
        for column, value in (
            ("customer_id", customer_id),
            ("invoice_id", invoice_id),
            ("status", status),
        ):
            if value:
                conditions.append(f"{column} = ?")
                params.append(value)
        self._add_time_conditions(conditions, params, created_after, created_before)
        rows, total = self._page(
            "payments",
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        payments = [self._payment_result(row) for row in rows]
        return self._page_result("payments", payments, total, offset, limit, scope)

    def _page(
        self,
        table: str,
        where: str,
        params: Sequence[Any],
        order: str,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        cte = _effective_cte(table)
        order = order.replace("created_at", "stripe_instant(created_at)")
        total = int(
            self._connection.execute(
                f"WITH {cte} SELECT COUNT(*) FROM effective_{table} WHERE {where}",
                tuple(params),
            ).fetchone()[0]
        )
        if self.config.max_result_bytes is not None:
            size = self._row_bytes(table)
            page_bytes = self._connection.execute(
                f"WITH {cte} SELECT COALESCE(SUM(n),0) FROM (SELECT {size} AS n FROM effective_{table} "
                f"WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?)",
                [*params, limit, offset],
            ).fetchone()[0]
            if page_bytes > self.config.max_result_bytes:
                raise StripeToolError(
                    "result_too_large", "Stripe page exceeds the result cap"
                )
        rows = self._rows(
            f"WITH {cte} SELECT * FROM effective_{table} WHERE {where} "
            f"ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return rows, total

    def _stored_result(self, table, row):
        result = dict(row)
        for field in JSON_COLUMNS & result.keys():
            result[field.removesuffix("_json")] = self._parse_json(result.pop(field))
        for field in BOOLEAN_COLUMNS & result.keys():
            result[field] = bool(result[field])
        result["object"] = {
            "customers": "customer",
            "products": "product",
            "prices": "price",
            "subscriptions": "subscription",
            "invoices": "invoice",
            "payments": "payment_intent",
            "refunds": "refund",
            "events": "event",
        }[table]
        return result

    def stripe_get_object(self, *, table, object_id):
        if not isinstance(table, str) or table not in TABLE_COLUMNS:
            raise StripeToolError("invalid_input", "Unknown stored Stripe table")
        object_id = self._optional_id(object_id, "object_id")
        if object_id is None:
            raise StripeToolError("invalid_input", "object_id is required")
        return {
            "ok": True,
            "record": self._stored_result(
                table, self._require_record(table, object_id, "object")
            ),
        }

    def stripe_list_objects(
        self,
        *,
        table,
        product_id=None,
        payment_id=None,
        page_size=None,
        start_cursor=None,
    ):
        if table not in ("products", "prices", "refunds"):
            raise StripeToolError(
                "invalid_input", "Expected products, prices or refunds"
            )
        conditions, params = ["account_id = ?"], [self.config.account_id]
        for field, value, parent, expected in (
            ("product_id", product_id, "products", "prices"),
            ("payment_id", payment_id, "payments", "refunds"),
        ):
            if value is not None:
                if table != expected:
                    raise StripeToolError(
                        "invalid_input", "Filter does not apply to this table"
                    )
                value = self._optional_id(value, field)
                self._require_record(parent, value, parent)
                conditions.append(f"{field} = ?")
                params.append(value)
        limit = self._page_size(page_size)
        scope = self._scope(
            table, product_id, payment_id, limit, self.config.account_id
        )
        offset = self._cursor_offset(start_cursor, scope)
        rows, total = self._page(
            table,
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        return self._page_result(
            table,
            [self._stored_result(table, row) for row in rows],
            total,
            offset,
            limit,
            scope,
        )

    def stripe_list_events(
        self,
        *,
        event_type=None,
        object_type=None,
        object_id=None,
        created_after=None,
        created_before=None,
        page_size=None,
        start_cursor=None,
    ):
        after = self._timestamp(created_after, "created_after")
        before = self._timestamp(created_before, "created_before")
        self._validate_range(after, before)
        conditions, params = ["account_id = ?"], [self.config.account_id]
        for field, value in (
            ("type", event_type),
            ("object_type", object_type),
            ("object_id", object_id),
        ):
            if value is not None:
                conditions.append(f"{field} = ?")
                params.append(self._optional_id(value, field))
        self._add_time_conditions(conditions, params, after, before)
        limit = self._page_size(page_size)
        scope = self._scope(
            "events",
            event_type,
            object_type,
            object_id,
            after,
            before,
            limit,
            self.config.account_id,
        )
        offset = self._cursor_offset(start_cursor, scope)
        rows, total = self._page(
            "events",
            " AND ".join(conditions),
            params,
            "created_at DESC, id",
            limit,
            offset,
        )
        return self._page_result(
            "events",
            [self._stored_result("events", row) for row in rows],
            total,
            offset,
            limit,
            scope,
        )

    def _record(self, table: str, record_id: str) -> dict[str, Any] | None:
        cte = _effective_cte(table)
        if self.config.max_result_bytes is not None:
            size = self._connection.execute(
                f"WITH {cte} SELECT {self._row_bytes(table)} FROM effective_{table} WHERE id=? AND account_id=?",
                (record_id, self.config.account_id),
            ).fetchone()
            if size and size[0] > self.config.max_result_bytes:
                raise StripeToolError(
                    "result_too_large", "Stripe object exceeds the result cap"
                )
        row = self._connection.execute(
            f"WITH {cte} SELECT * FROM effective_{table} "
            "WHERE id = ? AND account_id = ?",
            (record_id, self.config.account_id),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _row_bytes(table):
        return " + ".join(
            f'COALESCE(length(CAST("{column}" AS BLOB)),0)'
            for column in TABLE_COLUMNS[table]
        )

    def _require_record(self, table: str, record_id: str, label: str) -> dict[str, Any]:
        row = self._record(table, record_id)
        if row is None:
            raise StripeToolError("not_found", f"Stripe {label} was not found")
        return row

    def _require_customer(self, customer_id: str) -> dict[str, Any]:
        return self._require_record("customers", customer_id, "customer")

    def _refunds(self, payment_id: str) -> list[dict[str, Any]]:
        rows, _ = self._page(
            "refunds",
            "account_id = ? AND payment_id = ?",
            (self.config.account_id, payment_id),
            "created_at, id",
            self.config.max_page_size,
            0,
        )
        return [
            {
                "id": row["id"],
                "object": "refund",
                "payment_id": row["payment_id"],
                "status": row["status"],
                "currency": row["currency"],
                "amount": int(row["amount"]),
                "reason": row["reason"],
                "created_at": row["created_at"],
                "metadata": self._parse_json(row["metadata_json"]),
                "source": row["source"],
            }
            for row in rows
        ]

    def _customer_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        metadata = self._parse_json(row["metadata_json"])
        return {
            "id": row["id"],
            "object": "customer",
            "name": row["name"],
            "email": row["email"],
            "description": row["description"],
            "currency": row["currency"],
            "balance": int(row["balance"]),
            "delinquent": bool(row["delinquent"]),
            "tax_exempt": row["tax_exempt"],
            "hubspot_company_id": row["hubspot_company_id"],
            "billing_name": metadata.get("billing_name", row["name"]),
            "billing_address": metadata.get("billing_address"),
            "customer_since": metadata.get("customer_since"),
            "billing_cadence": metadata.get("billing_cadence"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": metadata,
            "source": row["source"],
        }

    def _price_result(self, price_id: str) -> dict[str, Any]:
        price = self._require_record("prices", price_id, "price")
        product = self._require_record("products", price["product_id"], "product")
        return {
            "id": price["id"],
            "object": "price",
            "active": bool(price["active"]),
            "currency": price["currency"],
            "unit_amount": int(price["unit_amount"]),
            "recurring": {
                "interval": price["recurring_interval"],
                "interval_count": int(price["recurring_interval_count"]),
                "usage_type": price["usage_type"],
            },
            "product": {
                "id": product["id"],
                "name": product["name"],
                "description": product["description"],
                "active": bool(product["active"]),
            },
            "metadata": self._parse_json(price["metadata_json"]),
            "source": price["source"],
        }

    def _subscription_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "object": "subscription",
            "customer_id": row["customer_id"],
            "status": row["status"],
            "price": self._price_result(row["price_id"]),
            "quantity": int(row["quantity"]),
            "cancel_at_period_end": bool(row["cancel_at_period_end"]),
            "current_period_start": row["current_period_start"],
            "current_period_end": row["current_period_end"],
            "trial_end": row["trial_end"],
            "canceled_at": row["canceled_at"],
            "ended_at": row["ended_at"],
            "latest_invoice_id": row["latest_invoice_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": self._parse_json(row["metadata_json"]),
            "source": row["source"],
        }

    def _invoice_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payment_rows, payment_total = self._page(
            "payments",
            "account_id = ? AND invoice_id = ?",
            (self.config.account_id, row["id"]),
            "created_at, id",
            self.config.max_page_size,
            0,
        )
        return {
            "id": row["id"],
            "object": "invoice",
            "number": row["number"],
            "customer_id": row["customer_id"],
            "subscription_id": row["subscription_id"],
            "status": row["status"],
            "currency": row["currency"],
            "subtotal": int(row["subtotal"]),
            "total": int(row["total"]),
            "amount_due": int(row["amount_due"]),
            "amount_paid": int(row["amount_paid"]),
            "amount_remaining": int(row["amount_remaining"]),
            "billing_reason": row["billing_reason"],
            "due_date": row["due_date"],
            "paid_at": row["paid_at"],
            "attempt_count": int(row["attempt_count"]),
            "next_payment_attempt": row["next_payment_attempt"],
            "period_start": row["period_start"],
            "period_end": row["period_end"],
            "hosted_invoice_url": row["hosted_invoice_url"],
            "payment_ids": [payment["id"] for payment in payment_rows],
            "payment_ids_has_more": payment_total > len(payment_rows),
            "created_at": row["created_at"],
            "metadata": self._parse_json(row["metadata_json"]),
            "source": row["source"],
        }

    def _payment_result(self, row: Mapping[str, Any]) -> dict[str, Any]:
        refunds = self._refunds(row["id"])
        refund_total, refunded = self._connection.execute(
            f"WITH {_effective_cte('refunds')} SELECT COUNT(*), "
            "COALESCE(SUM(CASE WHEN status='succeeded' AND currency=? THEN amount ELSE 0 END),0) "
            "FROM effective_refunds WHERE account_id=? AND payment_id=?",
            (row["currency"], self.config.account_id, row["id"]),
        ).fetchone()
        return {
            "id": row["id"],
            "object": "payment_intent",
            "customer_id": row["customer_id"],
            "invoice_id": row["invoice_id"],
            "status": row["status"],
            "currency": row["currency"],
            "amount": int(row["amount"]),
            "amount_received": int(row["amount_received"]),
            "amount_refunded": refunded,
            "refunds_has_more": refund_total > len(refunds),
            "payment_method_type": row["payment_method_type"],
            "description": row["description"],
            "failure_code": row["failure_code"],
            "failure_message": row["failure_message"],
            "receipt_email": row["receipt_email"],
            "latest_charge_id": row["latest_charge_id"],
            "created_at": row["created_at"],
            "canceled_at": row["canceled_at"],
            "refunds": refunds,
            "metadata": self._parse_json(row["metadata_json"]),
            "source": row["source"],
        }

    def _page_result(
        self,
        key: str,
        rows: list[dict[str, Any]],
        total: int,
        offset: int,
        limit: int,
        scope: str,
    ) -> dict[str, Any]:
        next_offset = offset + len(rows)
        has_more = next_offset < total
        return {
            "ok": True,
            "object": "list",
            key: rows,
            "total": total,
            "has_more": has_more,
            "next_cursor": (
                self._encode_cursor(next_offset, scope) if has_more else None
            ),
        }

    def _page_size(self, value: int | None) -> int:
        value = self.config.default_page_size if value is None else value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise StripeToolError(
                "invalid_input", "page_size must be a positive integer"
            )
        if value > self.config.max_page_size:
            raise StripeToolError(
                "invalid_input", f"page_size cannot exceed {self.config.max_page_size}"
            )
        return value

    def _search_query(self, value: Any) -> str:
        if not isinstance(value, str):
            raise StripeToolError("invalid_input", "query must be a string")
        normalized = value.strip()
        if len(normalized) > self.config.max_search_characters:
            raise StripeToolError(
                "invalid_input",
                f"query cannot exceed {self.config.max_search_characters} characters",
            )
        return normalized

    @staticmethod
    def _optional_id(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise StripeToolError(
                "invalid_input", f"{field} must be a non-empty string"
            )
        return value.strip()

    @staticmethod
    def _status(value: Any, allowed: set[str], *, allow_all: bool) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise StripeToolError("invalid_input", "status must be a string")
        normalized = value.strip().lower()
        if allow_all and normalized == "all":
            return normalized
        if normalized not in allowed:
            choices = sorted(allowed | ({"all"} if allow_all else set()))
            raise StripeToolError(
                "invalid_input", f"status must be one of: {', '.join(choices)}"
            )
        return normalized

    @staticmethod
    def _timestamp(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise StripeToolError(
                "invalid_input", f"{field} must be an ISO 8601 timestamp"
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise StripeToolError(
                "invalid_input", f"{field} must be an ISO 8601 timestamp"
            ) from error
        if parsed.tzinfo is None:
            raise StripeToolError("invalid_input", f"{field} must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _validate_range(after: str | None, before: str | None) -> None:
        if after and before and after >= before:
            raise StripeToolError(
                "invalid_input", "created_after must be before created_before"
            )

    @staticmethod
    def _add_time_conditions(
        conditions: list[str],
        params: list[Any],
        created_after: str | None,
        created_before: str | None,
    ) -> None:
        if created_after:
            conditions.append("stripe_instant(created_at) > ?")
            params.append(_instant(created_after))
        if created_before:
            conditions.append("stripe_instant(created_at) < ?")
            params.append(_instant(created_before))

    @staticmethod
    def _like(value: str) -> str:
        return value.replace("$", "$$").replace("%", "$%").replace("_", "$_")

    @staticmethod
    def _scope(*values: Any) -> str:
        return hashlib.sha256(_json(values).encode()).hexdigest()[:20]

    @staticmethod
    def _encode_cursor(offset: int, scope: str) -> str:
        payload = _json({"offset": offset, "scope": scope}).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def _cursor_offset(cursor: str | None, scope: str) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or not cursor or len(cursor) > 4096:
            raise StripeToolError(
                "invalid_cursor", "Stripe cursor is invalid for this request"
            )
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            )
            offset = payload["offset"]
            if (
                payload["scope"] != scope
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or not 0 <= offset < 2**63
            ):
                raise ValueError
            return offset
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise StripeToolError(
                "invalid_cursor", "Stripe cursor is invalid for this request"
            ) from error

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if not isinstance(value, str):
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}

    def _rows(self, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(query, tuple(params))]


_DEFAULT_TOOLS: StripeTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> StripeTools:
    global _DEFAULT_TOOLS
    with _DEFAULT_LOCK:
        if _DEFAULT_TOOLS is None:
            _DEFAULT_TOOLS = StripeTools()
        return _DEFAULT_TOOLS
