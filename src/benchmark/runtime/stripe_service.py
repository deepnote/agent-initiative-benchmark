"""Stripe-owned hooks for the shared runtime registry."""

from __future__ import annotations

from .stripe_environment import READS as TOOLS

NAME = "stripe"
READS = frozenset(TOOLS)
WRITES = frozenset()
DELTA_FILE = "delta/stripe/seed.json"


def definitions(scope, cap):
    from .stripe_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .stripe_environment import StripeEnvironment

    return StripeEnvironment(
        arguments["package"],
        manifest,
        task,
        config,
        arguments["work_dir"],
        arguments["max_result_bytes"],
        acquire,
    )


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file",
            "Stripe is not enabled in the pinned source",
            service=NAME,
        )
    return ("stripe/stripe.sqlite",)


def local_file(path):
    return path == "stripe.sqlite"


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "stripe.sqlite").is_file())
    ):
        raise RuntimeInputError(
            "Stripe local_declared requires explicit stripe_baseline_dir with stripe.sqlite"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local stripe_baseline_dir")
    if not enabled and root is not None:
        raise RuntimeInputError("Stripe baseline requires configured Stripe reads")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or (root is not None and any(f["path"] == "stripe.sqlite" for f in files)),
        "missing_stripe_baseline",
        "Stripe reads require explicit stripe.sqlite",
    )
    check(
        enabled or root is None,
        "unused_stripe_baseline",
        "Stripe baseline requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/stripe/data_overview.txt") if set(scope) & READS else ""


def native_success(name, result):
    return (
        isinstance(result, dict)
        and result.get("ok") is True
        and (
            isinstance(result.get("record"), dict)
            or (
                result.get("object") == "list"
                and type(result.get("total")) is int
                and type(result.get("has_more")) is bool
            )
        )
    )


def evidence_entries(result):
    return ()


def trace_facts(name, result):
    if not native_success(name, result):
        return
    if "record" in result:
        for key, value in result["record"].items():
            yield "/record/" + key, value
    else:
        from .stripe_environment import TABLE_COLUMNS

        for table in TABLE_COLUMNS:
            for index, row in enumerate(result.get(table, [])):
                for key, value in row.items():
                    yield f"/{table}/{index}/{key}", value


def trusted_error(error):
    from .stripe_environment import StripeEnvironmentError

    return isinstance(error, StripeEnvironmentError)
