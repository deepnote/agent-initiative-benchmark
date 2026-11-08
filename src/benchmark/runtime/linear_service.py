"""Service-owned runtime wiring. Native libraries own storage and queries."""

from __future__ import annotations


NAME = "linear"
READS = frozenset(("linear_search_issues", "linear_get_issue"))
WRITES = frozenset()
DELTA_FILE = "delta/linear/seed.json"


def definitions(scope, cap):
    from .linear_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .linear_environment import LinearEnvironment

    return LinearEnvironment(
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
            "Linear is not enabled in the pinned baseline",
            service=NAME,
        )
    return ("linear/linear.sqlite",)


def local_file(path):
    return path.endswith(".parquet") or path == "linear.sqlite"


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if mode == "local_declared" and enabled:
        if root is None or not (Path(root) / "linear.sqlite").is_file():
            raise RuntimeInputError(
                "Linear local_declared requires explicit linear_baseline_dir with linear.sqlite"
            )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local linear_baseline_dir")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or (root is not None and any(f["path"] == "linear.sqlite" for f in files)),
        "missing_linear_baseline",
        "Enabled Linear reads require explicit linear_baseline_dir with linear.sqlite",
    )
    check(
        enabled or root is None,
        "unused_linear_baseline",
        "Linear baseline input requires configured Linear reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/linear/data_overview.txt") if set(scope) & READS else ""


def evidence_entries(result):
    return ()


def native_success(name, result):
    return (
        isinstance(result, dict)
        and "ok" not in result
        and (
            (
                name == "linear_get_issue"
                and isinstance(result.get("id"), str)
                and isinstance(result.get("identifier"), str)
            )
            or (
                name == "linear_search_issues"
                and isinstance(result.get("issues"), list)
                and type(result.get("total")) is int
                and type(result.get("has_more")) is bool
                and (
                    result.get("next_cursor") is None
                    or isinstance(result.get("next_cursor"), str)
                )
            )
        )
    )


def trace_facts(name, result):
    if name not in READS or "error" in result:
        return

    def fields(row, prefix):
        for key, value in row.items():
            if key in {"comments", "relations", "history", "labels"} and isinstance(
                value, list
            ):
                for index, record in enumerate(value):
                    yield f"{prefix}/{key}/{index}", record
            else:
                yield f"{prefix}/{key}", value

    if name == "linear_search_issues":
        for index, row in enumerate(result.get("issues", [])):
            yield from fields(row, f"/issues/{index}")
    else:
        yield from fields(result, "")


def trusted_error(error):
    from .linear_environment import LinearEnvironmentError

    return isinstance(error, LinearEnvironmentError)
