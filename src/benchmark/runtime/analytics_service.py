"""Analytics hooks for the existing runtime registry."""

NAME = "analytics"
READS = frozenset(
    ("analytics_run_report", "analytics_get_metadata", "analytics_read_rows")
)
WRITES = frozenset()
DELTA_FILE = "delta/analytics/seed.json"


def definitions(scope, cap):
    from .analytics_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .analytics_environment import AnalyticsEnvironment

    return AnalyticsEnvironment(arguments, manifest, task, config, acquire)


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file", "Analytics is not enabled", service=NAME
        )
    return tuple(
        "analytics/" + p
        for p in ("manifest.json", "property.json", "dimensions.json", "metrics.json")
    )


def receipt_files(baseline):
    return baseline.plan_service(NAME).paths


def local_file(path):
    return path.endswith(".parquet") or path in (
        "manifest.json",
        "property.json",
        "dimensions.json",
        "metrics.json",
    )


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (
            root is None
            or not all(
                (Path(root) / p).is_file()
                for p in (
                    "manifest.json",
                    "property.json",
                    "dimensions.json",
                    "metrics.json",
                )
            )
        )
    ):
        raise RuntimeInputError(
            "Analytics requires explicit service_baseline_dirs.analytics with native metadata"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local Analytics roots")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and {"manifest.json", "property.json", "dimensions.json", "metrics.json"}
        <= {f["path"] for f in files},
        "missing_analytics_baseline",
        "Analytics requires an explicit local root with metadata",
    )
    check(
        enabled or root is None,
        "unused_analytics_baseline",
        "Analytics root requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/analytics/data_overview.txt") if set(scope) & READS else ""


def native_success(name, result):
    return (
        name in READS
        and isinstance(result, dict)
        and result.get("ok") is True
        and isinstance(result.get("kind"), str)
    )


def evidence_entries(result):
    return ()


def trace_facts(name, result):
    if native_success(name, result):

        def visit(value, pointer):
            if isinstance(value, dict):
                for key, child in value.items():
                    path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                    yield path, child
                    yield from visit(child, path)
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    path = pointer + "/" + str(i)
                    yield path, child
                    yield from visit(child, path)

        yield from visit(result, "")


def trusted_error(error):
    from .analytics_environment import AnalyticsEnvironmentError

    return isinstance(error, AnalyticsEnvironmentError)
