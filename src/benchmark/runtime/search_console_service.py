"""Search Console adapter for the shared runtime registry."""

NAME = "search_console"
WRITES = frozenset()
DELTA_FILE = "delta/search_console/seed.json"
READS = frozenset(
    (
        "search_console_query_performance",
        "search_console_inspect_url",
        "search_console_list_properties",
        "search_console_list_pages",
        "search_console_list_queries",
        "search_console_get_record",
    )
)


def definitions(scope, cap):
    from .search_console_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .search_console_environment import SearchConsoleEnvironment

    return SearchConsoleEnvironment(arguments, manifest, task, config, acquire)


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file", "Search Console is not enabled", service=NAME
        )
    return ("search_console/properties.json",)


def local_file(path):
    import re

    return path in ("properties.json", "pages.parquet", "queries.parquet") or bool(
        re.fullmatch(
            r"performance/(?:[^/]+/)*[^/]+\.parquet|url-inspection/bucket=\d{2}/snapshots\.json",
            path,
        )
    )


def receipt_files(baseline):
    return tuple(
        p for p in baseline.plan_service(NAME).paths if local_file(p.split("/", 1)[1])
    )


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "properties.json").is_file())
    ):
        raise RuntimeInputError(
            "Search Console requires explicit service_baseline_dirs.search_console with properties.json"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local Search Console roots")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "properties.json" for f in files),
        "missing_search_console_baseline",
        "Search Console requires an explicit local root with properties.json",
    )
    check(
        enabled or root is None,
        "unused_search_console_baseline",
        "Search Console root requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return (
        load_text("guides/search_console/data_overview.txt")
        if set(scope) & READS
        else ""
    )


def native_success(name, result):
    return name in READS and isinstance(result, dict) and result.get("ok") is True


def evidence_entries(result):
    return ()


def trace_facts(name, result):
    if not native_success(name, result):
        return

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
    from .search_console_environment import SearchConsoleEnvironmentError

    return isinstance(error, SearchConsoleEnvironmentError)
