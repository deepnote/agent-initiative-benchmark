"""Calendar hooks for shared dispatch, evidence, packages and leases."""

NAME = "calendar"
READS = frozenset(
    (
        "calendar_list_calendars",
        "calendar_search_events",
        "calendar_get_event",
        "calendar_query_free_busy",
    )
)
WRITES = frozenset()
DELTA_FILE = "delta/calendar/seed.json"


def definitions(scope, cap):
    from .calendar_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .calendar_environment import CalendarEnvironment

    return CalendarEnvironment(
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
            "Calendar is not enabled in the pinned baseline",
            service=NAME,
        )
    return ("calendar/calendar.sqlite",)


def local_file(path):
    return path == "calendar.sqlite"


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "calendar.sqlite").is_file())
    ):
        raise RuntimeInputError(
            "Calendar requires explicit calendar_baseline_dir with calendar.sqlite"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local calendar_baseline_dir")
    if not enabled and root is not None:
        raise RuntimeInputError(
            "Calendar local input requires configured Calendar reads"
        )


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "calendar.sqlite" for f in files),
        "missing_calendar_baseline",
        "Calendar reads require explicit calendar_baseline_dir with calendar.sqlite",
    )
    check(
        enabled or root is None,
        "unused_calendar_baseline",
        "Calendar input requires configured Calendar reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/calendar/data_overview.txt") if set(scope) & READS else ""


def evidence_entries(result):
    return ()


def native_success(name, result):
    if not isinstance(result, dict) or "error" in result:
        return False
    if name == "calendar_get_event":
        return isinstance(result.get("id"), str) and isinstance(
            result.get("calendar_id"), str
        )
    if name == "calendar_query_free_busy":
        return (
            isinstance(result.get("calendars"), dict)
            and "time_min" in result
            and "time_max" in result
        )
    collection = "events" if name == "calendar_search_events" else "calendars"
    return (
        isinstance(result.get(collection), list)
        and type(result.get("has_more")) is bool
        and type(result.get("total")) is int
    )


def trace_facts(name, result):
    if not native_success(name, result):
        return

    # Only pointers into delivered payloads. Computed occurrences remain tied to
    # their stored series ID and original_start_time, never invented native rows.
    def visit(value, pointer):
        if isinstance(value, dict):
            for k, v in value.items():
                p = pointer + "/" + k.replace("~", "~0").replace("/", "~1")
                yield p, v
                yield from visit(v, p)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                p = pointer + "/" + str(i)
                yield p, v
                yield from visit(v, p)

    yield from visit(result, "")


def trusted_error(error):
    from .calendar_environment import CalendarEnvironmentError

    return isinstance(error, CalendarEnvironmentError)
