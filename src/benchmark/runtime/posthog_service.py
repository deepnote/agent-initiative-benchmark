"""Service-owned PostHog wiring for the shared runtime registry."""

NAME = "posthog"
WRITES = frozenset()
DELTA_FILE = "delta/posthog/seed.json"
READS = frozenset(
    (
        "posthog_query_events",
        "posthog_query_persons",
        "posthog_query_insight",
        "posthog_get_feature_flag",
        "posthog_get_event",
        "posthog_list_catalog",
        "posthog_query_history",
    )
)


def definitions(scope, cap):
    from .posthog_environment import tool_definitions

    return tool_definitions(scope, cap)


def agent_definition(definition):
    """Advertise the tested agent's explicit discovery default, not native full."""
    if definition["name"] != "posthog_list_catalog":
        return definition
    import copy

    result = copy.deepcopy(definition)
    result["parameters"]["properties"]["projection"]["default"] = "summary"
    result["description"] += (
        " Tested-agent default: omitted projection is executed as summary. "
        'Use projection="full" with insight_id to read one complete saved insight, or full without an ID for a full catalog.'
    )
    return result


def agent_arguments(name, arguments):
    if name == "posthog_list_catalog" and "projection" not in arguments:
        return dict(arguments, projection="summary")
    return arguments


def create(arguments, manifest, task, config, acquire):
    from .posthog_environment import PostHogEnvironment

    return PostHogEnvironment(arguments, manifest, task, config, acquire)


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file", "PostHog is not enabled", service=NAME
        )
    return ("posthog/manifest.json",)


def receipt_files(baseline):
    return baseline.plan_service(NAME).paths


def local_file(path):
    return path.endswith(".parquet") or path in (
        "manifest.json",
        "event_definitions.json",
        "insights.json",
        "feature_flags.json",
    )


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "manifest.json").is_file())
    ):
        raise RuntimeInputError(
            "PostHog requires explicit service_baseline_dirs.posthog with manifest.json"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local PostHog roots")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "manifest.json" for f in files),
        "missing_posthog_baseline",
        "PostHog requires an explicit local root with manifest.json",
    )
    check(
        enabled or root is None,
        "unused_posthog_baseline",
        "PostHog root requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/posthog/data_overview.txt") if set(scope) & READS else ""


def native_success(name, result):
    return (
        name in READS
        and isinstance(result, dict)
        and result.get("ok") is True
        and isinstance(result.get("object"), str)
    )


def evidence_entries(result):
    return ()


def related_records(session, row):
    """Retain bounded, actually delivered linked evidence for the dashboard."""
    available = {d["name"] for d in session.definitions}
    linked = []
    if row.get("person_id") and "posthog_query_persons" in available:
        read = session.call(
            "posthog_query_persons", {"person_id": row["person_id"], "limit": 1}
        )
        linked.append(
            dict(
                label="Current person profile",
                source_id=read["source_id"],
                result=read["result"],
            )
        )
    if row.get("distinct_id") and "posthog_query_history" in available:
        read = session.call(
            "posthog_query_history",
            {"kind": "distinct_ids", "distinct_id": row["distinct_id"], "limit": 100},
        )
        linked.append(
            dict(
                label="Stored identity links",
                source_id=read["source_id"],
                result=read["result"],
            )
        )
    if "posthog_list_catalog" in available:
        for field, table, label in (
            ("workspace_id", "workspace_groups", "Current workspace group"),
            ("website_id", "website_groups", "Current website group"),
        ):
            if row.get(field):
                record = dict(table=table, id=row[field])
                data = session.record("posthog", record)
                linked.append(dict(label=label, record=record, result=data))
    return linked


def trace_facts(name, result):
    if not native_success(name, result):
        return

    # Every pointer resolves inside the delivered result, including analysis
    # dates, stored context and fresh counts. No unseen records are registered.
    def visit(value, pointer):
        if isinstance(value, dict):
            for key, child in value.items():
                path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                yield path, child
                yield from visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                path = pointer + "/" + str(index)
                yield path, child
                yield from visit(child, path)

    yield from visit(result, "")


def trusted_error(error):
    from .posthog_environment import PostHogEnvironmentError

    return isinstance(error, PostHogEnvironmentError)
