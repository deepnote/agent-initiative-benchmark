"""HubSpot hooks for the existing shared service registry."""

from __future__ import annotations
from pathlib import Path

NAME = "hubspot"
WRITES = frozenset()
DELTA_FILE = "delta/hubspot/seed.json"


def definitions(scope, cap):
    from .hubspot_environment import tool_definitions

    return tool_definitions(scope, cap)


from .hubspot_environment import SPECS

READS = frozenset(SPECS)


def create(arguments, manifest, task, config, acquire):
    from .hubspot_environment import HubSpotEnvironment

    env = HubSpotEnvironment(
        arguments["package"],
        manifest,
        task,
        config,
        arguments["work_dir"],
        arguments["max_result_bytes"],
        acquire,
    )
    root = config["storage"].get("baseline", {}).get("cache_dir")
    if root:
        env.configure_document_cache(Path(root) / "hubspot-text", config["build_ref"])
    return env


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file",
            "HubSpot is absent from the pinned source",
            service=NAME,
        )
    return ("hubspot/hubspot.sqlite",)


def local_file(path):
    return path == "hubspot.sqlite" or (
        path.startswith("contracts/") and path.endswith((".pdf", ".md"))
    )


def linked_file(path):
    from .hubspot_environment import document_path, HubSpotEnvironmentError

    try:
        return path.startswith("hubspot/") and bool(
            document_path(path[len("hubspot/") :])
        )
    except HubSpotEnvironmentError:
        return False


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "hubspot.sqlite").is_file())
    ):
        raise RuntimeInputError(
            "HubSpot local_declared requires an explicit service_baseline_dirs.hubspot root with hubspot.sqlite"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids a local HubSpot root")
    if root is not None and not enabled:
        raise RuntimeInputError("A HubSpot local root requires enabled HubSpot reads")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or (root is not None and any(f["path"] == "hubspot.sqlite" for f in files)),
        "missing_hubspot_baseline",
        "HubSpot reads require an explicit root with hubspot.sqlite",
    )
    check(
        enabled or root is None,
        "unused_hubspot_baseline",
        "HubSpot root requires enabled HubSpot reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/hubspot/data_overview.txt") if set(scope) & READS else ""


def evidence_entries(result):
    return ()


def native_success(name, result):
    return (
        isinstance(result, dict)
        and "error" not in result
        and (
            isinstance(result.get("results"), list)
            or isinstance(result.get("id"), str)
            or isinstance(result.get("document_id"), str)
        )
    )


def trace_facts(name, result):
    if name not in READS or "error" in result:
        return
    if name == "hubspot_get_record":
        for field, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    yield "/" + field + "/" + k.replace("~", "~0").replace("/", "~1"), v
            else:
                yield "/" + field, value
    elif "pages" in result:
        for field, value in result.items():
            if field == "pages":
                for i, page in enumerate(value):
                    yield f"/pages/{i}", page
            else:
                yield "/" + field, value
    elif "results" in result:
        for i, row in enumerate(result["results"]):
            yield f"/results/{i}", row
        for field, value in result.items():
            if field != "results":
                yield "/" + field, value
    else:
        for field, value in result.items():
            yield "/" + field, value


def trusted_error(error):
    from .hubspot_environment import HubSpotEnvironmentError

    return isinstance(error, HubSpotEnvironmentError)
