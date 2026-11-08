"""Notion hooks for the existing service registry."""

from .notion_environment import NOTION_READS, FILE

NAME = "notion"
READS = frozenset(NOTION_READS)
WRITES = frozenset()
DELTA_FILE = FILE


def definitions(scope, cap):
    from .notion_environment import definitions

    return definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .notion_environment import NotionEnvironment

    return NotionEnvironment(
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
            "Notion is not enabled in the pinned source",
            service=NAME,
        )
    return ("notion/catalog.sqlite",)


def local_file(path):
    import re

    return path in ("catalog.sqlite", "revisions.jsonl") or bool(
        re.fullmatch(r"(pages/[^/]+\.md|versions/[^/]+/[A-Za-z0-9_-]+\.md)", path)
    )


def receipt_paths(baseline, paths):
    return {"notion/catalog.sqlite"} | {
        p for p in paths if p.startswith("notion/") and local_file(p[7:])
    }


def validate_local(mode, enabled, root):
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if (
        mode == "local_declared"
        and enabled
        and (root is None or not (Path(root) / "catalog.sqlite").is_file())
    ):
        raise RuntimeInputError(
            "Enabled Notion reads require explicit notion_baseline_dir with catalog.sqlite"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local notion_baseline_dir")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "catalog.sqlite" for f in files),
        "missing_notion_baseline",
        "Enabled Notion reads require explicit notion_baseline_dir with catalog.sqlite",
    )
    check(
        enabled or root is None,
        "unused_notion_baseline",
        "Notion root requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/notion/data_overview.txt") if set(scope) & READS else ""


def native_success(name, result):
    return (
        isinstance(result, dict)
        and result.get("object") in ("list", "page", "page_version")
        and "error" not in result
    )


def native_error(name, result):
    """Recognize the read tool's exact error envelope without calling it a success."""
    codes = {
        "embedding_unavailable",
        "index_dimension_mismatch",
        "index_model_mismatch",
        "invalid_catalog",
        "invalid_content",
        "invalid_cursor",
        "invalid_filter",
        "invalid_id",
        "invalid_index",
        "invalid_path",
        "invalid_request",
        "invalid_sort",
        "not_found",
    }
    return (
        name in READS
        and isinstance(result, dict)
        and set(result) == {"object", "code", "message"}
        and result["object"] == "error"
        and isinstance(result["code"], str)
        and result["code"] in codes
        and isinstance(result["message"], str)
        and bool(result["message"].strip())
    )


def evidence_entries(result):
    return ()


def trace_facts(name, result):
    if not native_success(name, result):
        return
    for key, value in result.items():
        if key == "results":
            for index, row in enumerate(value):
                yield f"/results/{index}", row
        else:
            yield "/" + key, value


def trusted_error(error):
    from .notion_environment import NotionEnvironmentError

    return isinstance(error, NotionEnvironmentError)
