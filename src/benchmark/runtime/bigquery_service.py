"""BigQuery runtime service reads."""

NAME = "bigquery"
WRITES = frozenset()
DELTA_FILE = "delta/bigquery/seed.json"
READS = frozenset(
    (
        "bigquery_list_datasets",
        "bigquery_list_tables",
        "bigquery_get_table_schema",
        "bigquery_query",
    )
)


def definitions(scope, cap):
    from .bigquery_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .bigquery_environment import BigQueryEnvironment

    return BigQueryEnvironment(arguments, manifest, task, config, acquire)


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file", "BigQuery is not enabled", service=NAME
        )
    return ("bigquery/manifest.json",)


preparation_files = required_files


def discover_settings(root):
    import json
    from pathlib import Path
    from benchmark.tools.bigquery.bigquery import (
        BigQueryConfig,
        DEFAULT_ALLOWED_DATASETS,
    )

    manifest = json.loads((Path(root) / "manifest.json").read_text())
    return dict(
        project_id=manifest["project"]["id"],
        location=manifest["project"].get("location", "US"),
        identity=BigQueryConfig.agent_principal,
        read_dataset_ids=sorted(
            set(DEFAULT_ALLOWED_DATASETS) & {d["id"] for d in manifest["datasets"]}
        ),
    )


def local_file(path):
    from pathlib import PurePosixPath

    parts = PurePosixPath(path).parts
    return not any(p in (".", "..") for p in parts) and (
        path == "manifest.json"
        or path.startswith("schemas/")
        and path.endswith(".json")
        or path.endswith(".parquet")
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
        and (root is None or not (Path(root) / "manifest.json").is_file())
    ):
        raise RuntimeInputError(
            "BigQuery requires service_baseline_dirs.bigquery with manifest.json"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local BigQuery roots")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "manifest.json" for f in files),
        "missing_bigquery_baseline",
        "BigQuery requires an explicit local root with manifest.json",
    )
    check(
        enabled or root is None,
        "unused_bigquery_baseline",
        "BigQuery root requires configured reads",
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/bigquery/data_overview.txt") if set(scope) & READS else ""


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
            for index, child in enumerate(value):
                path = pointer + "/" + str(index)
                yield path, child
                yield from visit(child, path)

    yield from visit(result, "")


def trusted_error(error):
    from .bigquery_environment import BigQueryEnvironmentError
    from benchmark.tools.bigquery.bigquery import BigQueryToolError

    return isinstance(error, (BigQueryEnvironmentError, BigQueryToolError))
