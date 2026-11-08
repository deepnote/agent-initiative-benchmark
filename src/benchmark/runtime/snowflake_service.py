"""Snowflake hooks for shared dispatch, native task evidence and file leases."""

NAME = "snowflake"
READS = frozenset(
    (
        "snowflake_list_datasets",
        "snowflake_list_tables",
        "snowflake_get_table_schema",
        "snowflake_query",
    )
)
WRITES = frozenset()
DELTA_FILE = "delta/snowflake/seed.json"


def definitions(scope, cap):
    from .snowflake_environment import tool_definitions

    return tool_definitions(scope, cap)


def create(arguments, manifest, task, config, acquire):
    from .snowflake_environment import SnowflakeEnvironment

    return SnowflakeEnvironment(arguments, manifest, task, config, acquire)


def required_files(baseline):
    from .baseline_access import BaselineAccessError

    if NAME not in baseline.metadata["enabled_services"]:
        raise BaselineAccessError(
            "missing_required_file", "Snowflake is not enabled", service=NAME
        )
    return ("snowflake/manifest.json",)


def receipt_files(baseline):
    return tuple(
        p for p in baseline.plan_service(NAME).paths if local_file(p.split("/", 1)[1])
    )


def local_file(path):
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in path
        and (
            path == "manifest.json"
            or path.startswith("schemas/")
            and path.endswith(".json")
            or path.endswith(".parquet")
        )
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
            "Snowflake requires service_baseline_dirs.snowflake with manifest.json"
        )
    if mode == "verified_local" and root is not None:
        raise RuntimeInputError("verified_local forbids local Snowflake roots")
    if not enabled and root is not None:
        raise RuntimeInputError("Snowflake root requires configured reads")


def validate_judge_local(enabled, root, files, check):
    check(
        not enabled
        or root is not None
        and any(f["path"] == "manifest.json" for f in files),
        "missing_snowflake_baseline",
        "Snowflake requires a local root with manifest.json",
    )
    check(
        enabled or root is None,
        "unused_snowflake_baseline",
        "Snowflake root requires configured reads",
    )


def discover_settings(root):
    """Reuse the native wrapper's role/database policy, narrowed to this source.

    The source catalog has no employee ACL. Explicit schema grants narrow its
    native configured database boundary. This hook does not infer wider access.
    """
    import json
    from pathlib import Path
    from benchmark.tools.snowflake.snowflake import SnowflakeConfig

    manifest = json.loads((Path(root) / "manifest.json").read_text())
    defaults = SnowflakeConfig()
    databases = [
        d["name"]
        for d in manifest["databases"]
        if d["name"] in defaults.allowed_databases
    ]
    return dict(
        account_identifier=manifest["account"]["identifier"],
        role=defaults.role,
        warehouse=defaults.warehouse,
        allowed_databases=databases,
        allowed_schemas=sorted(
            s["database_name"] + "." + s["schema_name"]
            for s in manifest["schemas"]
            if s["database_name"] in databases
        ),
    )


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/snowflake/data_overview.txt") if set(scope) & READS else ""


def native_success(name, result):
    kinds = dict(
        snowflake_list_datasets="snowflake#databaseList",
        snowflake_list_tables="snowflake#tableList",
        snowflake_get_table_schema="snowflake#tableSchema",
        snowflake_query="snowflake#queryResult",
    )
    return (
        isinstance(result, dict)
        and result.get("kind") == kinds.get(name)
        and name in READS
    )


def evidence_entries(result):
    return ()


def trace_facts(name, result):
    if not native_success(name, result):
        return

    def visit(value, pointer):
        if isinstance(value, dict):
            for field, child in value.items():
                path = pointer + "/" + field.replace("~", "~0").replace("/", "~1")
                yield path, child
                yield from visit(child, path)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                path = pointer + "/" + str(i)
                yield path, child
                yield from visit(child, path)

    yield from visit(result, "")


def trusted_error(error):
    from .snowflake_environment import SnowflakeEnvironmentError

    return isinstance(error, SnowflakeEnvironmentError)
