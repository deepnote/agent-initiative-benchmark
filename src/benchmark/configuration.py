"""One project configuration and verified current task packages."""

import json
from itertools import combinations
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from .dataset import DEFAULT_DATA, DEFAULT_CACHE, Release, check_data, cache_at
from .runtime.agent_runner import resolve_limits, baseline_settings
from .runtime.baseline_access import open_baseline
from .runtime.contracts import load_config, load_task
from .runtime.judge_contracts import resolve_limits as judge_limits
from .runtime.packages import hash_file, verify_package
from .runtime.provider_adapters import create_adapter
from .runtime.responses_adapter import validate_model
from .asset_download import require

DEFAULT_CONFIG = Path("benchmark.json")
ROLES = ("agent", "judge-1", "judge-2")


def read_json(path):
    return json.loads(Path(path).read_text())


def project_path(base, value):
    require(
        isinstance(value, str) and value.strip(), "Project paths must be nonempty text"
    )
    return (base / Path(value).expanduser()).resolve()


def init_config(path=DEFAULT_CONFIG):
    path = Path(path)
    require(not path.exists(), f"{path} already exists. Edit it in place")

    def model(name, provider="responses"):
        return dict(
            provider=provider,
            model_id=name,
            endpoint=(
                "https://api.anthropic.com/v1/messages"
                if provider == "anthropic"
                else "https://api.openai.com/v1/responses"
            ),
            api_key_env=(
                "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            ),
            max_output_tokens=16384,
        )

    value = dict(
        schema_version=1,
        data_dir=str(DEFAULT_DATA),
        cache_dir=str(DEFAULT_CACHE),
        results_dir="results",
        tasks_dir="tasks",
        agent=model("your-agent-model"),
        judges=[model("your-first-judge"), model("your-second-judge", "anthropic")],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return path


def load_project(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    require(path.is_file(), f"Missing {path.name}. Run benchmark init first")
    value = read_json(path)
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "data_dir",
            "cache_dir",
            "results_dir",
            "tasks_dir",
            "agent",
            "judges",
        }
        and value["schema_version"] == 1,
        "Project config requires schema_version 1, data_dir, cache_dir, results_dir, tasks_dir, agent and two judges",
    )
    require(
        isinstance(value["judges"], list) and len(value["judges"]) == 2,
        "Configure exactly two judges",
    )
    models = dict(
        zip(
            ROLES,
            [
                validate_model(value["agent"]),
                *(validate_model(v) for v in value["judges"]),
            ],
        )
    )
    require(
        models["judge-1"]["model_id"] != models["judge-2"]["model_id"],
        "Choose two distinct judge model IDs",
    )
    require(
        models["judge-1"]["judge_parallel_reads"]
        == models["judge-2"]["judge_parallel_reads"],
        "Both judges must use the same judge_parallel_reads setting",
    )
    for model in models.values():
        require(
            model["model_id"]
            not in {"your-agent-model", "your-first-judge", "your-second-judge"},
            "Set the three model IDs in benchmark.json before running",
        )
    paths = {
        k: project_path(path.parent, value[k])
        for k in ["data_dir", "cache_dir", "results_dir", "tasks_dir"]
    }
    for left, right in combinations(paths, 2):
        require(
            not paths[left].is_relative_to(paths[right])
            and not paths[right].is_relative_to(paths[left]),
            f"{left} and {right} must be separate folders",
        )
    load_dotenv(path.parent / ".env", override=False)
    return dict(config_path=path, models=models, **paths)


def task_catalog(root, selected="all"):
    root = Path(root).resolve()
    index = read_json(root / "index.json")
    require(
        index.get("schema_version") == 1
        and isinstance(index.get("tasks"), list)
        and index["tasks"],
        "Invalid tasks/index.json",
    )
    tasks = []
    seen = set()
    for row in index["tasks"]:
        ident = row["id"]
        require(
            isinstance(ident, str)
            and re.fullmatch("task-[0-9]{2}", ident)
            and ident not in seen,
            "Task IDs must be unique task-01 style names",
        )
        seen.add(ident)
        require(
            row["package"] == f"{ident}/package"
            and row["config"] == f"{ident}/config.json",
            "Task index paths must match their public IDs",
        )
        package, config_path = root / row["package"], root / row["config"]
        require(
            hash_file(package / "manifest.json")[1] == row["manifest_sha256"],
            f"{ident}: package checksum differs from index",
        )
        verify_package(package, config_path=config_path)
        config = load_config(config_path)
        task = load_task(package / "task.json")
        require(
            task["meta"]["task_id"] == task["brief"]["task_id"] == ident
            and task["meta"]["version"] == row["version"],
            f"{ident}: task identity differs from index",
        )
        tasks.append(
            dict(
                **row,
                config_sha256=hash_file(config_path)[1],
                intent=task["brief"]["intent"],
                limits=resolve_limits(config),
                judge_limits=judge_limits(config),
            )
        )
    wanted = (
        selected.split(",")
        if isinstance(selected, str) and selected != "all"
        else [t["id"] for t in tasks]
    )
    require(
        wanted and len(set(wanted)) == len(wanted) and set(wanted) <= seen,
        "Unknown or repeated --task. Use all or comma-separated task IDs",
    )
    return [row for row in tasks if row["id"] in wanted]


def verify_tasks(root):
    return task_catalog(root)


async def check_models(models, *, credentials=True, factory=create_adapter):
    routes = {}
    for name, model in models.items():
        async with factory(model) as adapter:
            needed = (
                credentials is True
                or isinstance(credentials, (set, frozenset))
                and name in credentials
            )
            if needed and hasattr(adapter, "check"):
                adapter.check()
            routes[name] = adapter.route_id
    return routes


def check_inputs(project, tasks, *, release=None, progress=print):
    release = release or Release()
    check_data(
        project["data_dir"],
        cache_dir=project["cache_dir"],
        release=release,
        progress=progress,
    )
    from .runtime import service_registry as services

    with cache_at(project["cache_dir"]) as cache:
        baseline = open_baseline(
            release.lock["dataset"]["build_ref"],
            project["data_dir"],
            cache,
            manifest_sha256=release.lock["dataset"]["manifest_sha256"],
        )
        for task in tasks:
            config = load_config(project["tasks_dir"] / task["config"])
            require(
                config["build_ref"] == release.lock["dataset"]["build_ref"],
                f'{task["id"]}: dataset snapshot differs from the task',
            )
            settings = baseline_settings(
                config, "verified_local", project["data_dir"], None, None
            )
            require(
                settings["manifest_sha256"]
                == release.lock["dataset"]["manifest_sha256"],
                f'{task["id"]}: dataset manifest differs from the task',
            )
            from .runtime.baseline_access import CacheLimits

            cache.limits = CacheLimits(**settings["limits"])
            baseline.check_limits(
                services.required_files(baseline, config["tool_scope"])
            )
    return release.lock["dataset"]


def credentials_for(models):
    from .runtime.failure_causes import configured_secrets

    names = {m["api_key_env"] for m in models.values()} | {
        v for m in models.values() for v in m["headers_env"].values()
    }
    return tuple(
        set(configured_secrets())
        | {os.environ[name] for name in names if name and os.environ.get(name)}
    )


def results_directory(config_path=DEFAULT_CONFIG, override=None):
    path = Path(config_path).resolve()
    if override is not None:
        return Path(override).resolve()
    return (
        project_path(path.parent, read_json(path).get("results_dir", "results"))
        if path.exists()
        else path.parent / "results"
    )
