"""Run the public example set and inspect saved results."""

import argparse
import asyncio
import json
import tarfile

from .runtime.baseline_access import BaselineAccessError
from pathlib import Path

from .runtime.packages import verify_package, hash_file
from .runtime.runtime_identity import fingerprint
from .runtime.artifact_store import StorageError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark.json"),
        help="Project configuration",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify task package checksums")
    verify.add_argument("--tasks", type=Path, default=Path("tasks"))
    model = commands.add_parser("model", help="Check a model profile")
    actions = model.add_subparsers(dest="action", required=True)
    check = actions.add_parser(
        "check", help="Check configuration and credentials without a model call"
    )
    check.add_argument("profile", type=Path)
    data = commands.add_parser(
        "data", help="Download, prepare or check the fixed dataset"
    )
    data_actions = data.add_subparsers(dest="action", required=True)
    from .dataset import DEFAULT_DATA, DEFAULT_CACHE

    for name, help_text in [
        ("download", "Download and extract the pinned dataset"),
        ("prepare", "Extract already downloaded release archives"),
        ("check", "Verify extracted data and search weights"),
    ]:
        action = data_actions.add_parser(name, help=help_text)
        action.add_argument(
            "--dir",
            type=Path,
            default=DEFAULT_DATA,
            help="Extracted dataset folder (default: data/pagemantle)",
        )
        action.add_argument(
            "--cache-dir",
            type=Path,
            default=DEFAULT_CACHE,
            help="Shared verification cache",
        )
        if name == "check":
            action.add_argument(
                "--full",
                action="store_true",
                help="Rehash every file, bypassing cached verification",
            )
        else:
            action.add_argument(
                "--search-from",
                type=Path,
                help="Use a local folder containing the five pinned search model files",
            )
            action.add_argument(
                "--dry-run",
                action="store_true",
                help="Show sizes and location without changing files or using the network",
            )
            if name == "prepare":
                action.add_argument(
                    "--from",
                    dest="source",
                    type=Path,
                    required=True,
                    help="Release folder containing manifest.json, file-index.jsonl.gz and data/*.tar.gz",
                )
            else:
                action.add_argument(
                    "--repository", help="Hugging Face organization/name"
                )
                action.add_argument(
                    "--revision", help="Full fixed Hugging Face commit hash"
                )
    for name, help_text in [
        ("init", "Create a small project configuration"),
        ("tasks", "List public tasks"),
        ("check", "Check data, task packages and model profiles without API calls"),
        ("run", "Run agents and two independent judges"),
        ("resume", "Continue missing or interrupted work"),
        ("results", "Show saved scores"),
        ("inspect", "Show a task's messages and judge reasons"),
    ]:
        action = commands.add_parser(name, help=help_text)
        action.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        if name in {"check", "run"}:
            action.add_argument(
                "--task", default="all", help="all or comma-separated task-01 style IDs"
            )
        if name == "run":
            action.add_argument("--run", help="Optional new run ID")
        if name in {"results", "inspect", "resume"}:
            action.add_argument(
                "--run", help="Saved run ID. Defaults to the latest run"
            )
            action.add_argument("--results-dir", type=Path, help="Saved results folder")
        if name == "resume":
            action.add_argument(
                "--retry-errors",
                action="store_true",
                help="Retry technical failures as new attempts, keeping successful work",
            )
            action.add_argument(
                "--data-dir", type=Path, help="Relocated copy of the same pinned data"
            )
            action.add_argument(
                "--cache-dir", type=Path, help="Relocated writable cache"
            )
        if name == "results":
            action.add_argument("--list", action="store_true", help="List saved runs")
            action.add_argument(
                "--json", action="store_true", help="Print machine-readable results"
            )
            action.add_argument("--csv", type=Path, help="Export the task table as CSV")
        if name == "inspect":
            action.add_argument("--task", required=True, help="Task ID")
    args = parser.parse_args()
    try:
        if args.command in {
            "init",
            "tasks",
            "check",
            "run",
            "resume",
            "results",
            "inspect",
        }:
            return project_command(args)
        if args.command == "data":
            from .dataset import prepare_data, check_data

            if args.action == "check":
                check_data(args.dir, cache_dir=args.cache_dir, full=args.full)
            else:
                prepare_data(
                    args.dir,
                    cache_dir=args.cache_dir,
                    source=getattr(args, "source", None),
                    repository=getattr(args, "repository", None),
                    revision=getattr(args, "revision", None),
                    search_source=args.search_from,
                    dry_run=args.dry_run,
                )
            return
        if args.command == "model":
            from dotenv import load_dotenv
            from .runtime.responses_adapter import load_model
            from .runtime.provider_adapters import create_adapter

            load_dotenv(Path(args.config).resolve().parent / ".env", override=False)
            settings = load_model(args.profile)

            async def check_model():
                async with create_adapter(settings) as adapter:
                    if hasattr(adapter, "check"):
                        adapter.check()

            asyncio.run(check_model())
            print(
                f"Valid model profile: {settings['model_id']} ({settings['provider']}). No API call made."
            )
            return
        index = json.loads((args.tasks / "index.json").read_text())
        for task in index["tasks"]:
            package = args.tasks / task["package"]
            if hash_file(package / "manifest.json")[1] != task["manifest_sha256"]:
                raise ValueError(f"{task['id']}: package checksum differs from index")
            verify_package(package, config_path=args.tasks / task["config"])
        print(
            f"Verified {len(index['tasks'])} task packages. Runtime {fingerprint()[:12]}."
        )
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted. Saved work is preserved. Use benchmark resume for a run, or repeat the data command.\n",
        )
    except (
        ValueError,
        OSError,
        BaselineAccessError,
        tarfile.TarError,
        StorageError,
    ) as error:
        from .runtime.failure_causes import safe_text

        parser.exit(1, f"Error: {safe_text(str(error))}\n")
    except Exception as error:
        parser.exit(1, f"Error: {type(error).__name__}. Saved work is preserved.\n")


def project_command(args):
    import sys
    from .configuration import (
        init_config,
        load_project,
        task_catalog,
        check_inputs,
        check_models,
        results_directory,
        read_json,
    )
    from .execution import Session, start_run, resume_run, ID
    from .reporting import (
        build_summary,
        format_summary,
        inspect_task,
        export_csv,
        display,
    )
    from .runtime.provider_adapters import create_adapter
    from .asset_download import require

    def progress(value):
        print(display(value), file=sys.stderr, flush=True)

    if args.command == "init":
        path = init_config(args.config)
        print(
            f"Created {path}. Edit the three model profiles and set their environment keys. Then run benchmark check."
        )
        return
    if args.command == "tasks":
        path = Path(args.config).resolve()
        directory = (
            (path.parent / read_json(path).get("tasks_dir", "tasks")).resolve()
            if path.exists()
            else path.parent / "tasks"
        )
        for task in task_catalog(directory):
            print(f'{task["id"]}  {task["category"]:<10}  {task["title"]}')
        return
    if args.command in {"check", "run"}:
        project = load_project(args.config)
        if args.command == "check":
            tasks = task_catalog(project["tasks_dir"], args.task)
            check_inputs(project, tasks, progress=progress)
            asyncio.run(check_models(project["models"], factory=create_adapter))
            print(
                f"Ready: {len(tasks)} task packages, data and three model profiles. No API call made."
            )
            return
        session = asyncio.run(
            start_run(
                project,
                args.task,
                run_id=args.run,
                factory=create_adapter,
                progress=progress,
            )
        )
    else:
        root = results_directory(args.config, args.results_dir)
        runs = (
            sorted(
                [
                    p
                    for p in root.iterdir()
                    if p.is_dir() and (p / "settings.json").is_file()
                ],
                key=lambda p: read_json(p / "settings.json")["created_at"],
            )
            if root.is_dir()
            else []
        )
        if args.command == "results" and args.list:
            if not runs:
                print("No saved runs. Start one with benchmark run.")
                return
            for path in runs:
                saved = Session(path)
                summary = build_summary(saved)
                print(
                    display(
                        f'{saved.settings["run_id"]}  {saved.settings["models"]["agent"]["model_id"]}  {summary["scored"]}/{summary["selected"]} scored  {summary["overall"] if summary["overall"] is not None else "-"}'
                    )
                )
            return
        ident = args.run or (runs[-1].name if runs else None)
        require(ident is not None, "No saved runs. Start one with benchmark run")
        require(isinstance(ident, str) and ID.fullmatch(ident), "Invalid run ID")
        session = Session(root / ident)
        if args.command == "inspect":
            print(inspect_task(session, args.task))
            return
        if args.command == "resume":
            from dotenv import load_dotenv

            load_dotenv(Path(args.config).resolve().parent / ".env", override=False)
            asyncio.run(
                resume_run(
                    session,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    retry_errors=args.retry_errors,
                    factory=create_adapter,
                    progress=progress,
                )
            )
    summary = build_summary(session)
    if getattr(args, "csv", None):
        export_csv(summary, args.csv)
        progress(f"CSV: {args.csv}")
    print(
        json.dumps(summary, indent=2)
        if getattr(args, "json", False)
        else format_summary(summary)
    )
    if args.command in {"run", "resume"}:
        progress(f'View later: benchmark results --run {session.settings["run_id"]}')
        if summary["errors"] or summary["pending"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
