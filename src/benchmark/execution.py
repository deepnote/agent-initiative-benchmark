"""Durable CLI runs over the unchanged agent and judge protocols."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .asset_download import require
from .configuration import (
    ROLES,
    check_inputs,
    check_models,
    credentials_for,
    read_json,
    task_catalog,
)
from .runtime.agent_runner import run_attempt_async
from .runtime.artifact_store import ArtifactStore, PAYLOAD, StorageError
from .runtime.judge_runner import (
    run_judgment_async,
    prepare_anonymous_answer,
    retry_publication,
)
from .runtime.packages import hash_file
from .runtime.provider_adapters import create_adapter
from .runtime.runtime_identity import fingerprint, write_json
from .runtime.failure_causes import safe_text

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
AGENT_IDENTITY = (
    "execution_protocol",
    "run_id",
    "attempt_id",
    "task_ref",
    "config_ref",
    "model_ref",
    "baseline_ref",
)


def new_id():
    return (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )


@contextmanager
def run_lock(root):
    path = Path(root) / ".run.lock"
    require(not path.is_symlink(), "Run lock cannot be a symlink")
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is already active in another process") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def cache_environment(path):
    previous = os.environ.get("BENCHMARK_CACHE_DIR")
    os.environ["BENCHMARK_CACHE_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BENCHMARK_CACHE_DIR", None)
        else:
            os.environ["BENCHMARK_CACHE_DIR"] = previous


def check_cancelled():
    if asyncio.current_task().cancelling():
        raise asyncio.CancelledError


class Session:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(
            (self.root / "settings.json").is_file(), f"Run not found: {self.root.name}"
        )
        self.settings = read_json(self.root / "settings.json")
        self.progress = read_json(self.root / "progress.json")
        require(
            self.progress["settings_sha256"]
            == hash_file(self.root / "settings.json")[1],
            "Saved run settings changed",
        )
        require(
            self.settings["schema_version"] == self.progress["schema_version"] == 1
            and ID.fullmatch(self.settings["run_id"]),
            "Invalid saved run identity",
        )
        self.tasks = {t["id"]: t for t in self.settings["tasks"]}
        require(
            set(self.progress["tasks"]) == set(self.tasks),
            "Saved task progress differs from the selected tasks",
        )
        for ident, roles in self.progress["tasks"].items():
            require(
                re.fullmatch("task-[0-9]{2}", ident) and set(roles) == set(ROLES),
                "Invalid saved task roles",
            )
            for values in roles.values():
                require(
                    isinstance(values, list)
                    and values == list(range(1, len(values) + 1)),
                    "Invalid saved attempt sequence",
                )
        for role in ROLES:
            require(
                hash_file(self.model_path(role))[1]
                == self.settings["model_sha256"][role],
                f"Saved {role} profile changed",
            )
        self.store = ArtifactStore(self.root, prefix="artifacts/")

    @classmethod
    def create(cls, project, tasks, routes, dataset, *, run_id=None):
        ident = run_id or new_id()
        require(
            isinstance(ident, str) and ID.fullmatch(ident),
            "Run ID may contain letters, numbers, dots, underscores and hyphens",
        )
        root = project["results_dir"] / ident
        require(
            not root.exists(),
            "Run ID already exists. Use resume or choose another --run",
        )
        hashes = {
            role: hashlib.sha256(
                (
                    json.dumps(project["models"][role], indent=2, sort_keys=True) + "\n"
                ).encode()
            ).hexdigest()
            for role in ROLES
        }
        settings = dict(
            schema_version=1,
            run_id=ident,
            created_at=datetime.now(timezone.utc).isoformat(),
            runtime_sha256=fingerprint(),
            models=project["models"],
            routes=routes,
            model_sha256=hashes,
            paths={
                key: str(project[key]) for key in ["data_dir", "cache_dir", "tasks_dir"]
            },
            dataset=dataset,
            tasks=tasks,
        )
        secrets = credentials_for(project["models"])
        raw = json.dumps(settings, sort_keys=True)
        require(
            not any(secret in raw for secret in secrets),
            "Run settings contain a credential. Use environment variable names in model profiles",
        )
        root.mkdir(parents=True)
        (root / "inputs").mkdir()
        for role in ROLES:
            write_json(root / "inputs" / f"{role}.json", project["models"][role])
        write_json(root / "settings.json", settings)
        progress = dict(
            schema_version=1,
            settings_sha256=hash_file(root / "settings.json")[1],
            tasks={t["id"]: {r: [] for r in ROLES} for t in tasks},
            evaluation_errors={},
        )
        write_json(root / "progress.json", progress)
        return cls(root)

    def model_path(self, role):
        return self.root / "inputs" / f"{role}.json"

    def save(self):
        write_json(self.root / "progress.json", self.progress)

    def unit(self, ident, role, number=None):
        values = self.progress["tasks"][ident][role]
        number = number if number is not None else values[-1] if values else None
        return f"{ident}/{role}-{number:03d}" if number is not None else None

    def work(self, unit):
        return self.root / ".work" / unit.replace("/", "--")

    def bundle(self, unit):
        return self.root / self.store.prefix / unit / PAYLOAD

    def payload(self, ident, role, *, number=None, full=False):
        unit = self.unit(ident, role, number)
        if unit is None:
            return None
        marker = self.store.read_completion(unit)
        if marker is None:
            return None
        name = "run.json" if role == "agent" else "judgment.json"
        entries = (
            marker["files"]
            if full
            else [e for e in marker["files"] if e["path"] == name]
        )
        require(
            any(e["path"] == name for e in entries),
            f"{unit}: terminal result is missing",
        )
        for entry in entries:
            self.store.store.verify_object(
                "local",
                self.store._key(unit, PAYLOAD + "/" + entry["path"]),
                entry["bytes"],
                entry["sha256"],
                full=True,
            )
        value = read_json(self.bundle(unit) / name)
        require(
            value["unit_key"] == unit
            and value["task_ref"]["task_id"] == ident
            and value["config_ref"]["sha256"] == self.tasks[ident]["config_sha256"],
            f"{unit}: result identity differs",
        )
        model = value["model_ref"] if role == "agent" else value["judge"]["model_ref"]
        require(
            model["sha256"] == self.settings["model_sha256"][role]
            and model["settings"] == self.settings["models"][role]
            and model["route_id"] == self.settings["routes"][role],
            f"{unit}: model identity differs",
        )
        if role == "agent":
            require(
                value["run_id"] == self.settings["run_id"],
                f"{unit}: run identity differs",
            )
        else:
            require(
                value["judge"]["runtime_sha256"] == self.settings["runtime_sha256"],
                f"{unit}: runtime identity differs",
            )
        return value

    def snapshot_project(self, *, data_dir=None, cache_dir=None):
        return {
            **{k: Path(v) for k, v in self.settings["paths"].items()},
            "data_dir": (
                Path(data_dir).resolve()
                if data_dir
                else Path(self.settings["paths"]["data_dir"])
            ),
            "cache_dir": (
                Path(cache_dir).resolve()
                if cache_dir
                else Path(self.settings["paths"]["cache_dir"])
            ),
            "models": self.settings["models"],
            "results_dir": self.root.parent,
        }

    def verify_resume(self, project):
        require(
            self.settings["runtime_sha256"] == fingerprint(),
            "Runtime changed since this run. Restore the same version or start a new run",
        )
        current = task_catalog(project["tasks_dir"], ",".join(self.tasks))
        require(
            current == self.settings["tasks"],
            "Selected task inputs changed. Restore them or start a new run",
        )
        for ident, roles in self.progress["tasks"].items():
            for role, attempts in roles.items():
                for number in attempts:
                    self.payload(ident, role, number=number, full=True)

    def recover(self, ident, role):
        unit = self.unit(ident, role)
        if unit is None:
            return
        if self.store.is_complete(unit):
            self.payload(ident, role, full=True)
            self.clean_work(unit)
            return
        bundle = self.work(unit) / "bundle"
        path = bundle / ("run.json" if role == "agent" else "judgment.json")
        if path.is_file():
            value = read_json(path)
            require(
                value["unit_key"] == unit,
                f"{unit}: interrupted result identity differs",
            )
            if role == "agent":
                self.store.publish_directory(
                    bundle, unit, {k: value[k] for k in AGENT_IDENTITY}
                )
            else:
                retry_publication(bundle, unit_key=unit, artifact_store=self.store)
            self.clean_work(unit)
        elif self.store.store.list_keys("local", self.store.prefix + unit + "/"):
            raise StorageError(
                f"{unit}: partial result has no terminal bundle. Preserve this run and start a new run"
            )

    def clean_work(self, unit):
        work = self.work(unit)
        if self.store.is_complete(unit) and work.exists():
            shutil.rmtree(work)

    def start(self, ident, role):
        attempts = self.progress["tasks"][ident][role]
        attempts.append(len(attempts) + 1)
        self.save()
        return self.unit(ident, role)

    def retryable(self, ident, role, value):
        if role != "agent":
            return value["status"] == "technical_error"
        return (
            value["outcome"] == "technical_error"
            or value["stop_reason"] in {"worker_timeout", "cancelled"}
            or value["stop_reason"] == "model_timeout"
            and ident in self.progress["evaluation_errors"]
        )

    def needed_roles(self, *, retry_errors=False):
        required = set()
        for ident in self.tasks:

            def pending(role, value):
                return (
                    value is None
                    or (value.get("error") or {}).get("code") == "cancelled"
                    or retry_errors
                    and self.retryable(ident, role, value)
                )

            agent = self.payload(ident, "agent")
            if pending("agent", agent):
                required.update(ROLES)
                continue
            if (
                ident in self.progress["evaluation_errors"]
                or agent["outcome"] != "completed"
                and agent["stop_reason"] not in {"model_timeout", "model_budget"}
            ):
                continue
            for role in ROLES[1:]:
                if pending(role, self.payload(ident, role)):
                    required.add(role)
        return required

    def needs_work(self, *, retry_errors=False):
        return bool(self.needed_roles(retry_errors=retry_errors))


async def execute(
    session, project, *, factory=create_adapter, retry_errors=False, progress=print
):
    """One awaited API budget per core unit. A resume starts only missing work."""
    session.store = ArtifactStore(
        session.root, prefix="artifacts/", secrets=credentials_for(project["models"])
    )
    total = len(session.tasks)
    for position, (ident, task) in enumerate(session.tasks.items(), 1):
        check_cancelled()
        config_path = project["tasks_dir"] / task["config"]
        package = project["tasks_dir"] / task["package"]
        common = dict(
            package=package,
            config_path=config_path,
            baseline_dir=project["data_dir"],
            baseline_cache_dir=project["cache_dir"],
            artifact_store=session.store,
        )

        async def stage(role, prepared=None):
            check_cancelled()
            session.recover(ident, role)
            saved = session.payload(ident, role)
            interrupted = (
                saved is not None
                and (saved.get("error") or {}).get("code") == "cancelled"
            )
            retry = (
                saved is not None
                and retry_errors
                and session.retryable(ident, role, saved)
            )
            if saved is not None and not interrupted and not retry:
                session.clean_work(session.unit(ident, role))
                progress(f"[{position}/{total}] {ident}: {role} saved")
                return saved
            previous = session.unit(ident, role)
            unit = session.start(ident, role)
            progress(
                f"[{position}/{total}] {ident}: {role}"
                + (
                    f" restarting interrupted or failed attempt (previous: {previous})"
                    if previous
                    else ""
                )
            )
            async with factory(project["models"][role]) as adapter:
                require(
                    adapter.route_id == session.settings["routes"][role],
                    f"{role}: API settings or custom adapter changed",
                )
                if role == "agent":
                    await run_attempt_async(
                        **common,
                        model_path=session.model_path(role),
                        work_dir=session.work(unit),
                        run_id=session.settings["run_id"],
                        attempt_id=unit.split("/")[1],
                        unit_key=unit,
                        adapter=adapter,
                    )
                else:
                    await run_judgment_async(
                        **common,
                        model_path=session.model_path(role),
                        run_bundle=session.bundle(session.unit(ident, "agent")),
                        work_dir=session.work(unit),
                        judgment_id=unit.replace("/", "-"),
                        unit_key=unit,
                        adapter=adapter,
                        anonymous_preparation=prepared,
                    )
            saved = session.payload(ident, role)
            session.clean_work(unit)
            check_cancelled()
            return saved

        agent = await stage("agent")
        session.progress["evaluation_errors"].pop(ident, None)
        if agent["outcome"] != "completed" and agent["stop_reason"] not in {
            "model_timeout",
            "model_budget",
        }:
            progress(f'{ident}: agent stopped ({agent["stop_reason"]}). No score')
            continue
        if all(
            session.payload(ident, r) is not None
            and not (
                (session.payload(ident, r).get("error") or {}).get("code")
                == "cancelled"
            )
            and not (
                retry_errors
                and session.payload(ident, r)["status"] == "technical_error"
            )
            for r in ROLES[1:]
        ):
            continue
        try:
            prepared = prepare_anonymous_answer(
                package=package,
                config_path=config_path,
                run_bundle=session.bundle(session.unit(ident, "agent")),
            )
            body = json.loads(prepared.input_bytes)
            require(
                agent["outcome"] == "completed"
                or body["attempt"]["model_budget_termination"],
                "Agent stop is not a verified model-time budget stop",
            )
        except (ValueError, RuntimeError, OSError) as error:
            session.progress["evaluation_errors"][ident] = safe_text(str(error))
            session.save()
            progress(f"{ident}: evaluation unavailable. No score")
            continue
        for role in ROLES[1:]:
            await stage(role, prepared)
        session.save()
        from .reporting import build_summary, save_summary

        summary = build_summary(session)
        save_summary(session, summary)
        row = next(r for r in summary["tasks"] if r["id"] == ident)
        progress(
            f'{ident}: {row["status"]}'
            + (f' {row["overall"]:.2f}/10' if row["overall"] is not None else "")
        )


async def start_run(
    project, selected="all", *, run_id=None, factory=create_adapter, progress=print
):
    tasks = task_catalog(project["tasks_dir"], selected)
    routes = await check_models(project["models"], factory=factory)
    dataset = check_inputs(project, tasks, progress=progress)
    session = Session.create(project, tasks, routes, dataset, run_id=run_id)
    progress(f'Run {session.settings["run_id"]} | {len(tasks)} public example tasks')
    show_settings(session, progress)
    await drive(session, project, factory=factory, progress=progress)
    return session


async def resume_run(
    session,
    *,
    data_dir=None,
    cache_dir=None,
    retry_errors=False,
    factory=create_adapter,
    progress=print,
):
    with run_lock(session.root):
        for ident in session.tasks:
            for role in ROLES:
                session.recover(ident, role)
        from .reporting import build_summary, save_summary

        save_summary(session, build_summary(session))
    if not session.needs_work(retry_errors=retry_errors):
        progress("All selected work is saved. No API call needed")
        return session
    project = session.snapshot_project(data_dir=data_dir, cache_dir=cache_dir)
    session.verify_resume(project)
    check_inputs(project, list(session.tasks.values()), progress=progress)
    routes = await check_models(
        project["models"],
        credentials=session.needed_roles(retry_errors=retry_errors),
        factory=factory,
    )
    require(
        routes == session.settings["routes"],
        "API settings or custom adapter changed. Restore the saved implementation",
    )
    show_settings(session, progress)
    await drive(
        session, project, factory=factory, retry_errors=retry_errors, progress=progress
    )
    return session


def show_settings(session, progress):
    for role, model in session.settings["models"].items():
        progress(
            f'{role}: {model["model_id"]} ({model["provider"]}, effort {model["reasoning_effort"] or "API default"}, max output {model["max_output_tokens"]})'
        )
    judges = sorted(
        {t["judge_limits"]["judge_model_seconds"] for t in session.tasks.values()}
    )
    progress(
        f'Judge API-time budget: {", ".join(map(str,judges))} seconds per judge per task'
    )
    budgets = sorted(
        {(t["limits"]["agent_model_seconds"]) for t in session.tasks.values()}
    )
    progress(
        f'Agent API-time budget: {", ".join(map(str,budgets))} seconds per task. Results: {session.root}'
    )


async def drive(
    session, project, *, factory=create_adapter, retry_errors=False, progress=print
):
    from .reporting import build_summary, save_summary

    with run_lock(session.root), cache_environment(project["cache_dir"]):
        session.progress = read_json(session.root / "progress.json")
        try:
            await execute(
                session,
                project,
                factory=factory,
                retry_errors=retry_errors,
                progress=progress,
            )
        finally:
            session.save()
            save_summary(session, build_summary(session))
