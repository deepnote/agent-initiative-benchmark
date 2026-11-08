"""Run a bounded Slack attempt and save its completed result."""

from __future__ import annotations

from benchmark.prompts import load_text

from contextlib import closing
import asyncio
import logging
import hashlib
import math
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .artifact_store import StorageError, _safe_reason
from .attempt_worker import SlackWorker, WorkerFailure
from .baseline_access import BaselineAccessError, CacheLimits, validate_source
from .contracts import load_config, load_task
from .packages import hash_file, verify_package
from .failure_causes import (
    RuntimeInputError,
    exception_cause,
    provider_body,
    provider_cause,
    safe_text,
)
from .responses_adapter import load_model, model_response
from .slack_environment import SlackEnvironment, _encoded
from . import service_registry as services
from . import read_result_admission as read_admission
from .agent_protocol import (
    execution_protocol,
    preflight,
    arguments as _arguments,
    MAX_READ_CALLS,
    HARD_MAX_READ_CALLS,
    render_instruction,
)

DEFAULT_LIMITS = {
    "agent_model_seconds": 180,
    "max_read_calls_per_response": MAX_READ_CALLS,
    "recent_activity_window_minutes": 10,
    "tool_timeout_seconds": 60,
    "environment_setup_timeout_seconds": 180,
    "initial_context_max_bytes": 200_000,
    "max_result_bytes": 1_048_576,
    "max_model_input_bytes": 2_097_152,
    "max_trace_bytes": 134_217_728,
}
FINISH = {
    "type": "function",
    "name": "finish_task",
    "strict": False,
    "description": load_text("tools/finish_task.txt"),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}
REMINDER = load_text("agent/finish_reminder.txt")
WRITES = services.write_names()


class AttemptError(RuntimeError):
    def __init__(self, code: str, reason: str, *, cause=None):
        self.code, self.cause = code, cause
        super().__init__(reason)


class PublicationError(RuntimeError):
    """Retain the completed bundle if saving it needs another attempt."""

    def __init__(self, bundle, unit_key, reason):
        self.bundle, self.unit_key = bundle, unit_key
        super().__init__(
            f"publication failed: {reason}; retry unchanged bundle {bundle} at unit {unit_key}"
        )


def resolve_limits(config: dict) -> dict:
    values = {
        name: config["limits"].get(name, default)
        for name, default in DEFAULT_LIMITS.items()
    }
    integer_names = {name for name in values if name.endswith("_bytes")} | {
        "recent_activity_window_minutes",
        "max_read_calls_per_response",
    }
    for name, value in values.items():
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            or (name in integer_names and type(value) is not int)
        ):
            raise RuntimeInputError(
                f"limits.{name}: expected a positive {'integer' if name in integer_names else 'number'}"
            )
    if values["max_result_bytes"] < 512:
        raise RuntimeInputError(
            "limits.max_result_bytes: must be at least 512 for successful receipts"
        )
    if values["max_read_calls_per_response"] > HARD_MAX_READ_CALLS:
        raise RuntimeInputError(
            f"limits.max_read_calls_per_response: maximum is {HARD_MAX_READ_CALLS}"
        )
    return values


def baseline_settings(
    config,
    mode,
    baseline_dir,
    baseline_build_ref,
    linear_baseline_dir=None,
    notion_baseline_dir=None,
    stripe_baseline_dir=None,
    calendar_baseline_dir=None,
    service_baseline_dirs=None,
):
    """Resolve source without opening company storage or downloading bytes."""
    if isinstance(notion_baseline_dir, dict):
        service_baseline_dirs = services.local_roots(
            service_baseline_dirs, **notion_baseline_dir
        )
        notion_baseline_dir = None
    service_baseline_dirs = services.local_roots(
        service_baseline_dirs,
        notion=notion_baseline_dir,
        stripe=stripe_baseline_dir,
        calendar=calendar_baseline_dir,
    )
    notion_baseline_dir = service_baseline_dirs.pop("notion", None)
    settings = config["storage"].get("baseline")
    if set(service_baseline_dirs or {}) & {"slack", "linear"}:
        raise RuntimeInputError(
            "Use the existing explicit Slack/Linear baseline arguments"
        )
    services.validate_local_inputs(
        config["tool_scope"],
        mode,
        {
            "linear": linear_baseline_dir,
            "notion": notion_baseline_dir,
            **(service_baseline_dirs or {}),
        },
    )
    if mode == "local_declared":
        if settings is not None or baseline_dir is None or baseline_build_ref is None:
            raise RuntimeInputError(
                "local_declared requires baseline_dir and baseline_build_ref, without storage.baseline"
            )
        if baseline_build_ref != config["build_ref"]:
            raise RuntimeInputError(
                "baseline_build_ref must match pinned config; declaration is not byte verification"
            )
        return None
    if mode != "verified_local":
        raise RuntimeInputError(
            "baseline_mode must be local_declared or verified_local"
        )
    if baseline_dir is None:
        raise RuntimeInputError(
            "verified_local requires the extracted dataset directory"
        )
    if baseline_build_ref is not None and baseline_build_ref != config["build_ref"]:
        raise RuntimeInputError("Local source pin differs from config")
    fields = {
        "manifest_sha256",
        "cache_dir",
        "max_cache_bytes",
        "max_file_bytes",
        "min_free_bytes",
        "lock_wait_seconds",
    }
    optional = {"ready_max_bytes", "manifest_max_bytes", "source"}
    if (
        not isinstance(settings, dict)
        or not fields.issubset(settings)
        or set(settings) - fields - optional
    ):
        raise RuntimeInputError(
            "storage.baseline requires manifest_sha256, cache_dir and explicit cache/file/free-space/lock limits"
        )
    for name in ("manifest_sha256", "cache_dir"):
        value = settings[name]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise RuntimeInputError(
                f"storage.baseline.{name} must be explicit nonempty text"
            )
    values = {
        name: value
        for name, value in settings.items()
        if name not in {"manifest_sha256", "cache_dir", "source"}
    }
    try:
        CacheLimits(**values)
        source = validate_source(settings.get("source"), config["build_ref"])
    except BaselineAccessError as exc:
        raise RuntimeInputError(safe_text(str(exc))) from None
    return {
        "root": str(Path(baseline_dir).resolve()),
        "manifest_sha256": settings["manifest_sha256"],
        "cache_dir": str(Path(settings["cache_dir"]).expanduser().absolute()),
        "limits": values,
        **({"source": source} if source else {}),
    }


def baseline_identity(mode, build_ref, settings=None):
    return {
        "mode": mode,
        "build_ref": build_ref,
        **({"manifest_sha256": settings["manifest_sha256"]} if settings else {}),
        **(
            {"source": settings["source"]}
            if settings and settings.get("source")
            else {}
        ),
    }


def digest(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


class Trace:
    # Error/stop events have their own small reserve, even with a one-byte cap.
    RESERVE = 65_536

    def __init__(self, path, limit, started):
        self.file = path.open("xb")
        self.limit, self.started = limit, started
        self.seq, self.size, self.reserve_used = 0, 0, 0
        self.model_seconds = 0.0
        self.evidence_size = 0

    def ensure(self, size):
        if self.size + self.evidence_size + size > self.limit:
            raise AttemptError(
                "trace_limit", "Trace exceeds max_trace_bytes; raise the pinned limit"
            )

    def append(self, kind, payload, *, terminal=False, **ids):
        event = {
            "seq": self.seq + 1,
            "kind": kind,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "charged_model_seconds": self.model_seconds,
            **ids,
            "payload": payload,
        }
        encoded = _encoded(event) + b"\n"
        if terminal:
            if self.reserve_used + len(encoded) > self.RESERVE:
                raise AttemptError(
                    "terminal_trace_limit", "Terminal error reserve exceeded"
                )
            self.reserve_used += len(encoded)
        else:
            self.ensure(len(encoded))
            self.size += len(encoded)
        self.file.write(encoded)
        self.file.flush()
        self.seq += 1

    def close(self):
        self.file.close()

    def append_native(self, payload):
        # Same byte budget, separate private file. Never a UI/judge source event.
        encoded = _encoded(payload) + b"\n"
        self.ensure(len(encoded))
        with Path(self.file.name).with_name("provider-native.jsonl").open(
            "ab"
        ) as stream:
            stream.write(encoded)
        self.size += len(encoded)


def _freeze_slack(source: Path, destination: Path):
    destination.mkdir(parents=True)
    if not source.exists():
        return
    # SQLite backup recovers committed WAL state after worker cancellation.
    database = source / "overlay.sqlite"
    if database.exists():
        with closing(sqlite3.connect(database)) as original, closing(
            sqlite3.connect(destination / "overlay.sqlite")
        ) as final:
            original.backup(final)
    seed = source / "seed.json"
    if seed.exists():
        shutil.copyfile(seed, destination / "seed.json")


def run_attempt(**kwargs):
    """Run the asynchronous runtime from synchronous Python code."""
    return asyncio.run(run_attempt_async(**kwargs))


async def run_attempt_async(
    *,
    package,
    config_path,
    model_path,
    baseline_dir=None,
    baseline_build_ref=None,
    work_dir,
    run_id,
    attempt_id,
    unit_key,
    adapter,
    artifact_store=None,
    offline=False,
    environment_factory=SlackEnvironment,
    baseline_mode="verified_local",
    baseline_store_factory=None,
    linear_baseline_dir=None,
    notion_baseline_dir=None,
    stripe_baseline_dir=None,
    calendar_baseline_dir=None,
    baseline_cache_dir=None,
    service_baseline_dirs=None,
) -> dict:
    service_baseline_dirs = services.local_roots(
        service_baseline_dirs,
        notion=notion_baseline_dir,
        stripe=stripe_baseline_dir,
        calendar=calendar_baseline_dir,
    )
    notion_baseline_dir = service_baseline_dirs.pop("notion", None)
    """Adapter implements async respond(request, timeout_seconds) -> ModelResponse.

    No automatic retry. offline=True is mandatory without ArtifactStore.
    Worker factories must be importable (spawn); normally use SlackEnvironment.
    local_declared preserves caller-declared paths. verified_local uses nested
    storage.baseline settings and creates baseline_store_factory() in the child.
    """
    return await _run_attempt(
        package=package,
        config_path=config_path,
        model_path=model_path,
        baseline_dir=baseline_dir,
        baseline_build_ref=baseline_build_ref,
        work_dir=work_dir,
        run_id=run_id,
        attempt_id=attempt_id,
        unit_key=unit_key,
        adapter=adapter,
        artifact_store=artifact_store,
        offline=offline,
        environment_factory=environment_factory,
        baseline_mode=baseline_mode,
        baseline_store_factory=baseline_store_factory,
        linear_baseline_dir=linear_baseline_dir,
        notion_baseline_dir=notion_baseline_dir,
        stripe_baseline_dir=stripe_baseline_dir,
        calendar_baseline_dir=calendar_baseline_dir,
        baseline_cache_dir=baseline_cache_dir,
        service_baseline_dirs=service_baseline_dirs,
    )


async def _run_attempt(
    *,
    package,
    config_path,
    model_path,
    baseline_dir,
    baseline_build_ref,
    work_dir,
    run_id,
    attempt_id,
    unit_key,
    adapter,
    artifact_store,
    offline,
    environment_factory,
    baseline_mode="verified_local",
    baseline_store_factory=None,
    linear_baseline_dir=None,
    notion_baseline_dir=None,
    stripe_baseline_dir=None,
    calendar_baseline_dir=None,
    baseline_cache_dir=None,
    service_baseline_dirs=None,
):
    service_baseline_dirs = services.local_roots(
        service_baseline_dirs,
        notion=notion_baseline_dir,
        stripe=stripe_baseline_dir,
        calendar=calendar_baseline_dir,
    )
    notion_baseline_dir = service_baseline_dirs.pop("notion", None)
    started = time.monotonic()
    config, model = load_config(config_path), load_model(model_path)
    limits = resolve_limits(config)
    unsupported = set(config["tool_scope"]) - services.tool_names()
    if unsupported:
        raise RuntimeInputError(
            "Attempt runner: unsupported tool_scope: " + ", ".join(sorted(unsupported))
        )
    verify_package(package, config_path=config_path)
    task = load_task(Path(package) / "task.json")
    settings = baseline_settings(
        config,
        baseline_mode,
        baseline_dir,
        baseline_build_ref,
        linear_baseline_dir,
        notion_baseline_dir=notion_baseline_dir,
        service_baseline_dirs=service_baseline_dirs,
    )
    if baseline_cache_dir is not None:
        if settings is None:
            raise RuntimeInputError("baseline_cache_dir requires verified_local")
        settings["cache_dir"] = str(Path(baseline_cache_dir).expanduser().resolve())
    for name, value in (
        ("run_id", run_id),
        ("attempt_id", attempt_id),
        ("unit_key", unit_key),
    ):
        if not isinstance(value, str) or not value.strip():
            raise RuntimeInputError(name + " must be explicit nonempty text")
    if artifact_store is None and not offline:
        raise RuntimeInputError(
            "ArtifactStore is required unless offline=True explicitly"
        )
    task_ref = {
        "task_id": task["meta"]["task_id"],
        "version": task["meta"]["version"],
        "manifest_sha256": hash_file(Path(package) / "manifest.json")[1],
        "build_ref": config["build_ref"],
    }
    model_ref = {
        "sha256": hash_file(model_path)[1],
        "settings": model,
        "route_id": getattr(adapter, "route_id", "unavailable"),
    }
    protocol = execution_protocol(
        limits,
        provider_admission=getattr(adapter, "quota_protocol", None),
        read_result_admission=read_admission.profile(adapter, model),
    )
    identity = {
        "execution_protocol": protocol,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "task_ref": task_ref,
        "config_ref": task["meta"]["config_ref"],
        "model_ref": model_ref,
        "baseline_ref": baseline_identity(baseline_mode, config["build_ref"], settings),
    }
    instruction = render_instruction(config, limits)
    instruction_bytes = instruction.encode("utf-8")
    instruction_sha256 = hashlib.sha256(instruction_bytes).hexdigest()
    if artifact_store is not None:
        try:
            marker = artifact_store.read_completion(unit_key)
        except StorageError:
            raise AttemptError(
                "completed_identity_conflict",
                "Completed unit and its instruction identity could not be verified. Preserve existing artifacts and choose a fresh unit",
            ) from None
        if marker is not None:
            if marker["metadata"] != identity:
                raise AttemptError(
                    "completed_identity_conflict",
                    "Completed unit has different task/config/model/run identity or baseline source/pin; choose a fresh unit",
                )
            saved_instruction = next(
                (
                    entry
                    for entry in marker["files"]
                    if entry["path"] == "instructions.txt"
                ),
                None,
            )
            if (
                saved_instruction is None
                or saved_instruction["sha256"] != instruction_sha256
                or saved_instruction["bytes"] != len(instruction_bytes)
            ):
                raise AttemptError(
                    "completed_identity_conflict",
                    "Completed unit has different or unverifiable rendered agent instructions. Preserve the existing completion and choose a fresh unit",
                )
            saved_protocol = next(
                (
                    entry
                    for entry in marker["files"]
                    if entry["path"] == "execution-protocol.json"
                ),
                None,
            )
            if (
                saved_protocol is None
                or saved_protocol["sha256"] != digest(protocol)
                or saved_protocol["bytes"] != len(_encoded(protocol))
            ):
                raise AttemptError(
                    "completed_identity_conflict",
                    "Completed unit has missing or conflicting execution protocol artifact. Preserve existing artifacts and choose a fresh unit",
                )
            return {"skipped": True, "completion": marker}
        # A missing completion marker does not prove the unit is absent from the
        # missing marker: partial immutable objects must never get a fresh run.
        prefix = f"{artifact_store.prefix}{unit_key}/"
        if artifact_store.store.list_keys(artifact_store.bucket, prefix):
            raise AttemptError(
                "partial_publication",
                "Unit contains partial publication; retry its unchanged terminal directory or choose a fresh attempt/unit",
            )
    work = Path(work_dir).resolve()
    for input_path in (
        package,
        config_path,
        model_path,
        *[
            p
            for p in (
                baseline_dir,
                linear_baseline_dir,
                notion_baseline_dir,
                *(service_baseline_dirs or {}).values(),
            )
            if p is not None
        ],
    ):
        immutable = Path(input_path).resolve()
        if work.is_relative_to(immutable) or immutable.is_relative_to(work):
            raise RuntimeInputError("work_dir must be separate from immutable inputs")
    if settings is not None:
        cache = Path(settings["cache_dir"]).resolve()
        for input_path in (package, config_path, model_path, work):
            path = Path(input_path).resolve()
            if path.is_relative_to(cache) or cache.is_relative_to(path):
                raise RuntimeInputError(
                    "baseline cache must be separate from immutable inputs and work_dir"
                )
    if work.exists():
        raise RuntimeInputError(
            "work_dir must be fresh and absent; never rerun a saved/uncertain attempt"
        )
    work.mkdir(parents=True)
    bundle = work / "bundle"
    bundle.mkdir()
    trace = Trace(bundle / "trace.jsonl", limits["max_trace_bytes"], started)
    if hasattr(adapter, "serialize_request"):
        (bundle / "provider-native.jsonl").touch(exist_ok=False)
    tools, initial = [], None
    worker = None
    model_used = tool_wait = setup_seconds = 0.0
    quota_wait = backoff = 0.0
    provider_timing = None
    managed = hasattr(adapter, "respond_with_budget")
    messages, uncertain, provider = [], [], []
    outcome, stop_reason, error = "technical_error", "setup_failure", None
    phase, active_request, active_call = "setup", None, None
    graceful = True
    try:
        setup_start = time.monotonic()
        worker = SlackWorker(
            {
                "package": package,
                "config_path": config_path,
                "baseline_dir": baseline_dir,
                "baseline_build_ref": config["build_ref"],
                "baseline_settings": settings,
                "linear_baseline_dir": linear_baseline_dir,
                "notion_baseline_dir": notion_baseline_dir,
                "service_baseline_dirs": service_baseline_dirs,
                "work_dir": work / "environment",
                "initial_context_max_bytes": limits["initial_context_max_bytes"],
                "recent_activity_window_minutes": limits[
                    "recent_activity_window_minutes"
                ],
                "max_result_bytes": limits["max_result_bytes"],
            },
            environment_factory,
            baseline_store_factory,
        )
        try:
            ready = await worker.receive_async(
                max(
                    0,
                    limits["environment_setup_timeout_seconds"]
                    - (time.monotonic() - setup_start),
                )
            )
        finally:
            setup_seconds = time.monotonic() - setup_start
        initial = ready["context"]
        tools = [
            {"type": "function", "strict": False, **services.agent_definition(tool)}
            for tool in ready["tools"]
        ] + [FINISH]
        trace.append(
            "context",
            {"instruction": instruction, "tools": tools, "initial_input": initial},
        )
        history = [
            {"role": "developer", "content": instruction},
            {"role": "user", "content": _encoded(initial).decode()},
        ]
        request_number = 0
        all_call_ids = set()
        while True:
            phase = "model"
            active_call = None
            remaining = limits["agent_model_seconds"] - model_used
            if remaining <= 0:
                raise AttemptError(
                    "model_budget",
                    "Model-response budget exhausted without finish_task",
                )
            request_number += 1
            active_request = f"request-{request_number}"
            request = read_admission.request(
                model, history, tools, protocol, model_used
            )
            if len(_encoded(request)) > limits["max_model_input_bytes"]:
                raise AttemptError(
                    "model_input_limit",
                    "Complete model request exceeds max_model_input_bytes; history was not omitted",
                )
            if hasattr(adapter, "serialize_request"):
                native_request = adapter.serialize_request(request)
                if len(_encoded(native_request)) > limits["max_model_input_bytes"]:
                    raise AttemptError(
                        "model_input_limit",
                        "Complete native model request exceeds max_model_input_bytes; history was not omitted",
                    )
                trace.append_native(
                    {
                        "kind": "request",
                        "request_id": active_request,
                        "route_id": adapter.route_id,
                        "payload": native_request,
                    }
                )
            trace.append(
                "model_request",
                request,
                request_id=active_request,
                deadline_seconds=remaining,
            )
            provider_timing = dict(
                model_seconds=0.0,
                quota_wait_seconds=0.0,
                backoff_seconds=0.0,
                attempts=[],
            )
            try:
                response = await model_response(
                    adapter,
                    request,
                    timeout_seconds=remaining,
                    timing=provider_timing,
                    monotonic=time.monotonic,
                )
            finally:
                model_used += provider_timing["model_seconds"]
                quota_wait += provider_timing["quota_wait_seconds"]
                backoff += provider_timing["backoff_seconds"]
                trace.model_seconds = model_used
            body = provider_body(response.body, http_status=response.http_status)
            response.request_id = safe_text(response.request_id, limit=256)
            if getattr(response, "native_trace", None) is not None:
                trace.append_native(
                    {
                        "kind": "response",
                        "request_id": active_request,
                        **{
                            k: v
                            for k, v in response.native_trace.items()
                            if k not in {"request", "request_id"}
                        },
                        "provider_request_id": response.request_id,
                    }
                )
            trace.append(
                "model_response",
                body,
                request_id=active_request,
                provider_request_id=response.request_id,
                http_status=response.http_status,
                **({"provider_timing": provider_timing} if managed else {}),
            )
            provider.append(
                {
                    "request_id": response.request_id or "unavailable",
                    "response_id": body.get("id", "unavailable"),
                    "model": body.get("model", "unavailable"),
                    "version": body.get("model_version", "unavailable"),
                }
            )
            if model_used >= limits["agent_model_seconds"]:
                raise AttemptError(
                    "model_budget",
                    "Model response reached the budget deadline; its calls were not executed",
                )
            output = body.get("output")
            if (
                body.get("status") != "completed"
                or body.get("error")
                or (response.http_status and response.http_status >= 400)
                or not isinstance(output, list)
            ):
                cause = provider_cause(
                    body,
                    http_status=response.http_status,
                    request_id=response.request_id,
                )
                raise AttemptError(
                    "provider_incomplete",
                    cause["reason"] + "; no calls from this response executed",
                    cause=cause,
                )
            calls = []
            for item in output:
                if not isinstance(item, dict) or item.get("type") not in {
                    "message",
                    "reasoning",
                    "function_call",
                }:
                    raise AttemptError(
                        "unsupported_provider_action",
                        "Provider returned an unsupported action; no calls from this response executed",
                    )
                if item.get("status", "completed") != "completed":
                    raise AttemptError(
                        "provider_incomplete",
                        "Provider output item was incomplete; no calls from this response executed",
                    )
                if item["type"] == "function_call":
                    calls.append(item)
            rejected, invalid_ids = preflight(
                calls, all_call_ids, tools, config["tool_scope"], protocol
            )
            all_call_ids.update(
                c["call_id"] for c in calls if isinstance(c.get("call_id"), str)
            )
            history.extend(output)  # Retain ALL output, including opaque reasoning.
            if not calls:
                reminder = {"role": "user", "content": REMINDER}
                trace.append("reminder", reminder, request_id=active_request)
                history.append(reminder)
                continue
            # Reserve the complete group before any effects, including writes.
            trace.ensure(
                sum(len(_encoded(c)) + limits["max_result_bytes"] + 2048 for c in calls)
            )
            recoverable_reads = (
                bool(protocol.get("read_result_admission"))
                and rejected is None
                and all(c.get("name") in services.read_names() for c in calls)
            )
            if recoverable_reads:
                # Reserve every pending receipt before any read body is exposed.
                minimum = read_admission.candidate(
                    model,
                    history,
                    tools,
                    protocol,
                    model_used,
                    calls[0]["call_id"],
                    read_admission.ERROR,
                    calls[1:],
                )
                if not read_admission.budget(
                    adapter,
                    minimum,
                    limits["max_model_input_bytes"],
                    quota_limit=protocol["read_result_admission"]["quota_limit"],
                )["admissible"]:
                    raise AttemptError(
                        "read_admission_base",
                        "The complete pending read receipts cannot fit the next request; no read bodies were delivered",
                    )
            for index, call in enumerate(calls):
                phase = "tool"
                active_call = call
                name, cid = call.get("name"), call.get("call_id")
                # Reserve the largest service result BEFORE a possible write.
                trace.ensure(len(_encoded(call)) + limits["max_result_bytes"] + 2048)
                arguments = _arguments(call.get("arguments"))
                requested_arguments = arguments
                if rejected is None:
                    arguments = services.agent_arguments(name, arguments)
                executed_call = (
                    call
                    if arguments == requested_arguments
                    else {**call, "arguments": _encoded(arguments).decode()}
                )
                # Preserve the raw model_response/history. The tool_request is
                # the actual native call, with its requested form beside it.
                trace.append(
                    "tool_request",
                    executed_call,
                    request_id=active_request,
                    call_id=cid,
                    **(
                        {"requested_arguments": requested_arguments}
                        if arguments != requested_arguments
                        else {}
                    ),
                )
                finished = False
                if rejected is not None:
                    result = rejected[index]
                elif name == "finish_task":
                    finished = True
                    result = {"ok": True}
                else:
                    tool_start = time.monotonic()
                    try:
                        result = await worker.call_async(
                            name, arguments, limits["tool_timeout_seconds"]
                        )
                    except BaseException:
                        if isinstance(name, str) and name in WRITES:
                            uncertain.append(
                                {
                                    "call_id": cid,
                                    "tool": name,
                                    "reason": "Delivery may have committed; never retry automatically",
                                }
                            )
                        graceful = False
                        raise
                    finally:
                        tool_wait += time.monotonic() - tool_start
                withheld = None
                if recoverable_reads:
                    next_request = read_admission.candidate(
                        model,
                        history,
                        tools,
                        protocol,
                        model_used,
                        cid,
                        result,
                        calls[index + 1 :],
                    )
                    decision = read_admission.budget(
                        adapter,
                        next_request,
                        limits["max_model_input_bytes"],
                        quota_limit=protocol["read_result_admission"]["quota_limit"],
                    )
                    if not decision["admissible"]:
                        withheld = read_admission.save(
                            bundle, trace, executed_call, arguments, result, decision
                        )
                        result = dict(read_admission.ERROR)
                trace.append(
                    "tool_result",
                    result,
                    request_id=active_request,
                    call_id=cid,
                    **({"read_admission": withheld} if withheld else {}),
                )
                history.append(read_admission.output(cid, result))
                if isinstance(name, str) and name in WRITES and result.get("ok"):
                    message = result.get("message", result)
                    messages.append(
                        {
                            "call_id": cid,
                            "tool": name,
                            "channel_id": message["channel_id"],
                            "ts": message["ts"],
                            **(
                                {"destination": result["destination"]}
                                if "destination" in result
                                else {}
                            ),
                        }
                    )
                if finished:
                    outcome, stop_reason = "completed", "finish_task"
                    break
            if invalid_ids:
                raise AttemptError(
                    "invalid_provider_call",
                    "Invalid call IDs prevent provider continuation. No calls executed",
                )
            if outcome == "completed":
                break
    except (Exception, asyncio.CancelledError) as exc:
        cause = getattr(exc, "cause", None)
        if isinstance(exc, (TimeoutError, asyncio.CancelledError)):
            code = (
                "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "model_timeout"
            )
            reason = (
                "Attempt cancelled"
                if code == "cancelled"
                else "Model request exceeded its remaining response budget"
            )
        elif isinstance(exc, (AttemptError, WorkerFailure)):
            code, reason = exc.code, str(exc)
        else:
            cause = exception_cause(exc)
            code, reason = (
                "unexpected_failure",
                f"{phase} failed: {cause['type']} ({cause['code']}); transport details omitted",
            )
        if phase == "setup" and not setup_seconds:
            setup_seconds = time.monotonic() - setup_start
        stop_reason = code
        outcome = (
            "timeout"
            if code in {"model_timeout", "model_budget", "worker_timeout", "cancelled"}
            else "technical_error"
        )
        graceful = False
        error = {
            "phase": phase,
            "code": code,
            "reason": reason,
            "request_id": active_request,
            "call_id": active_call.get("call_id") if active_call else None,
            "tool": active_call.get("name") if active_call else None,
            **({"cause": cause} if cause else {}),
        }
        trace.append(
            "error",
            error,
            terminal=True,
            **(
                {"provider_timing": provider_timing}
                if managed and provider_timing
                else {}
            ),
        )
    finally:
        if worker is not None:
            worker.close(graceful=graceful)
    # No worker can write after this point. Publication never mutates this bundle.
    receipt_path = work / "worker-ipc/baseline-receipt.json"
    if receipt_path.is_file():
        shutil.copyfile(receipt_path, bundle / "baseline-receipt.json")
    _freeze_slack(work / "environment/slack", bundle / "final_state/slack")
    trace.append(
        "stop", {"outcome": outcome, "stop_reason": stop_reason}, terminal=True
    )
    trace.close()
    (bundle / "execution-protocol.json").write_bytes(_encoded(protocol))
    (bundle / "instructions.txt").write_bytes(instruction_bytes)
    (bundle / "tools.json").write_bytes(_encoded(tools))
    shutil.copyfile(model_path, bundle / "model.json")
    run = {
        "schema_version": 1,
        **identity,
        "unit_key": unit_key,
        "outcome": outcome,
        "stop_reason": stop_reason,
        "message_refs": messages,
        "uncertain_delivery": uncertain,
        "resolved_limits": limits,
        "hashes": {
            "instruction_sha256": instruction_sha256,
            "tool_definitions_sha256": digest(tools),
            "input_sha256": digest(initial),
        },
        "provider_responses": provider,
        "baseline_receipt_path": (
            "baseline-receipt.json" if receipt_path.is_file() else None
        ),
        "timings": {
            "budget_seconds_used": model_used,
            "tool_wait_seconds": tool_wait,
            **(
                {"quota_wait_seconds": quota_wait, "backoff_seconds": backoff}
                if managed
                else {}
            ),
            "setup_seconds": setup_seconds,
            "wall_seconds": time.monotonic() - started,
        },
        "trace_path": "trace.jsonl",
        "final_state_path": "final_state/slack",
        "error": error,
    }
    (bundle / "run.json").write_bytes(_encoded(run) + b"\n")
    if artifact_store is not None:
        publish_start = time.monotonic()
        try:
            artifact_store.publish_directory(bundle, unit_key, identity)
        except Exception as exc:
            raise PublicationError(bundle, unit_key, _safe_reason(exc)) from None
        logging.getLogger(__name__).info(
            f"publication completed unit={unit_key} seconds={time.monotonic() - publish_start:.6f}"
        )
    return {"skipped": False, "bundle": str(bundle), "run": run}
