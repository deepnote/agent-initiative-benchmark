"""Judge input, result and validation contracts."""

from __future__ import annotations

from benchmark.prompts import load_text

import hashlib
import json
import math
import re
from typing import Literal, TypedDict, NotRequired

from .failure_causes import RuntimeInputError
from . import service_registry as services

SCHEMA_VERSION = 2
from .judge_rubrics import RUBRIC_VERSION, SCORE_MAX, score_max, uses_content_quality

PROMPT_VERSION = "benchmark-judge-v1"
ERROR_POLICY_VERSION = "judge-error-policy-v2"
CITATION_FEEDBACK_VERSION = "citation-feedback-2"
EVALUATION_VERSION = "judge-repair-v1"
CRITERIA = (
    "intervention_or_silence",
    "correctness_and_evidence",
    "manner_of_intervention",
    "audience_and_location",
)
READS = services.read_names()
DEFAULT_LIMITS = {
    "judge_model_seconds": 180,
    "judge_read_timeout_seconds": 60,
    "judge_setup_timeout_seconds": 180,
    "judge_max_requests": 64,
    "judge_initial_input_max_bytes": 1048576,
    "judge_max_model_input_bytes": 2097152,
    "judge_max_result_bytes": 1048576,
    "judge_max_trace_bytes": 134217728,
    "judge_max_source_file_bytes": 134217728,
    "judge_max_source_bytes": 268435456,
    "judge_max_source_count": 20000,
    "judge_max_work_bytes": 536870912,
    "judge_duckdb_memory_limit": "256MiB",
    "judge_duckdb_threads": 2,
    "judge_duckdb_max_temp_directory_size": "128MiB",
}
Score = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class BuildRef(TypedDict):
    build_id: str
    manifest_sha256: str


class ConfigRef(TypedDict):
    uri: str
    sha256: str


class TaskRef(TypedDict):
    task_id: str
    version: int
    manifest_sha256: str
    build_ref: BuildRef


class FileEntry(TypedDict):
    path: str
    bytes: int
    sha256: str


class BaselineRef(TypedDict):
    build_ref: BuildRef
    local_inventory_sha256: str
    provenance: Literal["local_fixture_only"]


class VerifiedBaselineRef(TypedDict):
    mode: Literal["verified_local"]
    build_ref: BuildRef
    source_receipt_sha256: str | None


class RunRef(TypedDict):
    run_id: str
    attempt_id: str
    unit_key: str
    schema_version: Literal[1]
    run_json_sha256: str
    bundle_sha256: str


class ChannelDestination(TypedDict):
    kind: Literal["channel"]
    channel_id: str


class ThreadDestination(TypedDict):
    kind: Literal["thread"]
    channel_id: str
    thread_ts: str


class DMDestination(TypedDict):
    kind: Literal["dm"]
    user_id: str


class DeliveredMessage(TypedDict):
    evidence_id: str
    call_id: str
    tool: Literal["slack_post_message", "slack_reply_to_thread", "slack_send_dm"]
    channel_id: str
    ts: str
    user_id: str
    text: str
    destination: ChannelDestination | ThreadDestination | DMDestination


class SourceEntry(TypedDict):
    id: str
    kind: Literal["task", "snapshot", "message", "trace", "artifact", "read_result"]
    locator: dict
    sha256: str


class InitialSlackMessage(TypedDict):
    channel_id: str
    user_id: str
    ts: str
    thread_ts: str | None
    text: str


class Policy(TypedDict):
    snapshot_time: str
    recent_activity_window_minutes: int
    agent_instructions: str
    allowed_tools: list[str]


class Attempt(TypedDict):
    outcome: str
    stop_reason: str
    model_budget_termination: bool


class InitialContext(TypedDict):
    snapshot_time: str
    messages: list[InitialSlackMessage]


class AnonymousTaskRef(TypedDict):
    task_id: str
    version: int


class AnonymousRunRef(TypedDict):
    answer_id: str


class OpaqueRef(TypedDict):
    id: str


class JudgeInput(TypedDict):
    schema_version: Literal[2, 3]
    task_ref: TaskRef | AnonymousTaskRef
    config_ref: ConfigRef | OpaqueRef
    run_ref: RunRef | AnonymousRunRef
    baseline_ref: BaselineRef | VerifiedBaselineRef | OpaqueRef
    task: dict  # Task content, with neutral provenance in anonymous inputs.
    policy: Policy
    attempt: Attempt
    initial_context: InitialContext
    delivered_messages: list[DeliveredMessage]
    sources: list[SourceEntry]


class Criterion(TypedDict):
    score: Score | None
    reason: str
    evidence: list[str]


class Criteria(TypedDict):
    intervention_or_silence: Criterion
    correctness_and_evidence: Criterion
    manner_of_intervention: Criterion
    audience_and_location: Criterion


class ReferenceDifference(TypedDict):
    material: bool | None
    reason: str
    evidence: list[str]
    task_review_reason: str | None


class ModelJudgment(TypedDict):
    status: Literal["scored", "task_error", "technical_error"]
    overall: Score | None
    criteria: Criteria
    reason: str
    evidence: list[str]
    reference_difference: ReferenceDifference


class JudgeLimitsV1(TypedDict):
    judge_model_seconds: float
    judge_read_timeout_seconds: float
    judge_setup_timeout_seconds: float
    judge_max_requests: int
    judge_initial_input_max_bytes: int
    judge_max_model_input_bytes: int
    judge_max_result_bytes: int
    judge_max_trace_bytes: int
    judge_max_source_file_bytes: int
    judge_max_source_bytes: int
    judge_max_source_count: int
    judge_max_work_bytes: int
    judge_duckdb_memory_limit: str
    judge_duckdb_threads: int
    judge_duckdb_max_temp_directory_size: str


class ProviderIdentityV1(TypedDict):
    request_id: str
    response_id: str
    model: str
    version: str


class ModelRef(TypedDict):
    sha256: str
    settings: dict  # Existing ModelV1, validated by load_model.
    route_id: str


class JudgePins(TypedDict):
    judge_model_budget_override: NotRequired[dict]
    read_result_admission: NotRequired[dict]
    anonymous_projection_version: NotRequired[str]
    model_ref: ModelRef
    provider_responses: list[ProviderIdentityV1]
    rubric_version: Literal["benchmark-rubric-v1"]
    prompt_version: str
    prompt_sha256: str
    tools_sha256: str
    input_sha256: str
    limits_sha256: str
    runtime_sha256: str


class Diagnostic(TypedDict):
    phase: str
    code: str
    reason: str
    cause: dict | None


class Timings(TypedDict):
    model_seconds: float
    read_seconds: float
    setup_seconds: float
    wall_seconds: float


class Judgment(TypedDict):
    status: Literal["scored", "task_error", "technical_error"]
    overall: Score | None
    criteria: Criteria
    reason: str
    reference_difference: ReferenceDifference
    schema_version: Literal[2, 3]
    judgment_id: str
    unit_key: str
    task_ref: TaskRef
    config_ref: ConfigRef
    run_ref: RunRef
    baseline_ref: BaselineRef | VerifiedBaselineRef
    judge: JudgePins
    resolved_limits: JudgeLimitsV1
    evidence: list[SourceEntry]
    error: Diagnostic | None
    timings: Timings
    trace_path: Literal["trace.jsonl"]


class JudgeError(RuntimeError):
    def __init__(self, code: str, reason: str, *, cause=None):
        self.code, self.cause = code, cause
        super().__init__(reason)


def parse_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("Nonfinite JSON value")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)

    def check(item):
        if isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Nonfinite JSON value")
        elif isinstance(item, dict):
            for k, v in item.items():
                check(k)
                check(v)
        elif isinstance(item, list):
            for v in item:
                check(v)

    check(value)
    return value


def fields(value, names, label):
    if not isinstance(value, dict) or value.keys() != set(names):
        raise JudgeError("invalid_judgment", f"{label}: unexpected or missing fields")


def text(value, label, *, limit=None):
    if (
        not isinstance(value, str)
        or not value.strip()
        or (limit and len(value) > limit)
    ):
        raise JudgeError(
            "invalid_judgment",
            f"{label}: expected nonempty text"
            + (f" up to {limit} characters" if limit else ""),
        )
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise JudgeError("invalid_judgment", f"{label}: invalid UTF-8") from None


def score(value, *, rubric_version=RUBRIC_VERSION):
    try:
        maximum = score_max(rubric_version)
    except ValueError as exc:
        raise JudgeError("invalid_judgment", str(exc)) from None
    if value is not None and (type(value) is not int or not 0 <= value <= maximum):
        raise JudgeError(
            "invalid_judgment", f"Scores must be integers from 0 to {maximum} or null"
        )


def memory_bytes(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[1-9][0-9]*(?:B|KB|MB|GB|KiB|MiB|GiB)", value
    ):
        raise RuntimeInputError(
            "invalid_judge_settings: expected a positive DuckDB byte size"
        )
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    number, unit = re.fullmatch(r"([0-9]+)([A-Za-z]+)", value).groups()
    return int(number) * units[unit]


JUDGE_BUDGET_OVERRIDE = {
    "version": "judge-model-budget-override-1",
    "judge_model_seconds": 600,
}


def budget_override(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value != JUDGE_BUDGET_OVERRIDE
        or set(value) != set(JUDGE_BUDGET_OVERRIDE)
        or type(value.get("judge_model_seconds")) is not int
    ):
        raise RuntimeInputError("Invalid judge-only model budget override")
    return dict(value)


def evaluation_binding(source_binding, override):
    policy = budget_override(override)
    if policy is None:
        return source_binding
    if not isinstance(source_binding, str) or not re.fullmatch(
        "[0-9a-f]{64}", source_binding
    ):
        raise RuntimeInputError("Judge budget override needs an exact source binding")
    raw = json.dumps(
        {"source_binding": source_binding, "judge_model_budget_override": policy},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def bound_budget_override(value, source_binding):
    policy = budget_override(value)
    return (
        {
            **policy,
            "source_binding": source_binding,
            "evaluation_binding": evaluation_binding(source_binding, policy),
        }
        if policy is not None
        else None
    )


def recorded_budget_override(value, source_binding):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(JUDGE_BUDGET_OVERRIDE) | {
        "source_binding",
        "evaluation_binding",
    }:
        raise RuntimeInputError("Unbound recorded judge budget override")
    policy = budget_override({key: value[key] for key in JUDGE_BUDGET_OVERRIDE})
    if value != bound_budget_override(policy, source_binding):
        raise RuntimeInputError("Recorded judge budget override source binding differs")
    return policy


def resolve_limits(config, *, judge_model_budget_override=None) -> JudgeLimitsV1:
    settings = config["judge_settings"]
    if settings.keys() - {"limits"} or not isinstance(settings.get("limits", {}), dict):
        raise RuntimeInputError(
            "invalid_judge_settings: expected {} or {limits: {...}}"
        )
    overrides = settings.get("limits", {})
    if overrides.keys() - DEFAULT_LIMITS.keys():
        raise RuntimeInputError("invalid_judge_settings: unknown limit keys")
    limits = {**DEFAULT_LIMITS, **overrides}
    for name, value in limits.items():
        if name in {
            "judge_duckdb_memory_limit",
            "judge_duckdb_max_temp_directory_size",
        }:
            memory_bytes(value)
        elif (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            or (not name.endswith("_seconds") and type(value) is not int)
        ):
            raise RuntimeInputError(
                f"invalid_judge_settings: {name} must be positive and finite with the specified type"
            )
    if limits["judge_max_result_bytes"] < 512:
        raise RuntimeInputError(
            "invalid_judge_settings: judge_max_result_bytes must be at least 512"
        )
    policy = budget_override(judge_model_budget_override)
    if policy is not None:
        if limits["judge_model_seconds"] != 180:
            raise RuntimeInputError(
                "Judge budget override requires the original 180-second policy"
            )
        limits["judge_model_seconds"] = policy["judge_model_seconds"]
    return limits


def validate_model_judgment(
    value,
    *,
    delivered,
    sources,
    accessible,
    assemble_evidence=False,
    rubric_version=RUBRIC_VERSION,
) -> ModelJudgment:
    fields(value, ModelJudgment.__annotations__, "judgment")
    if value["status"] not in ("scored", "task_error", "technical_error"):
        raise JudgeError("invalid_judgment", "Invalid status")
    score(value["overall"], rubric_version=rubric_version)
    text(value["reason"], "reason", limit=4000)
    fields(value["criteria"], CRITERIA, "criteria")

    def citations(ids):
        if not isinstance(ids, list) or len(ids) > 100:
            raise JudgeError(
                "invalid_judgment", "Evidence must be a list of at most 100 IDs"
            )
        for item in ids:
            text(item, "evidence ID")
        if len(set(ids)) != len(ids) or any(
            i not in accessible or i not in sources for i in ids
        ):
            raise JudgeError(
                "invalid_citation", "Duplicate, fabricated or unread evidence ID"
            )

    citations(value["evidence"])
    for name, criterion in value["criteria"].items():
        fields(criterion, Criterion.__annotations__, name)
        score(criterion["score"], rubric_version=rubric_version)
        text(criterion["reason"], name + ".reason", limit=4000)
        citations(criterion["evidence"])
    ref = value["reference_difference"]
    fields(ref, ReferenceDifference.__annotations__, "reference_difference")
    if ref["material"] is not None and type(ref["material"]) is not bool:
        raise JudgeError(
            "invalid_judgment", "reference_difference.material must be boolean or null"
        )
    text(ref["reason"], "reference_difference.reason", limit=4000)
    if ref["task_review_reason"] is not None:
        text(ref["task_review_reason"], "task_review_reason", limit=4000)
    citations(ref["evidence"])
    if assemble_evidence:
        # Keep the raw model object intact. Validate every original list first,
        # then derive only the redundant aggregate in a stable field order.
        union = list(
            dict.fromkeys(
                value["evidence"]
                + [
                    sid
                    for name in CRITERIA
                    for sid in value["criteria"][name]["evidence"]
                ]
                + ref["evidence"]
            )
        )
        citations(union)
        value = {**value, "evidence": union}
    cited = set(ref["evidence"]).union(
        *(c["evidence"] for c in value["criteria"].values())
    )
    if not cited <= set(value["evidence"]):
        raise JudgeError(
            "invalid_citation",
            "All criterion/reference citations must appear in evidence",
        )
    if value["status"] == "scored":
        if type(ref["material"]) is not bool:
            raise JudgeError(
                "invalid_judgment",
                "Scored judgments require boolean reference_difference.material",
            )
        if value["overall"] is None or value["criteria"][CRITERIA[0]]["score"] is None:
            raise JudgeError(
                "invalid_judgment", "Scored judgments require overall and criterion 1"
            )
        for name in CRITERIA[1:]:
            rating = value["criteria"][name]["score"]
            if (not delivered and rating is not None) or (
                delivered
                and (name in CRITERIA[2:] or uses_content_quality(rubric_version))
                and rating is None
            ):
                raise JudgeError(
                    "invalid_judgment", "Invalid silence/message applicability"
                )
        for criterion in [
            c for c in value["criteria"].values() if c["score"] is not None
        ] + [value]:
            ids = criterion["evidence"]
            if not ids or all(sources[i]["kind"] == "task" for i in ids):
                raise JudgeError(
                    "invalid_citation",
                    "Scored explanations require observed evidence, not private prose alone",
                )
        if ref["material"] and not ref["evidence"]:
            raise JudgeError(
                "invalid_citation", "Material reference differences require citations"
            )
        if uses_content_quality(rubric_version):
            all_suitable = all(
                c["score"] == 10
                for c in value["criteria"].values()
                if c["score"] is not None
            )
            if all_suitable != (value["overall"] == 10):
                raise JudgeError(
                    "inconsistent_scores",
                    "Overall 10 and all applicable categories at 10 must agree",
                )
    else:
        if value["overall"] is not None or any(
            c["score"] is not None for c in value["criteria"].values()
        ):
            raise JudgeError("invalid_judgment", "Error scores must all be null")
        if value["status"] == "task_error" and not value["evidence"]:
            raise JudgeError(
                "invalid_citation", "A task error must cite its concrete fixture defect"
            )
    return value


def score_consistency_feedback(error, *, rubric_version=RUBRIC_VERSION):
    from benchmark.prompts import load_text

    if not isinstance(error, JudgeError) or error.code != "inconsistent_scores":
        raise ValueError("Feedback requires a score-consistency error")
    if rubric_version != RUBRIC_VERSION:
        raise ValueError("No consistency feedback for this rubric")
    return {
        "ok": False,
        "error": error.code,
        "message": load_text("judge/score_consistency_reminder.txt"),
    }


def evidence_assembly(raw, result):
    """Explicit derived representation, alongside the unchanged raw trace call."""
    return {
        "protocol": "citation-union-1",
        "submitted": raw["evidence"],
        "derived": result["evidence"],
    }


def error_result(reason):
    return {
        "status": "technical_error",
        "overall": None,
        "criteria": {
            name: {
                "score": None,
                "reason": "Technical failure; not applicable",
                "evidence": [],
            }
            for name in CRITERIA
        },
        "reason": reason,
        "evidence": [],
        "reference_difference": {
            "material": None,
            "reason": "Technical failure; not assessed",
            "evidence": [],
            "task_review_reason": None,
        },
    }


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string", "minLength": 1}
_REASON = {**_STRING, "maxLength": 4000}
_IDS = {"type": "array", "items": _STRING, "uniqueItems": True, "maxItems": 100}
_CRITERION = obj(
    {
        "score": {"type": ["integer", "null"], "enum": [*range(SCORE_MAX + 1), None]},
        "reason": _REASON,
        "evidence": _IDS,
    }
)
MODEL_JUDGMENT_SCHEMA = obj(
    {
        "status": {
            "type": "string",
            "enum": ["scored", "task_error", "technical_error"],
        },
        "overall": {"type": ["integer", "null"], "enum": [*range(SCORE_MAX + 1), None]},
        "criteria": obj({name: _CRITERION for name in CRITERIA}),
        "reason": _REASON,
        "evidence": _IDS,
        "reference_difference": obj(
            {
                "material": {"type": ["boolean", "null"]},
                "reason": _REASON,
                "evidence": _IDS,
                "task_review_reason": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 4000,
                },
            }
        ),
    }
)


def _submit_parameters(schema):
    # Azure strict output rejects these length/uniqueness keywords. Keep them
    # in the full schema; validate_model_judgment still enforces them locally.
    if isinstance(schema, dict):
        return {
            key: _submit_parameters(value)
            for key, value in schema.items()
            if key not in {"minLength", "maxLength", "uniqueItems", "maxItems"}
        }
    if isinstance(schema, list):
        return [_submit_parameters(value) for value in schema]
    return schema


SUBMIT = {
    "type": "function",
    "name": "submit_judgment",
    "strict": True,
    "description": load_text("tools/judge/submit_judgment.txt"),
    "parameters": _submit_parameters(MODEL_JUDGMENT_SCHEMA),
}
READ_SOURCE = {
    "type": "function",
    "name": "judge_read_source",
    "strict": False,
    "description": load_text("tools/judge/read_source.txt"),
    "parameters": obj(
        {
            "source_id": {
                "anyOf": [
                    _STRING,
                    {
                        "type": "array",
                        "items": _STRING,
                        "minItems": 1,
                        "maxItems": 8,
                        "uniqueItems": True,
                    },
                ]
            },
            "cursor": {"type": ["string", "null"]},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 65536},
        }
    ),
}


def source_read_requests(args):
    """Preflight one exact page or independent first pages without any I/O."""
    if (
        not isinstance(args, dict)
        or args.keys() != {"source_id", "cursor", "max_chars"}
        or type(args["max_chars"]) is not int
        or not 1 <= args["max_chars"] <= 65536
        or args["cursor"] is not None
        and not isinstance(args["cursor"], str)
    ):
        return None
    ids = args["source_id"]
    if isinstance(ids, str):
        return [args]
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 8
        or any(not isinstance(sid, str) for sid in ids)
        or len(set(ids)) != len(ids)
        or args["cursor"] is not None
        or args["max_chars"] * len(ids) > 65536
    ):
        return None
    return [{**args, "source_id": sid} for sid in ids]


def judge_read_group_decline(calls, tools):
    """Decline a malformed or oversized read group before any read occurs."""
    from .agent_protocol import arguments, valid_schema

    if len(calls) <= 1:
        return None
    schemas = {tool["name"]: tool["parameters"] for tool in tools}
    invalid = len(calls) > 8
    for call in calls:
        args = arguments(call.get("arguments"))
        name = call.get("name")
        invalid = (
            invalid
            or args is None
            or name not in schemas
            or not valid_schema(schemas[name], args)
        )
        if name == "judge_read_source" and source_read_requests(args) is None:
            invalid = True
    if invalid:
        return {
            "ok": False,
            "error": "not_executed_read_group",
            "message": load_text("judge/read_group_reminder.txt"),
        }
    return None
