"""Versioned request status and effect-free tool-group preflight."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
import math

from benchmark.prompts import load_text
from . import service_registry as services
from . import read_result_admission as read_admission

VERSION = "agent-execution-2"
MAX_READ_CALLS = 8
HARD_MAX_READ_CALLS = 16


def execution_protocol(limits, *, provider_admission=None, read_result_admission=True):
    protocol = {
        "version": VERSION,
        "accounting": "awaited_model_api_response_seconds",
        "configured_model_seconds": limits["agent_model_seconds"],
        "parallel_tool_calls": True,
        "max_read_calls_per_response": limits["max_read_calls_per_response"],
        "worker_execution": "sequential",
        "read_error_policy": "continue_on_result_stop_on_exception",
        "status_history": "request_only",
        "display_decimal_places": 3,
        "text": json.loads(load_text("agent/execution_protocol.json")),
    }
    if read_result_admission:
        protocol["read_result_admission"] = (
            read_admission.protocol()
            if read_result_admission is True
            else read_result_admission
        )
    if provider_admission is not None:
        protocol["provider_admission"] = provider_admission
    return protocol


def recorded_protocols(limits, model=None):
    """Supported execution records for the current agent policy."""
    return [
        execution_protocol(limits, read_result_admission=p)
        for p in read_admission.protocols(model)
    ]


def runtime_status(protocol, consumed):
    def rounded(value, mode):
        return str(
            Decimal.from_float(float(value)).quantize(Decimal("0.001"), rounding=mode)
        )

    total = protocol["configured_model_seconds"]
    content = protocol["text"]["status"].format(
        total=total,
        consumed=rounded(consumed, ROUND_CEILING),
        remaining=rounded(max(0, total - consumed), ROUND_FLOOR),
    )
    content += "\n" + protocol["text"]["grouping"].format(
        max_read_calls=protocol["max_read_calls_per_response"]
    )
    return {"role": "developer", "content": content}


def arguments(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate argument key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite argument")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
        if not isinstance(value, dict):
            return None
        # Reject overflow numbers and invalid Unicode before effects as well.
        if not valid_schema({"type": "object", "additionalProperties": {}}, value):
            return None
        json.dumps(value, ensure_ascii=False).encode("utf-8")
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return None


def valid_schema(schema, value):
    """Structural preflight of published native schemas, before worker validation.

    Native readers still own semantic and permission validation. No read is
    opened here. Unconstrained additional properties recurse for finite JSON.
    """
    if "anyOf" in schema and not any(valid_schema(s, value) for s in schema["anyOf"]):
        return False
    kind = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }.get(type(value))
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if kind is None or (
        types and kind not in types and not (kind == "integer" and "number" in types)
    ):
        return False
    if "enum" in schema and not any(
        type(value) is type(v) and value == v for v in schema["enum"]
    ):
        return False
    if kind in {"integer", "number"}:
        return math.isfinite(value) and schema.get(
            "minimum", -math.inf
        ) <= value <= schema.get("maximum", math.inf)
    if kind == "string":
        return (
            schema.get("minLength", 0)
            <= len(value)
            <= schema.get("maxLength", math.inf)
        )
    if kind == "array":
        return schema.get("minItems", 0) <= len(value) <= schema.get(
            "maxItems", math.inf
        ) and all(valid_schema(schema.get("items", {}), v) for v in value)
    if kind == "object":
        props, extra = schema.get("properties", {}), schema.get(
            "additionalProperties", True
        )
        return set(schema.get("required", [])) <= value.keys() and all(
            (
                valid_schema(props[k], v)
                if k in props
                else extra is not False
                and valid_schema(extra if isinstance(extra, dict) else {}, v)
            )
            for k, v in value.items()
        )
    return True


def preflight(calls, seen, tools, scope, protocol):
    """Return one deterministic rejection per call, or None for a valid group.

    Invalid IDs are terminal after recording every declined call. They cannot
    form a valid provider continuation. Ordinary rejections may be corrected.
    """
    definitions = {t["name"]: t["parameters"] for t in tools}
    messages = protocol["text"]

    def error(code, key):
        return {"ok": False, "error": code, "message": messages[key]}

    failures, ids, invalid_ids = [], set(), False
    for call in calls:
        cid, name = call.get("call_id"), call.get("name")
        args = arguments(call.get("arguments"))
        if not isinstance(cid, str) or not cid.strip() or cid in seen or cid in ids:
            failure = error("invalid_provider_call", "invalid_call_id")
            invalid_ids = True
        elif (
            not isinstance(name, str)
            or name not in definitions
            or (name != "finish_task" and name not in scope)
        ):
            failure = error("unknown_tool", "unknown_tool")
        elif args is None:
            failure = error("invalid_arguments", "invalid_arguments")
        elif not valid_schema(definitions[name], args):
            failure = error(
                "invalid_arguments",
                "finish_arguments" if name == "finish_task" else "schema_arguments",
            )
        else:
            failure = None
        if isinstance(cid, str):
            ids.add(cid)
        failures.append(failure)
    if invalid_ids:
        return [
            f or error("not_executed_invalid_batch", "invalid_batch") for f in failures
        ], True
    if len(calls) > 1 and any(c.get("name") == "finish_task" for c in calls):
        rejection = {
            "ok": False,
            "error": "not_executed_mixed_finish",
            "message": load_text("agent/mixed_finish_reminder.txt"),
        }
    elif len(calls) > 1 and any(
        isinstance(c.get("name"), str) and c["name"] in services.write_names()
        for c in calls
    ):
        rejection = error("not_executed_mixed_write", "mixed_write")
    elif len(calls) > protocol["max_read_calls_per_response"]:
        rejection = error("not_executed_batch_limit", "batch_cap")
    elif len(calls) > 1 and any(
        not isinstance(c.get("name"), str) or c["name"] not in services.read_names()
        for c in calls
    ):
        rejection = error("not_executed_invalid_batch", "invalid_batch")
    else:
        rejection = None
    if rejection or any(failures):
        return [
            f or rejection or error("not_executed_invalid_batch", "invalid_batch")
            for f in failures
        ], False
    return None, False


RUN_IDENTITY = (
    "run_id",
    "attempt_id",
    "task_ref",
    "config_ref",
    "model_ref",
    "baseline_ref",
)


def run_identity(run):
    keys = RUN_IDENTITY + (
        ("execution_protocol",) if "execution_protocol" in run else ()
    )
    return {key: run[key] for key in keys if key != "baseline_ref" or key in run}


def render_instruction(config, limits):
    text = config["agent_instructions"].replace(
        "{recent_activity_window_minutes}",
        str(limits["recent_activity_window_minutes"]),
    )
    return services.render_guides(text, config["tool_scope"])
