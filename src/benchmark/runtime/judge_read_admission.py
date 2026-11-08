"""Versioned judge delivery checks over the existing pure read admission proof."""

from . import read_result_admission as shared

VERSION = "judge-read-result-admission-1"
ERROR = shared.ERROR


def protocol(adapter, model):
    transport = None
    return {
        "version": VERSION,
        "quota_limit": shared.profile(adapter, model)["quota_limit"],
        "provider_admission": transport,
    }


def supported(value, model):
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "quota_limit", "provider_admission"}
        or value["version"] != VERSION
    ):
        return False
    if value["provider_admission"] is not None:
        return False
    return shared.supported(shared.protocol(value["quota_limit"]), model)


def limits(value):
    return {
        "max_model_input_bytes": value["judge_max_model_input_bytes"],
        "max_result_bytes": value["judge_max_result_bytes"],
        "max_trace_bytes": value["judge_max_trace_bytes"],
    }


def request(model, history, tools, parallel):
    return {
        "model": model["model_id"],
        "input": history,
        "tools": tools,
        "store": False,
        "parallel_tool_calls": parallel,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": model["reasoning_effort"]},
        "max_output_tokens": model["max_output_tokens"],
    }


def candidate(model, history, tools, parallel, cid, result, pending):
    return request(
        model,
        history
        + [shared.output(cid, result)]
        + [shared.output(c["call_id"], ERROR) for c in pending],
        tools,
        parallel,
    )


def replay(root, model, value, bounds, route):
    if not supported(value, model):
        raise ValueError("Unsupported judge read-result admission policy")
    return shared.ReplayBudget(
        root,
        model,
        shared.protocol(value["quota_limit"]),
        limits(bounds),
        provider_admission=value["provider_admission"],
        route_id=route,
    )
