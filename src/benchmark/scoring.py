"""Combine compatible independent verdicts without any viewer dependency."""

from statistics import mean
from .runtime.judge_anonymity import VERSION


def combine_judgments(left, right):
    result = dict(status="incomplete", overall=None, absolute_difference=None)
    if any(j["status"] != "scored" or j["overall"] is None for j in (left, right)):
        result["status"] = (
            "task_fault"
            if any(j["status"] == "task_error" for j in (left, right))
            else (
                "needs_repair"
                if any(j["status"] == "technical_error" for j in (left, right))
                else "incomplete"
            )
        )
        return result
    if any(
        j["judge"].get("anonymous_projection_version") != VERSION for j in (left, right)
    ):
        raise ValueError("Both judges must use the shared anonymous input")
    fields = (
        "rubric_version",
        "prompt_version",
        "prompt_sha256",
        "tools_sha256",
        "input_sha256",
        "limits_sha256",
        "anonymous_projection_version",
        "runtime_sha256",
    )
    if not all(left["judge"].get(k) == right["judge"].get(k) for k in fields):
        raise ValueError("Judgments use different evaluation inputs or protocols")
    if not all(
        left.get(k) == right.get(k)
        for k in ("task_ref", "run_ref", "baseline_ref", "resolved_limits")
    ):
        raise ValueError("Judgments do not refer to the same answer and source")
    if (
        left["judgment_id"] == right["judgment_id"]
        or left["judge"]["model_ref"] == right["judge"]["model_ref"]
    ):
        raise ValueError("Two independent judge identities are required")
    values = [left["overall"], right["overall"]]
    return dict(
        status="complete",
        overall=mean(values),
        absolute_difference=abs(values[0] - values[1]),
    )
