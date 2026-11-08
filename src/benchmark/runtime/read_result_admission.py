"""Pure next-request checks and private evidence for withheld read bodies."""

from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

from .slack_environment import _encoded

VERSION = "read-result-admission-3"
ERROR = {
    "ok": False,
    "error": "read_result_admission",
    "message": "The read completed, but its full result was not delivered because the next request would exceed its input or admission limit. Request smaller pages, fewer fields, or tighter filters.",
}


def protocol(quota_limit=None, *, version=VERSION):
    if quota_limit is not None or version != VERSION:
        raise ValueError("Only full-request byte admission is supported")
    return {"version": VERSION, "quota_limit": None}


def protocols(model=None):
    return [protocol()]


def supported(value, model=None):
    return isinstance(value, dict) and value == protocol()


def profile(adapter, model):
    return protocol()


def planned_profile(adapter, model):
    return profile(adapter, model)


def planned_execution(limits, adapter, model):
    from .agent_protocol import execution_protocol

    return execution_protocol(limits, read_result_admission=profile(adapter, model))


def digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def request(model, history, tools, protocol, consumed):
    from .agent_protocol import runtime_status

    return {
        "model": model["model_id"],
        "input": history + [runtime_status(protocol, consumed)],
        "tools": tools,
        "store": False,
        "parallel_tool_calls": protocol["parallel_tool_calls"],
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": model["reasoning_effort"]},
        "max_output_tokens": model["max_output_tokens"],
    }


def output(call_id, result):
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": _encoded(result).decode(),
    }


def candidate(model, history, tools, protocol, consumed, call_id, result, pending):
    tail = [output(call_id, result)] + [output(c["call_id"], ERROR) for c in pending]
    return request(model, history + tail, tools, protocol, consumed)


def budget(adapter, value, max_bytes, *, quota_limit=None):
    """Measure complete logical and native requests without truncation."""
    if quota_limit is not None:
        raise ValueError("Provider account quotas are not benchmark input limits")
    native = (
        adapter.serialize_request(value)
        if hasattr(adapter, "serialize_request")
        else value
    )
    logical, encoded = len(_encoded(value)), len(_encoded(native))
    return {
        "candidate_sha256": digest(value),
        "logical_bytes": logical,
        "native_bytes": encoded,
        "max_model_input_bytes": max_bytes,
        "quota_tokens": None,
        "quota_limit": None,
        "admissible": logical <= max_bytes and encoded <= max_bytes,
    }


def save(bundle, trace, call, arguments, result, decision):
    """Charge full evidence to the existing trace-byte bound before writing it."""
    payload = {
        "schema_version": 1,
        "protocol": VERSION,
        "raw_model_visible": False,
        "call_id": call["call_id"],
        "tool": call["name"],
        "arguments": arguments,
        "result": result,
        "budget": decision,
    }
    raw = _encoded(payload)
    trace.ensure(len(raw) + 4096)
    folder = Path(bundle) / "read-results"
    folder.mkdir(exist_ok=True)
    name = "read-results/" + str(trace.seq + 1).zfill(8) + ".json"
    with (Path(bundle) / name).open("xb") as stream:
        stream.write(raw)
    (Path(bundle) / name).chmod(0o600)
    trace.evidence_size += len(raw)
    return {
        "protocol": VERSION,
        "raw_model_visible": False,
        "path": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "arguments_sha256": digest(arguments),
        "result_bytes": len(_encoded(result)),
        "result_sha256": digest(result),
    }


def validate(root, event, call, expected_candidate, limits, replay):
    """Replay the exact delivered error and bind its complete private native result."""
    proof = event.get("read_admission")
    keys = {
        "protocol",
        "raw_model_visible",
        "path",
        "bytes",
        "sha256",
        "arguments_sha256",
        "result_bytes",
        "result_sha256",
    }
    if (
        not isinstance(proof, dict)
        or set(proof) != keys
        or proof["protocol"] != VERSION
        or proof["protocol"] != replay.profile["version"]
        or proof["raw_model_visible"] is not False
    ):
        raise ValueError("Invalid withheld-read proof")
    path = proof["path"]
    expected = "read-results/" + str(event["seq"]).zfill(8) + ".json"
    if path != expected or not re.fullmatch(r"read-results/[0-9]{8}\.json", path):
        raise ValueError("Invalid withheld-read path")
    target = Path(root) / path
    if (
        target.is_symlink()
        or target.stat().st_nlink != 1
        or target.stat().st_size != proof["bytes"]
        or proof["bytes"] > limits["max_trace_bytes"]
    ):
        raise ValueError("Invalid withheld-read file bounds")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof["sha256"]:
        raise ValueError("Withheld-read bytes changed")
    saved = json.loads(raw)
    if set(saved) != {
        "schema_version",
        "protocol",
        "raw_model_visible",
        "call_id",
        "tool",
        "arguments",
        "result",
        "budget",
    }:
        raise ValueError("Invalid withheld-read body")
    arguments = json.loads(call["arguments"])
    if (
        saved["schema_version"] != 1
        or saved["protocol"] != proof["protocol"]
        or saved["raw_model_visible"] is not False
        or saved["call_id"] != call["call_id"]
        or saved["tool"] != call["name"]
        or saved["arguments"] != arguments
        or digest(arguments) != proof["arguments_sha256"]
        or digest(saved["result"]) != proof["result_sha256"]
        or len(_encoded(saved["result"])) != proof["result_bytes"]
        or proof["result_bytes"] > limits["max_result_bytes"]
        or event["payload"] != ERROR
    ):
        raise ValueError("Withheld-read call/result binding differs")
    value = expected_candidate(saved["result"])
    d = budget(
        replay.owner,
        value,
        limits["max_model_input_bytes"],
        quota_limit=replay.profile["quota_limit"],
    )
    if saved["budget"] != d or d["admissible"] is not False:
        raise ValueError(
            "Withheld-read rejection does not match the independently replayed budget"
        )
    return path


class ReplayBudget:
    """Reconstruct native continuations and full requests without network access."""

    def __init__(
        self, root, model, profile, limits, *, provider_admission=None, route_id=None
    ):
        import types
        from .provider_adapters import create_adapter
        from .provider_codecs import CODECS

        if not supported(profile, model) or provider_admission is not None:
            raise ValueError("Unsupported recorded read-admission policy")
        self.profile = profile
        self.native = (
            create_adapter(model) if model["provider"] in {*CODECS, "custom"} else None
        )
        self.owner = self.native or types.SimpleNamespace()
        self.responses, self.requests = {}, {}
        if self.native is not None:
            if self.native.route_id != route_id:
                raise ValueError("Provider settings or adapter implementation changed")
            path = Path(root) / "provider-native.jsonl"
            if path.is_symlink() or path.stat().st_size > limits["max_trace_bytes"]:
                raise ValueError("Invalid native trace bounds")
            with path.open("rb") as stream:
                for line in stream:
                    event = json.loads(line)
                    kind, identity = event.get("kind"), event.get("request_id")
                    target = (
                        self.requests
                        if kind == "request"
                        else self.responses if kind == "response" else None
                    )
                    if (
                        target is None
                        or not isinstance(identity, str)
                        or identity in target
                        or event.get("route_id") != route_id
                    ):
                        raise ValueError("Invalid native trace lifecycle")
                    target[identity] = event
        self.number = 0

    def request(self, value):
        self.number += 1
        if self.native is not None:
            event = self.requests.get(f"request-{self.number}")
            if event is None or self.native.serialize_request(value) != event.get(
                "payload"
            ):
                raise ValueError(
                    "Native request differs from the recorded full conversation"
                )

    def response(self, request_id, value):
        if self.native is not None:
            from .provider_adapters import normalize_record

            event = self.responses.get(request_id)
            if (
                event is None
                or normalize_record(
                    self.native, event.get("response"), event.get("http_status")
                )
                != value
            ):
                raise ValueError(
                    "Native response differs from the recorded normalized response"
                )
