"""Translate complete benchmark conversations to provider JSON protocols."""

from __future__ import annotations

import copy
import json

from .failure_causes import RuntimeInputError


def encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def failed(code):
    return {
        "status": "failed",
        "output": [],
        "error": {"code": code, "message": code.replace("_", " ")},
    }


class Codec:
    def __init__(self, model):
        self.model = model
        self._blocks = {}

    def remember(self, block, identity):
        if identity in self._blocks and self._blocks[identity] != block:
            raise RuntimeInputError(
                "Provider reused a continuation ID with different content"
            )
        self._blocks[identity] = copy.deepcopy(block)
        return {"type": "reasoning", "id": identity, "summary": []}

    def restore(self, marker):
        if marker != {"type": "reasoning", "id": marker.get("id"), "summary": []}:
            raise RuntimeInputError("Invalid provider continuation marker")
        try:
            return copy.deepcopy(self._blocks[marker["id"]])
        except KeyError:
            raise RuntimeInputError(
                "Missing provider continuation. Use one adapter per attempt"
            ) from None

    def check(self, request):
        fields = {
            "model",
            "input",
            "tools",
            "store",
            "parallel_tool_calls",
            "include",
            "reasoning",
            "max_output_tokens",
        }
        if (
            not isinstance(request, dict)
            or set(request) != fields
            or request["model"] != self.model["model_id"]
            or request["reasoning"] != {"effort": self.model["reasoning_effort"]}
            or request["max_output_tokens"] != self.model["max_output_tokens"]
            or request["store"] is not False
            or type(request["parallel_tool_calls"]) is not bool
            or request["include"] != ["reasoning.encrypted_content"]
            or not isinstance(request["input"], list)
            or not isinstance(request["tools"], list)
        ):
            raise RuntimeInputError(
                "Request differs from the configured model or benchmark protocol"
            )

    def parameters(self, native):
        for name, value in self.model["parameters"].items():
            if isinstance(value, dict) and isinstance(native.get(name), dict):
                if any(
                    k in native[name] and native[name][k] != v for k, v in value.items()
                ):
                    raise RuntimeInputError(
                        f"Parameter {name} conflicts with the benchmark request"
                    )
                native[name].update(copy.deepcopy(value))
            elif name in native and native[name] != value:
                raise RuntimeInputError(
                    f"Parameter {name} conflicts with the benchmark request"
                )
            else:
                native[name] = copy.deepcopy(value)
        return native

    def serialize_request(self, request):
        self.check(request)
        return self.parameters(self.request(request))


def assistant_text(item):
    blocks = item.get("content")
    if not isinstance(blocks, list) or any(
        not isinstance(b, dict)
        or b.get("type") != "output_text"
        or not isinstance(b.get("text"), str)
        for b in blocks
    ):
        raise RuntimeInputError("Unsupported assistant content")
    return [b["text"] for b in blocks]


class ResponsesCodec(Codec):
    def request(self, request):
        native = copy.deepcopy(request)
        native["input"] = [
            self.restore(item) if item.get("type") == "reasoning" else item
            for item in native["input"]
        ]
        if self.model["reasoning_effort"] is None:
            native.pop("reasoning")
        return native

    def normalize_response(self, raw):
        if (
            not isinstance(raw.get("id"), str)
            or not raw["id"]
            or not isinstance(raw.get("output"), list)
            or raw.get("status")
            not in {"completed", "incomplete", "failed", "cancelled"}
        ):
            return failed("invalid_response")
        body = copy.deepcopy(raw)
        output = []
        for i, item in enumerate(body["output"]):
            if not isinstance(item, dict) or item.get("type") not in {
                "message",
                "reasoning",
                "function_call",
            }:
                return failed("unsupported_provider_action")
            if (
                item.get("status", "completed") != "completed"
                and raw["status"] == "completed"
            ):
                return failed("incomplete_output_item")
            if item["type"] == "message":
                content = item.get("content")
                if (
                    item.get("role") != "assistant"
                    or not isinstance(content, list)
                    or any(
                        not isinstance(b, dict)
                        or b.get("type") != "output_text"
                        or not isinstance(b.get("text"), str)
                        for b in content
                    )
                ):
                    return failed("unsupported_provider_content")
            if item["type"] == "reasoning":
                item = self.remember(item, raw["id"] + ":" + str(i))
            output.append(item)
        body["output"] = output
        return body


class ChatCodec(Codec):
    def request(self, request):
        messages = []
        current = None

        def assistant():
            nonlocal current
            if current is None:
                current = {"role": "assistant", "content": None}
            return current

        def flush():
            nonlocal current
            if current is not None:
                messages.append(current)
                current = None

        for item in request["input"]:
            kind, role = item.get("type"), item.get("role")
            if kind == "reasoning":
                assistant().update(self.restore(item))
            elif kind == "message" and role == "assistant":
                text = "".join(assistant_text(item))
                message = assistant()
                message["content"] = (message["content"] or "") + text
            elif kind == "function_call":
                assistant().setdefault("tool_calls", []).append(
                    dict(
                        id=item["call_id"],
                        type="function",
                        function=dict(name=item["name"], arguments=item["arguments"]),
                    )
                )
            elif kind == "function_call_output" and isinstance(item.get("output"), str):
                flush()
                messages.append(
                    dict(
                        role="tool",
                        tool_call_id=item["call_id"],
                        content=item["output"],
                    )
                )
            elif (
                kind is None
                and role in {"user", "developer"}
                and isinstance(item.get("content"), str)
            ):
                flush()
                messages.append(
                    dict(
                        role=(
                            self.model["instruction_role"]
                            if role == "developer"
                            else role
                        ),
                        content=item["content"],
                    )
                )
            else:
                raise RuntimeInputError(
                    "Unsupported conversation item for Chat Completions"
                )
        flush()
        tools = []
        for tool in request["tools"]:
            definition = {
                k: copy.deepcopy(v)
                for k, v in tool.items()
                if k != "type" and (k != "strict" or v)
            }
            tools.append(dict(type="function", function=definition))
        native = dict(
            model=request["model"],
            messages=messages,
            tools=tools,
            parallel_tool_calls=request["parallel_tool_calls"],
            stream=False,
            n=1,
        )
        native[self.model["token_limit_field"]] = request["max_output_tokens"]
        if self.model["reasoning_effort"] is not None:
            native["reasoning_effort"] = self.model["reasoning_effort"]
        return native

    def normalize_response(self, raw):
        choices = raw.get("choices")
        if (
            not isinstance(raw.get("id"), str)
            or not raw["id"]
            or not isinstance(choices, list)
            or len(choices) != 1
        ):
            return failed("invalid_chat_response")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return failed("invalid_chat_message")
        content, calls = message.get("content"), message.get("tool_calls", [])
        if (
            content is not None
            and not isinstance(content, str)
            or not isinstance(calls, list)
        ):
            return failed("unsupported_chat_content")
        known = {
            "role",
            "content",
            "tool_calls",
            "function_call",
            "audio",
            "refusal",
            "annotations",
            "reasoning_content",
            "reasoning_details",
            "reasoning",
        }
        if any(
            v is not None and k not in known for k, v in message.items()
        ) or message.get("annotations"):
            return failed("unsupported_chat_content")
        if (
            message.get("function_call") is not None
            or message.get("audio") is not None
            or message.get("refusal")
        ):
            return failed("unsupported_chat_action")
        output = []
        state = {
            k: copy.deepcopy(message[k])
            for k in ("reasoning_content", "reasoning_details", "reasoning")
            if message.get(k) is not None
        }
        if state:
            output.append(self.remember(state, raw["id"] + ":reasoning"))
        if content is not None:
            output.append(
                dict(
                    type="message",
                    role="assistant",
                    content=[dict(type="output_text", text=content)],
                )
            )
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if (
                not isinstance(function, dict)
                or call.get("type") != "function"
                or not all(
                    isinstance(v, str) and v
                    for v in [call.get("id"), function.get("name")]
                )
                or not isinstance(function.get("arguments"), str)
            ):
                return failed("invalid_chat_tool_call")
            output.append(
                dict(
                    type="function_call",
                    call_id=call["id"],
                    name=function["name"],
                    arguments=function["arguments"],
                    status="completed",
                )
            )
        stop = choice.get("finish_reason")
        if stop in {"stop", "tool_calls"} and bool(calls) == (stop == "tool_calls"):
            body = dict(status="completed", output=output)
        elif stop == "length":
            body = dict(
                status="incomplete",
                output=output,
                incomplete_details=dict(reason="max_output_tokens"),
            )
        else:
            return failed("unsupported_chat_stop")
        body.update(id=raw["id"], model=raw.get("model", self.model["model_id"]))
        usage = raw.get("usage")
        if isinstance(usage, dict):
            body["usage"] = dict(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        return body


def nullable_choices(value):
    if isinstance(value, list):
        return [nullable_choices(v) for v in value]
    if not isinstance(value, dict):
        return value
    value = {k: nullable_choices(v) for k, v in value.items()}
    types, choices = value.get("type"), value.get("enum")
    if (
        isinstance(types, list)
        and len(types) == 2
        and set(types) == {"integer", "null"}
        and isinstance(choices, list)
        and choices
        and all(v is None or type(v) is int for v in choices)
    ):
        integers = [v for v in choices if v is not None]
        branches = ([dict(type="integer", enum=integers)] if integers else []) + (
            [dict(type="null")] if None in choices else []
        )
        value = {k: v for k, v in value.items() if k not in {"type", "enum"}}
        value["anyOf"] = branches
    return value


class AnthropicCodec(Codec):
    def request(self, request):
        system, messages = [], []

        def append(role, blocks):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append(dict(role=role, content=blocks))

        for item in request["input"]:
            kind, role = item.get("type"), item.get("role")
            if (
                kind is None
                and role == "developer"
                and isinstance(item.get("content"), str)
            ):
                system.append(dict(type="text", text=item["content"]))
            elif (
                kind is None and role == "user" and isinstance(item.get("content"), str)
            ):
                append("user", [dict(type="text", text=item["content"])])
            elif kind == "reasoning":
                append("assistant", [self.restore(item)])
            elif kind == "function_call":
                from .agent_protocol import arguments

                append(
                    "assistant",
                    [
                        dict(
                            type="tool_use",
                            id=item["call_id"],
                            name=item["name"],
                            input=arguments(item["arguments"]),
                        )
                    ],
                )
            elif kind == "function_call_output" and isinstance(item.get("output"), str):
                append(
                    "user",
                    [
                        dict(
                            type="tool_result",
                            tool_use_id=item["call_id"],
                            content=item["output"],
                        )
                    ],
                )
            elif kind == "message" and role == "assistant":
                append(
                    "assistant",
                    [dict(type="text", text=t) for t in assistant_text(item)],
                )
            else:
                raise RuntimeInputError(
                    "Unsupported conversation item for Anthropic Messages"
                )
        tools = [
            dict(
                name=t["name"],
                description=t["description"],
                input_schema=nullable_choices(t["parameters"]),
            )
            for t in request["tools"]
        ]
        native = dict(
            model=request["model"],
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=request["max_output_tokens"],
            stream=False,
        )
        if not request["parallel_tool_calls"]:
            native["tool_choice"] = dict(type="auto", disable_parallel_tool_use=True)
        if self.model["reasoning_effort"] is not None:
            native["output_config"] = dict(effort=self.model["reasoning_effort"])
        return native

    def normalize_response(self, raw):
        if (
            raw.get("type") != "message"
            or raw.get("role") != "assistant"
            or not isinstance(raw.get("id"), str)
            or not raw["id"]
            or not isinstance(raw.get("content"), list)
        ):
            return failed("invalid_anthropic_response")
        output = []
        for i, block in enumerate(raw["content"]):
            if not isinstance(block, dict):
                return failed("invalid_anthropic_block")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                output.append(
                    dict(
                        type="message",
                        role="assistant",
                        content=[dict(type="output_text", text=block["text"])],
                    )
                )
            elif (
                kind == "tool_use"
                and isinstance(block.get("input"), dict)
                and all(
                    isinstance(block.get(k), str) and block[k] for k in ("id", "name")
                )
            ):
                output.append(
                    dict(
                        type="function_call",
                        call_id=block["id"],
                        name=block["name"],
                        arguments=encoded(block["input"]),
                        status="completed",
                    )
                )
            elif (
                kind == "thinking"
                and isinstance(block.get("thinking"), str)
                and isinstance(block.get("signature"), str)
                and block["signature"]
                or kind == "redacted_thinking"
                and isinstance(block.get("data"), str)
            ):
                output.append(self.remember(block, raw["id"] + ":" + str(i)))
            else:
                return failed("unsupported_anthropic_block")
        calls = any(v["type"] == "function_call" for v in output)
        stop = raw.get("stop_reason")
        if stop in {"end_turn", "stop_sequence", "tool_use"} and calls == (
            stop == "tool_use"
        ):
            body = dict(status="completed", output=output)
        elif stop in {"max_tokens", "model_context_window_exceeded", "pause_turn"}:
            body = dict(
                status="incomplete", output=output, incomplete_details=dict(reason=stop)
            )
        else:
            return failed("unsupported_anthropic_stop")
        body.update(id=raw["id"], model=raw.get("model", self.model["model_id"]))
        if isinstance(raw.get("usage"), dict):
            body["usage"] = copy.deepcopy(raw["usage"])
        return body


CODECS = {
    "responses": ResponsesCodec,
    "chat_completions": ChatCodec,
    "anthropic": AnthropicCodec,
}
