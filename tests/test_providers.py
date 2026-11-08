import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from benchmark.runtime.agent_runner import run_attempt
from benchmark.runtime.artifact_store import ArtifactStore
from benchmark.runtime.failure_causes import RuntimeInputError
from benchmark.runtime.judge_contracts import CRITERIA
from benchmark.runtime.judge_runner import prepare_anonymous_answer, run_judgment
from benchmark.runtime.model_config import validate_model
from benchmark.runtime.provider_adapters import HTTPAdapter, create_adapter
from benchmark.runtime.provider_codecs import CODECS
from benchmark.runtime.read_result_admission import ReplayBudget, budget, protocol
from benchmark.runtime.responses_adapter import model_response
from benchmark.scoring import combine_judgments
from helpers import call, fixture

PROVIDERS = tuple(CODECS)


def profile(provider="responses", **changes):
    return validate_model(
        dict(
            provider=provider,
            model_id="any/model:version",
            endpoint="https://model.example/v1",
            max_output_tokens=1024,
            **changes,
        )
    )


def request(model, history=None, parallel=True):
    return dict(
        model=model["model_id"],
        input=history
        or [
            dict(role="developer", content="Instructions"),
            dict(role="user", content="Inspect activity"),
        ],
        tools=[
            dict(
                type="function",
                name="inspect",
                description="Read activity",
                strict=False,
                parameters=dict(
                    type="object",
                    properties={
                        "limit": dict(type=["integer", "null"], enum=[1, 2, None])
                    },
                    required=["limit"],
                    additionalProperties=False,
                ),
            )
        ],
        store=False,
        parallel_tool_calls=parallel,
        include=["reasoning.encrypted_content"],
        reasoning=dict(effort=model["reasoning_effort"]),
        max_output_tokens=model["max_output_tokens"],
    )


def native_response(
    provider, items, identity="response-1", *, truncated=False, thinking=False
):
    if provider == "responses":
        output = copy.deepcopy(items)
        if thinking:
            output.insert(
                0,
                dict(
                    type="reasoning",
                    id="opaque-" + identity,
                    encrypted_content="opaque-signature",
                    summary=[],
                ),
            )
        return dict(
            id=identity,
            model="any/model:version",
            status="incomplete" if truncated else "completed",
            output=output,
            usage=dict(input_tokens=20, output_tokens=10),
            **(
                dict(incomplete_details=dict(reason="max_output_tokens"))
                if truncated
                else {}
            ),
        )
    calls = [i for i in items if i["type"] == "function_call"]
    text = "".join(
        b["text"] for i in items if i["type"] == "message" for b in i["content"]
    )
    if provider == "chat_completions":
        message = dict(role="assistant", content=text or None)
        if calls:
            message["tool_calls"] = [
                dict(
                    id=c["call_id"],
                    type="function",
                    function=dict(name=c["name"], arguments=c["arguments"]),
                )
                for c in calls
            ]
        if thinking:
            message["reasoning_content"] = "opaque thought"
            message["reasoning_details"] = [
                dict(type="reasoning.encrypted", data="opaque-signature")
            ]
        return dict(
            id=identity,
            model="any/model:version",
            choices=[
                dict(
                    index=0,
                    message=message,
                    finish_reason=(
                        "length" if truncated else "tool_calls" if calls else "stop"
                    ),
                )
            ],
            usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )
    content = []
    if thinking:
        content += [
            dict(
                type="thinking", thinking="opaque thought", signature="opaque-signature"
            ),
            dict(type="redacted_thinking", data="opaque-redacted"),
        ]
    if text:
        content.append(dict(type="text", text=text))
    content.extend(
        dict(
            type="tool_use",
            id=c["call_id"],
            name=c["name"],
            input=json.loads(c["arguments"]),
        )
        for c in calls
    )
    return dict(
        id=identity,
        type="message",
        role="assistant",
        model="any/model:version",
        content=content,
        stop_reason="max_tokens" if truncated else "tool_use" if calls else "end_turn",
        usage=dict(input_tokens=20, output_tokens=10),
    )


class CodecTests(unittest.TestCase):
    def test_complete_multi_tool_conversation_and_opaque_state(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                settings = profile(provider)
                codec = CODECS[provider](settings)
                first = request(settings)
                raw = native_response(
                    provider,
                    [
                        call("inspect", {"limit": 1}, "a"),
                        call("inspect", {"limit": 2}, "b"),
                    ],
                    thinking=True,
                )
                body = codec.normalize_response(raw)
                self.assertEqual(body["status"], "completed")
                outputs = [
                    dict(
                        type="function_call_output",
                        call_id=c,
                        output=json.dumps({"value": c}),
                    )
                    for c in ["a", "b"]
                ]
                native = codec.serialize_request(
                    request(
                        settings,
                        first["input"]
                        + body["output"]
                        + outputs
                        + [dict(role="developer", content="Remaining time")],
                    )
                )
                saved = json.dumps(native)
                self.assertIn("opaque-signature", saved)
                self.assertIn("Remaining time", saved)
                if provider == "responses":
                    self.assertEqual(native["input"][2], raw["output"][0])
                    self.assertEqual(native["input"][-3:-1], outputs)
                    self.assertNotIn("reasoning", native)
                elif provider == "chat_completions":
                    assistant = native["messages"][2]
                    self.assertEqual(
                        assistant["reasoning_details"],
                        raw["choices"][0]["message"]["reasoning_details"],
                    )
                    self.assertEqual(
                        [c["id"] for c in assistant["tool_calls"]], ["a", "b"]
                    )
                    self.assertEqual(
                        [
                            m["tool_call_id"]
                            for m in native["messages"]
                            if m["role"] == "tool"
                        ],
                        ["a", "b"],
                    )
                else:
                    self.assertEqual(
                        native["messages"][1]["content"][:2], raw["content"][:2]
                    )
                    self.assertEqual(
                        [b["tool_use_id"] for b in native["messages"][2]["content"]],
                        ["a", "b"],
                    )
                    self.assertEqual(native["system"][-1]["text"], "Remaining time")
                    self.assertEqual(
                        native["tools"][0]["input_schema"]["properties"]["limit"][
                            "anyOf"
                        ],
                        [dict(type="integer", enum=[1, 2]), dict(type="null")],
                    )

    def test_explicit_options_and_protocol_translations(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                params = (
                    {"reasoning": {"summary": "auto"}}
                    if provider == "responses"
                    else (
                        {"temperature": 0}
                        if provider == "chat_completions"
                        else {
                            "thinking": {"type": "adaptive"},
                            "output_config": {"format": {"type": "text"}},
                        }
                    )
                )
                settings = profile(
                    provider,
                    reasoning_effort="high",
                    parameters=params,
                    instruction_role="developer",
                    token_limit_field="max_tokens",
                )
                codec = CODECS[provider](settings)
                native = codec.serialize_request(request(settings, parallel=False))
                self.assertEqual(native["model"], settings["model_id"])
                if provider == "responses":
                    self.assertEqual(
                        native["reasoning"], dict(effort="high", summary="auto")
                    )
                    self.assertIs(native["store"], False)
                elif provider == "chat_completions":
                    self.assertEqual(native["messages"][0]["role"], "developer")
                    self.assertEqual(native["max_tokens"], 1024)
                    self.assertNotIn("max_completion_tokens", native)
                    self.assertEqual(native["temperature"], 0)
                else:
                    self.assertEqual(native["output_config"]["effort"], "high")
                    self.assertEqual(native["thinking"], {"type": "adaptive"})
                    self.assertTrue(native["tool_choice"]["disable_parallel_tool_use"])

    def test_truncation_never_becomes_completed(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                codec = CODECS[provider](profile(provider))
                self.assertEqual(
                    codec.normalize_response(
                        native_response(provider, [call("finish_task")], truncated=True)
                    )["status"],
                    "incomplete",
                )

    def test_unsupported_actions_and_partial_groups_fail(self):
        cases = {
            "responses": {
                "id": "r",
                "status": "completed",
                "output": [call("finish_task"), dict(type="web_search_call")],
            },
            "chat_completions": {
                "id": "r",
                "choices": [
                    dict(
                        message=dict(
                            role="assistant",
                            content=None,
                            tool_calls=[dict(type="custom", id="x")],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            },
            "anthropic": {
                "id": "r",
                "type": "message",
                "role": "assistant",
                "content": [
                    dict(type="tool_use", id="a", name="finish_task", input={}),
                    dict(type="server_tool_use"),
                ],
                "stop_reason": "tool_use",
            },
        }
        for provider, raw in cases.items():
            with self.subTest(provider=provider):
                body = CODECS[provider](profile(provider)).normalize_response(raw)
                self.assertEqual(body["status"], "failed")
                self.assertEqual(body["output"], [])

    def test_missing_or_conflicting_continuation_is_rejected(self):
        codec = CODECS["responses"](profile())
        marker = codec.remember(dict(type="reasoning", encrypted_content="a"), "r:0")
        with self.assertRaises(RuntimeInputError):
            codec.remember(dict(type="reasoning", encrypted_content="b"), "r:0")
        with self.assertRaises(RuntimeInputError):
            CODECS["responses"](profile()).serialize_request(
                request(profile(), [marker])
            )

    def test_response_reasoning_status_and_refusal_are_not_dropped(self):
        codec = CODECS["responses"](profile())
        for item in [
            dict(type="reasoning", status="in_progress"),
            dict(
                type="message",
                role="assistant",
                content=[dict(type="refusal", refusal="No")],
            ),
        ]:
            body = codec.normalize_response(
                dict(id="r", status="completed", output=[item, call("finish_task")])
            )
            self.assertEqual(body["status"], "failed")

    def test_profile_rejects_structural_overrides_and_secrets_in_url(self):
        for field in [
            "messages",
            "tools",
            "store",
            "max_tokens",
            "previous_response_id",
            "truncation",
            "parallel_tool_calls",
            "instructions",
            "functions",
            "background",
        ]:
            with self.subTest(field=field), self.assertRaises(RuntimeInputError):
                profile(parameters={field: None})
        for endpoint in [
            "https://user:secret@model.example",
            "https://model.example/?api_key=secret",
            "file:///tmp/model",
        ]:
            with self.subTest(endpoint=endpoint), self.assertRaises(RuntimeInputError):
                validate_model(
                    dict(
                        provider="responses",
                        model_id="x",
                        endpoint=endpoint,
                        max_output_tokens=1,
                    )
                )
        self.assertIsNone(profile()["reasoning_effort"])

    def test_full_native_bytes_are_checked(self):
        adapter = HTTPAdapter(profile("anthropic"))
        value = request(adapter.model)
        result = budget(adapter, value, 1)
        self.assertFalse(result["admissible"])
        self.assertEqual(
            result["native_bytes"],
            len(
                json.dumps(
                    adapter.serialize_request(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ),
        )


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_headers_exact_options_and_no_secret_trace(self):
        seen = []

        def handler(req):
            seen.append(req)
            return httpx.Response(
                200,
                json={
                    **native_response("responses", [call("finish_task")]),
                    "request_headers": {"Authorization": "secret-value"},
                },
                headers={"x-request-id": "request-secret-value"},
            )

        with patch.dict(
            os.environ,
            {"MY_CREDENTIAL": "secret-value", "EXTRA_HEADER": "special-value"},
        ):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                adapter = HTTPAdapter(
                    profile(
                        api_key_env="MY_CREDENTIAL",
                        headers_env={"X-Option": "EXTRA_HEADER"},
                        parameters={"temperature": 0},
                    ),
                    client=client,
                )
                adapter.check()
                result = await adapter.respond(
                    request(adapter.model), timeout_seconds=1
                )
        self.assertEqual(seen[0].headers["authorization"], "Bearer secret-value")
        self.assertEqual(seen[0].headers["x-option"], "special-value")
        self.assertEqual(json.loads(seen[0].content)["temperature"], 0)
        self.assertNotIn("secret-value", json.dumps(result.__dict__))
        self.assertNotIn("special-value", json.dumps(result.__dict__))

    async def test_http_error_and_redirect_make_one_request(self):
        for status in [400, 401, 429, 500, 307]:
            with self.subTest(status=status):
                seen = []

                def handler(req):
                    seen.append(req)
                    return httpx.Response(
                        status,
                        json={
                            "error": {
                                "message": "bad secret-value",
                                "debug": {"key": "secret-value"},
                            },
                            "request_headers": {"Authorization": "secret-value"},
                        },
                        headers={"location": "https://other.example"},
                    )

                with patch.dict(os.environ, {"MY_CREDENTIAL": "secret-value"}):
                    async with httpx.AsyncClient(
                        transport=httpx.MockTransport(handler)
                    ) as client:
                        adapter = HTTPAdapter(
                            profile(api_key_env="MY_CREDENTIAL"), client=client
                        )
                        result = await adapter.respond(
                            request(adapter.model), timeout_seconds=1
                        )
                self.assertEqual(len(seen), 1)
                self.assertEqual(result.body["status"], "failed")
                self.assertNotIn("secret-value", json.dumps(result.__dict__))

    async def test_invalid_json_and_response_size_fail_without_truncation(self):
        for payload in [
            b"not json",
            b"[]",
            b'{"id": NaN}',
            b'{"x": 1e999}',
            b'{"id":"first","id":"second"}',
        ]:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(200, content=payload)
                )
            ) as client:
                adapter = HTTPAdapter(profile(), client=client)
                result = await adapter.respond(
                    request(adapter.model), timeout_seconds=1
                )
                self.assertEqual(result.body["status"], "failed")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=b"large response")
            )
        ) as client:
            adapter = HTTPAdapter(profile(max_response_bytes=4), client=client)
            with self.assertRaisesRegex(RuntimeInputError, "exceeds"):
                await adapter.respond(request(adapter.model), timeout_seconds=1)

    async def test_timeout_is_charged_and_cancelled_once(self):
        calls, cancelled = [], []

        async def handler(req):
            calls.append(req)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = HTTPAdapter(profile(), client=client)
            timing = {"model_seconds": 0}
            with self.assertRaises(TimeoutError):
                await model_response(
                    adapter, request(adapter.model), timeout_seconds=0.02, timing=timing
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(cancelled, [True])
        self.assertGreaterEqual(timing["model_seconds"], 0.015)
        self.assertLess(timing["model_seconds"], 1)

    async def test_missing_credentials_and_conflicting_header_do_not_send(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = HTTPAdapter(profile(api_key_env="ABSENT_CREDENTIAL"))
            with self.assertRaisesRegex(RuntimeInputError, "ABSENT_CREDENTIAL"):
                adapter.check()
        with patch.dict(os.environ, {"KEY": "value"}):
            adapter = HTTPAdapter(profile(headers_env={"Content-Type": "KEY"}))
            with self.assertRaises(RuntimeInputError):
                adapter.check()


class NativeEngineTests(unittest.TestCase):
    def test_all_protocols_run_two_blind_judges_and_resume_offline(self):
        for provider in PROVIDERS:
            with self.subTest(
                provider=provider
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = fixture(root)
                settings = profile(provider)
                args["model_path"].write_text(json.dumps(settings))
                requests = []

                def agent_handler(req):
                    requests.append(json.loads(req.content))
                    items = (
                        [
                            call("slack_list_channels"),
                            call("slack_list_users", call_id="users"),
                        ]
                        if len(requests) == 1
                        else [call("finish_task", call_id="finish")]
                    )
                    return httpx.Response(
                        200,
                        json=native_response(
                            provider, items, f"agent-{len(requests)}", thinking=True
                        ),
                    )

                client = httpx.AsyncClient(transport=httpx.MockTransport(agent_handler))
                adapter = HTTPAdapter(settings, client=client)
                store = ArtifactStore(root / "results")
                attempt_args = dict(
                    **args,
                    work_dir=root / "attempt",
                    run_id="test",
                    attempt_id="1",
                    unit_key="runs/test/1",
                    adapter=adapter,
                    artifact_store=store,
                )
                attempt = run_attempt(**attempt_args)
                self.assertEqual(
                    attempt["run"]["stop_reason"],
                    "finish_task",
                    attempt["run"].get("error"),
                )
                self.assertEqual(len(requests), 2)
                # Pure native replay must work without resolving any API credential.
                replay_model = attempt["run"]["model_ref"]["settings"]
                replay = ReplayBudget(
                    attempt["bundle"],
                    replay_model,
                    protocol(),
                    attempt["run"]["resolved_limits"],
                    route_id=adapter.route_id,
                )
                for event in map(
                    json.loads,
                    (Path(attempt["bundle"]) / "trace.jsonl").read_text().splitlines(),
                ):
                    if event["kind"] == "model_request":
                        replay.request(event["payload"])
                    if event["kind"] == "model_response":
                        replay.response(event["request_id"], event["payload"])
                self.assertTrue(run_attempt(**attempt_args)["skipped"])
                self.assertEqual(len(requests), 2)
                prepared = prepare_anonymous_answer(
                    package=args["package"],
                    config_path=args["config_path"],
                    run_bundle=attempt["bundle"],
                )
                judgments = []
                for n in range(2):
                    judge_settings = {**settings, "model_id": f"judge-{n}"}
                    model_path = root / f"judge-{n}.json"
                    model_path.write_text(json.dumps(judge_settings))
                    seen = []

                    def judge_handler(req):
                        native = json.loads(req.content)
                        seen.append(native)
                        messages = (
                            native.get("input")
                            if provider == "responses"
                            else native["messages"]
                        )
                        user = next(m for m in messages if m.get("role") == "user")
                        content = user["content"]
                        inp = json.loads(
                            content if isinstance(content, str) else content[0]["text"]
                        )
                        source = next(
                            s["id"] for s in inp["sources"] if s["kind"] == "snapshot"
                        )
                        value = dict(
                            status="scored",
                            overall=10,
                            criteria={
                                name: dict(
                                    score=10 if i == 0 else None,
                                    reason="Mock verdict",
                                    evidence=[source] if i == 0 else [],
                                )
                                for i, name in enumerate(CRITERIA)
                            },
                            reason="Mock verdict",
                            evidence=[source],
                            reference_difference=dict(
                                material=False,
                                reason="No difference",
                                evidence=[],
                                task_review_reason=None,
                            ),
                        )
                        return httpx.Response(
                            200,
                            json=native_response(
                                provider,
                                [call("submit_judgment", value)],
                                f"judge-response-{n}",
                            ),
                        )

                    judge_client = httpx.AsyncClient(
                        transport=httpx.MockTransport(judge_handler)
                    )
                    judge_adapter = HTTPAdapter(judge_settings, client=judge_client)
                    kwargs = {
                        **args,
                        "model_path": model_path,
                        "run_bundle": attempt["bundle"],
                        "work_dir": root / f"judge-work-{n}",
                        "judgment_id": f"judge-{n}",
                        "unit_key": f"judgments/{n}",
                        "adapter": judge_adapter,
                        "artifact_store": store,
                        "anonymous_preparation": prepared,
                    }
                    judged = run_judgment(**kwargs)
                    self.assertEqual(
                        judged["judgment"]["status"],
                        "scored",
                        judged["judgment"].get("error"),
                    )
                    judgments.append(judged["judgment"])
                    self.assertNotIn("any/model:version", json.dumps(seen))
                    self.assertTrue(run_judgment(**kwargs)["skipped"])
                    self.assertEqual(len(seen), 1)
                    asyncio.run(judge_client.aclose())
                self.assertEqual(combine_judgments(*judgments)["overall"], 10)
                asyncio.run(client.aclose())

    def test_truncated_write_does_not_deliver_a_message(self):
        for provider in PROVIDERS:
            with self.subTest(
                provider=provider
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = fixture(root)
                settings = profile(provider)
                args["model_path"].write_text(json.dumps(settings))
                raw = native_response(
                    provider,
                    [
                        call(
                            "slack_post_message",
                            dict(channel_id="C_PUBLIC", text="Must not send"),
                        )
                    ],
                    truncated=True,
                )
                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda req: httpx.Response(200, json=raw)
                    )
                )
                result = run_attempt(
                    **args,
                    work_dir=root / "attempt",
                    run_id="test",
                    attempt_id="1",
                    unit_key="runs/test/1",
                    adapter=HTTPAdapter(settings, client=client),
                    artifact_store=ArtifactStore(root / "results"),
                )
                self.assertEqual(result["run"]["message_refs"], [])
                self.assertEqual(result["run"]["outcome"], "technical_error")
                asyncio.run(client.aclose())

    def test_custom_factory_and_implementation_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample_adapter.py"
            path.write_text(
                """from benchmark.runtime.provider_adapters import HTTPAdapter

def create(profile):
    return HTTPAdapter({**profile, "provider": "responses"})
"""
            )
            sys.path.insert(0, str(root))
            try:
                settings = profile("custom", adapter="sample_adapter:create")
                adapter = create_adapter(settings)
                self.assertEqual(
                    adapter.serialize_request(request(settings))["model"],
                    settings["model_id"],
                )
                raw = native_response("responses", [call("finish_task")])
                self.assertEqual(adapter.normalize_response(raw)["status"], "completed")
                first = adapter.route_id
                path.write_text(path.read_text() + "\n# Changed adapter\n")
                self.assertNotEqual(create_adapter(settings).route_id, first)
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("sample_adapter", None)
                importlib.invalidate_caches()

    def test_custom_adapter_native_trace_replays_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mock_bridge.py").write_text(
                "from benchmark.runtime.provider_adapters import HTTPAdapter\nCLIENT = None\ndef create(profile):\n    return HTTPAdapter({**profile, 'provider': 'responses'}, client=CLIENT)\n"
            )
            sys.path.insert(0, str(root))
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(
                        200,
                        json=native_response(
                            "responses", [call("finish_task")], thinking=True
                        ),
                    )
                )
            )
            try:
                module = importlib.import_module("mock_bridge")
                module.CLIENT = client
                settings = profile(
                    "custom",
                    adapter="mock_bridge:create",
                    api_key_env="CUSTOM_CREDENTIAL",
                )
                args = fixture(root)
                args["model_path"].write_text(json.dumps(settings))
                adapter = create_adapter(settings)
                with patch.dict(
                    os.environ, {"CUSTOM_CREDENTIAL": "private-credential"}
                ):
                    attempt = run_attempt(
                        **args,
                        work_dir=root / "attempt",
                        run_id="custom",
                        attempt_id="1",
                        unit_key="runs/custom/1",
                        adapter=adapter,
                        artifact_store=ArtifactStore(root / "results"),
                    )
                self.assertEqual(
                    attempt["run"]["stop_reason"],
                    "finish_task",
                    attempt["run"].get("error"),
                )
                with patch.dict(os.environ, {}, clear=True):
                    replay = ReplayBudget(
                        attempt["bundle"],
                        settings,
                        protocol(),
                        attempt["run"]["resolved_limits"],
                        route_id=adapter.route_id,
                    )
                    for event in map(
                        json.loads,
                        (Path(attempt["bundle"]) / "trace.jsonl")
                        .read_text()
                        .splitlines(),
                    ):
                        if event["kind"] == "model_request":
                            replay.request(event["payload"])
                        if event["kind"] == "model_response":
                            replay.response(event["request_id"], event["payload"])
                native_path = Path(attempt["bundle"]) / "provider-native.jsonl"
                self.assertNotIn("private-credential", native_path.read_text())
                events = [
                    json.loads(line) for line in native_path.read_text().splitlines()
                ]
                events[0]["payload"]["model"] = "tampered"
                native_path.write_text(
                    "".join(json.dumps(event) + "\n" for event in events)
                )
                changed = ReplayBudget(
                    attempt["bundle"],
                    settings,
                    protocol(),
                    attempt["run"]["resolved_limits"],
                    route_id=adapter.route_id,
                )
                with self.assertRaisesRegex(ValueError, "Native request differs"):
                    changed.request(request(settings))
            finally:
                asyncio.run(client.aclose())
                sys.path.remove(str(root))
                sys.modules.pop("mock_bridge", None)
                importlib.invalidate_caches()

    def test_cli_profile_check_makes_no_network_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(profile(api_key_env="TEST_MODEL_KEY")))
            env = {
                **os.environ,
                "TEST_MODEL_KEY": "private-test-value",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            result = subprocess.run(
                [sys.executable, "-m", "benchmark", "model", "check", str(path)],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No API call made", result.stdout)
            self.assertNotIn("private-test-value", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
