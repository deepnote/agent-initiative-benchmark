"""Explicit HTTP providers and importable custom adapters."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path

import httpx

from .failure_causes import RuntimeInputError, provider_body, safe_text
from .model_config import validate_model
from .provider_codecs import CODECS, encoded, failed
from .responses_adapter import ModelResponse


def diagnostic_error(value, secrets=()):
    if isinstance(value, dict):
        return {
            k: safe_text(v, secrets=secrets)
            for k, v in value.items()
            if k in {"code", "message", "type", "param"}
            and (v is None or isinstance(v, str))
        }
    if isinstance(value, str):
        return {"message": safe_text(value, secrets=secrets)}
    return None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate response field")
        result[key] = value
    return result


def clean_record(model, raw, status, secrets=()):
    """Keep model content exact and omit transport/debug extensions."""
    if not 200 <= status < 300:
        error = diagnostic_error(raw.get("error"), secrets)
        return {
            "error": error
            or {"code": "http_error", "message": f"Model API returned HTTP {status}"}
        }
    fields = {
        "responses": {
            "id",
            "status",
            "output",
            "error",
            "incomplete_details",
            "usage",
            "model",
            "model_version",
        },
        "chat_completions": {"id", "choices", "model", "usage", "error"},
        "anthropic": {
            "id",
            "type",
            "role",
            "content",
            "model",
            "usage",
            "stop_reason",
            "stop_sequence",
            "error",
        },
    }
    result = {
        k: copy.deepcopy(v) for k, v in raw.items() if k in fields[model["provider"]]
    }
    if "error" in result:
        result["error"] = diagnostic_error(result["error"], secrets)
    if "incomplete_details" in result:
        result["incomplete_details"] = provider_body(
            {"incomplete_details": result["incomplete_details"]}, secrets=secrets
        ).get("incomplete_details")
    return result


def normalize_record(adapter, raw, status):
    if (
        isinstance(adapter, HTTPAdapter)
        and status is not None
        and not 200 <= status < 300
    ):
        body = failed("http_error")
        body["error"] = raw["error"]
        return body
    return provider_body(adapter.normalize_response(raw))


def route_id(model, implementation=None):
    raw = encoded(dict(model=model, implementation=implementation)).encode()
    return "api-" + hashlib.sha256(raw).hexdigest()


def environment(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeInputError(f"Required environment variable {name} is missing")
    if "\r" in value or "\n" in value:
        raise RuntimeInputError(f"Invalid header value in {name}")
    return value


class HTTPAdapter:
    def __init__(self, model, *, client=None):
        self.model = validate_model(model)
        if self.model["provider"] not in CODECS or "endpoint" not in self.model:
            raise RuntimeInputError(
                "HTTP model requires a supported provider and endpoint"
            )
        self.codec = CODECS[self.model["provider"]](self.model)
        self.route_id = route_id(self.model)
        self._client = client
        self._owns_client = client is None

    def serialize_request(self, request):
        return self.codec.serialize_request(request)

    def normalize_response(self, body):
        return self.codec.normalize_response(body)

    def headers(self):
        headers = {"content-type": "application/json"}
        secrets = []
        if self.model["api_key_env"] is not None:
            key = environment(self.model["api_key_env"])
            headers[self.model["auth_header"].lower()] = self.model["auth_prefix"] + key
            secrets.append(key)
        if self.model["provider"] == "anthropic":
            headers["anthropic-version"] = self.model["api_version"]
        for name, variable in self.model["headers_env"].items():
            if name.lower() in headers or name.lower() in {
                "host",
                "content-length",
                "transfer-encoding",
            }:
                raise RuntimeInputError("Custom header conflicts with request headers")
            value = environment(variable)
            headers[name.lower()] = value
            secrets.append(value)
        return headers, tuple(secrets)

    def check(self):
        self.headers()

    async def respond(self, request, *, timeout_seconds):
        if (
            type(timeout_seconds) not in (float, int)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise RuntimeInputError("API timeout must be positive and finite")
        native = self.serialize_request(request)
        headers, secrets = self.headers()
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=0),
            )
        try:
            async with self._client.stream(
                "POST",
                self.model["endpoint"],
                headers=headers,
                content=encoded(native).encode(),
                timeout=timeout_seconds,
                follow_redirects=False,
            ) as response:
                parts, size = [], 0
                async for part in response.aiter_bytes():
                    size += len(part)
                    if size > self.model["max_response_bytes"]:
                        raise RuntimeInputError(
                            "Complete API response exceeds max_response_bytes"
                        )
                    parts.append(part)
                status = response.status_code
                try:
                    raw = json.loads(
                        b"".join(parts),
                        object_pairs_hook=unique_object,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                    )
                    encoded(raw)
                except (ValueError, UnicodeError):
                    raw = failed("invalid_response_json")
                if not isinstance(raw, dict):
                    raw = failed("invalid_response_object")
                request_id = safe_text(
                    response.headers.get("x-request-id")
                    or response.headers.get("request-id"),
                    secrets=secrets,
                    limit=256,
                )
        except httpx.TimeoutException:
            raise TimeoutError("Model API request timed out") from None
        except httpx.HTTPError:
            raise RuntimeInputError("Model API connection failed") from None
        raw = clean_record(self.model, raw, status, secrets)
        body = normalize_record(self, raw, status)
        trace = dict(
            protocol=self.model["provider"],
            route_id=self.route_id,
            response=raw,
            http_status=status,
        )
        return ModelResponse(body, request_id, status, trace)

    async def close(self):
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class CustomAdapter:
    def __init__(self, model, adapter, implementation):
        self.model, self.delegate = model, adapter
        self.route_id = route_id(model, implementation)
        for method in ("serialize_request", "normalize_response", "respond"):
            if not callable(getattr(adapter, method, None)):
                raise RuntimeInputError(f"Custom adapter requires {method}")

    def serialize_request(self, request):
        return self.delegate.serialize_request(request)

    def normalize_response(self, body):
        return self.delegate.normalize_response(body)

    async def respond(self, request, *, timeout_seconds):
        response = await self.delegate.respond(request, timeout_seconds=timeout_seconds)
        if (
            not isinstance(response, ModelResponse)
            or not isinstance(response.native_trace, dict)
            or not isinstance(response.native_trace.get("response"), dict)
        ):
            raise RuntimeInputError(
                "Custom adapter must return ModelResponse with its native response trace"
            )
        response.native_trace = {
            "protocol": "custom",
            "route_id": self.route_id,
            "response": response.native_trace["response"],
            "http_status": response.http_status,
        }
        return response

    def check(self):
        if self.model["api_key_env"] is not None:
            environment(self.model["api_key_env"])
        for variable in self.model["headers_env"].values():
            environment(variable)
        check = getattr(self.delegate, "check", None)
        if check:
            check()

    async def close(self):
        close = getattr(self.delegate, "close", None)
        if close:
            value = close()
            if inspect.isawaitable(value):
                await value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def create_adapter(model, *, client=None):
    model = validate_model(model)
    if model["provider"] in CODECS:
        return HTTPAdapter(model, client=client)
    if model["provider"] != "custom" or "adapter" not in model:
        raise RuntimeInputError(
            "provider must be responses, chat_completions, anthropic or custom"
        )
    module_name, factory_name = model["adapter"].split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise RuntimeInputError(
            "Cannot import the custom adapter module. Install it or add its folder to PYTHONPATH"
        ) from None
    factory = getattr(module, factory_name, None)
    if not callable(factory) or not getattr(module, "__file__", None):
        raise RuntimeInputError("Custom adapter must be an importable Python factory")
    implementation = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return CustomAdapter(model, factory(copy.deepcopy(model)), implementation)
