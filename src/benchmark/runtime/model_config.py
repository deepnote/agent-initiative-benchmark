"""Portable model profiles. Credentials are named by environment variable."""

import copy
import json
import re
from urllib.parse import parse_qsl, urlsplit

from .failure_causes import RuntimeInputError

FIELDS = {
    "schema_version",
    "provider",
    "model_id",
    "max_output_tokens",
    "reasoning_effort",
    "endpoint",
    "api_key_env",
    "auth_header",
    "auth_prefix",
    "api_version",
    "headers_env",
    "parameters",
    "instruction_role",
    "token_limit_field",
    "max_response_bytes",
    "adapter",
    "judge_parallel_reads",
}
PROTECTED = {
    "model",
    "input",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "store",
    "stream",
    "n",
    "include",
    "parallel_tool_calls",
    "previous_response_id",
    "conversation",
    "truncation",
    "context_management",
    "mcp_servers",
    "plugins",
    "web_search_options",
    "instructions",
    "prompt",
    "functions",
    "function_call",
    "background",
    "stream_options",
    "max_tool_calls",
}
HEADER = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
ENVIRONMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def require(condition, message):
    if not condition:
        raise RuntimeInputError(message)


def validate_model(value):
    require(
        isinstance(value, dict)
        and {"provider", "model_id", "max_output_tokens"} <= value.keys(),
        "Model profile requires provider, model_id and max_output_tokens",
    )
    require(not value.keys() - FIELDS, "Unknown model profile fields")
    result = copy.deepcopy(value)
    result.setdefault("schema_version", 1)
    require(
        type(result["schema_version"]) is int and result["schema_version"] == 1,
        "Model schema_version must be 1",
    )
    for name in ("provider", "model_id"):
        require(
            isinstance(result[name], str)
            and bool(result[name].strip())
            and result[name] == result[name].strip(),
            f"Model {name} must be explicit text",
        )
    require(
        type(result["max_output_tokens"]) is int and result["max_output_tokens"] > 0,
        "max_output_tokens must be positive",
    )
    result.setdefault("judge_parallel_reads", True)
    require(
        type(result["judge_parallel_reads"]) is bool,
        "judge_parallel_reads must be boolean",
    )
    result.setdefault("parameters", {})
    require(isinstance(result["parameters"], dict), "parameters must be an object")
    require(
        not result["parameters"].keys() & PROTECTED,
        "parameters cannot override benchmark messages, tools, storage or limits",
    )
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError):
        raise RuntimeInputError(
            "Model settings must contain finite JSON values"
        ) from None
    params = result["parameters"]
    nested = (
        params.get("reasoning", {})
        if result["provider"] == "responses"
        else params.get("output_config", {})
    )
    effort = nested.get("effort") if isinstance(nested, dict) else None
    if result["provider"] == "chat_completions":
        effort = params.get("reasoning_effort")
    result.setdefault("reasoning_effort", effort)
    require(
        result["reasoning_effort"] is None
        or isinstance(result["reasoning_effort"], str)
        and bool(result["reasoning_effort"].strip()),
        "reasoning_effort must be text or null",
    )
    require(
        effort is None or effort == result["reasoning_effort"],
        "Conflicting reasoning effort settings",
    )
    result.setdefault("api_key_env", None)
    require(
        result["api_key_env"] is None
        or isinstance(result["api_key_env"], str)
        and ENVIRONMENT.fullmatch(result["api_key_env"]),
        "api_key_env must name an environment variable",
    )
    result.setdefault("auth_header", "Authorization")
    result.setdefault("auth_prefix", "Bearer ")
    require(
        isinstance(result["auth_header"], str)
        and HEADER.fullmatch(result["auth_header"])
        and result["auth_header"].lower()
        not in {
            "host",
            "content-type",
            "content-length",
            "transfer-encoding",
            "anthropic-version",
        },
        "Invalid auth_header",
    )
    require(
        isinstance(result["auth_prefix"], str)
        and not re.search(r"[\r\n]", result["auth_prefix"]),
        "Invalid auth_prefix",
    )
    result.setdefault("headers_env", {})
    require(isinstance(result["headers_env"], dict), "headers_env must be an object")
    for name, variable in result["headers_env"].items():
        require(
            isinstance(name, str)
            and HEADER.fullmatch(name)
            and isinstance(variable, str)
            and ENVIRONMENT.fullmatch(variable),
            "headers_env maps header names to environment variable names",
        )
    result.setdefault("instruction_role", "system")
    require(
        isinstance(result["instruction_role"], str)
        and result["instruction_role"] in {"system", "developer"},
        "instruction_role must be system or developer",
    )
    result.setdefault("token_limit_field", "max_completion_tokens")
    require(
        isinstance(result["token_limit_field"], str)
        and result["token_limit_field"] in {"max_tokens", "max_completion_tokens"},
        "Invalid Chat output-token field",
    )
    result.setdefault("max_response_bytes", 134217728)
    require(
        type(result["max_response_bytes"]) is int and result["max_response_bytes"] > 0,
        "max_response_bytes must be positive",
    )
    if result["provider"] == "anthropic":
        result.setdefault("api_version", "2026-10-21")
    if "api_version" in result:
        require(
            isinstance(result["api_version"], str)
            and bool(result["api_version"])
            and not re.search(r"[\r\n]", result["api_version"]),
            "Invalid API version",
        )
    if "endpoint" in result:
        require(isinstance(result["endpoint"], str), "endpoint must be a URL")
        try:
            url = urlsplit(result["endpoint"])
            valid = (
                url.scheme in {"http", "https"}
                and bool(url.hostname)
                and not url.username
                and not url.password
                and not url.fragment
            )
            _ = url.port
            valid = valid and not any(
                k.lower()
                in {
                    "key",
                    "api_key",
                    "api-key",
                    "token",
                    "access_token",
                    "password",
                    "signature",
                    "sig",
                }
                for k, _ in parse_qsl(url.query)
            )
            require(valid, "endpoint must be an HTTP URL without embedded credentials")
        except ValueError:
            raise RuntimeInputError("Invalid endpoint URL") from None
    if "adapter" in result:
        require(
            isinstance(result["adapter"], str)
            and re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", result["adapter"]),
            "adapter must be module:factory",
        )
    return result
