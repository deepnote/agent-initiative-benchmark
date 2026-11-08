"""Safe owner-only failure diagnostics, separate from business/tool content."""

from __future__ import annotations

import os
import re


class RuntimeInputError(ValueError):
    """Known runtime validation, safe to describe after diagnostic redaction."""

    code = "invalid_runtime_input"


_SECRET_NAMES = {
    "AZURE_AI_API_KEY",
    "OPENAI_API_KEY",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}


def configured_secrets():
    """Shared credential aliases for diagnostic redaction and output quarantine."""
    return tuple(
        secret
        for name, secret in os.environ.items()
        if secret
        and (
            name in _SECRET_NAMES
            or name.endswith(("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))
        )
    )


def safe_text(value, *, limit=2048, secrets=()):
    if not isinstance(value, str):
        return None
    # Environment values are never emitted. Redact before imposing the cap so
    # truncation cannot leave a visible fragment of a configured credential.
    configured = configured_secrets()
    for secret in sorted(
        set(configured) | {s for s in secrets if isinstance(s, str)},
        key=len,
        reverse=True,
    ):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"https?://[^\s<>\"']+", "[REDACTED URL]", value, flags=re.I)
    value = re.sub(
        r"\b(?:Bearer|Basic)\s+[^\s,;\"']+",
        "[REDACTED authorization]",
        value,
        flags=re.I,
    )
    credential = (
        r"\b(?:api[_ -]?key|(?:secret[_ -]?access[_ -]?|access[_ -]?)key|password|"
        r"(?:client[_ -]?)?secret|(?:access[_ -]?|refresh[_ -]?|session[_ -]?)?token|authorization)"
        r"[\"']?\s*[:=]\s*[\"']?[^\s,;\"']+"
    )
    value = re.sub(credential, "[REDACTED credential]", value, flags=re.I)
    return " ".join(value.split())[:limit]


def safe_code(value, fallback="unclassified_exception"):
    value = safe_text(value, limit=128)
    return value if value and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) else fallback


def exception_cause(error, *, trusted_reason=False):
    """Unknown exception messages/SDK debug text never enter saved diagnostics."""
    cause = {
        "type": safe_code(type(error).__name__, "Exception"),
        "code": safe_code(getattr(error, "code", None)),
        "reason": (
            safe_text(str(error)) if trusted_reason else "Exception details omitted"
        ),
    }
    validation_reason = getattr(error, "validation_reason", None)
    if trusted_reason and validation_reason is not None:
        cause["reason"] = safe_text(validation_reason)
    record = getattr(error, "record", None)
    if trusted_reason and isinstance(record, dict):
        cause["record"] = {
            key: safe_text(value, limit=256)
            for key, value in record.items()
            if key
            in {
                "channel_id",
                "ts",
                "message_ts",
                "name",
                "user_id",
                "table",
                "id",
                "issue_id",
                "label_id",
            }
        }
    source = getattr(error, "source", None)
    if trusted_reason and isinstance(source, str):
        cause["source"] = safe_text(source, limit=256)
    return cause


def provider_body(body, *, http_status=None, secrets=()):
    """Keep output/business values exact; sanitize only failure diagnostics."""
    if not isinstance(body, dict):
        return {
            "status": "failed",
            "error": {
                "code": "invalid_provider_response",
                "message": "Provider response JSON was not an object",
            },
        }
    if (
        body.get("status") == "completed"
        and not body.get("error")
        and not (http_status and http_status >= 400)
    ):
        return body
    if http_status and http_status >= 400:
        # HTTP error bodies are diagnostics, not successful model/business
        # output. Do not retain arbitrary server/proxy debug extensions.
        result = {
            key: value
            for key, value in body.items()
            if key
            in {
                "status",
                "id",
                "error",
                "incomplete_details",
                "output",
                "usage",
                "model",
                "model_version",
            }
        }
    else:
        result = {
            key: value
            for key, value in body.items()
            if key
            not in {
                "debug",
                "sdk_debug",
                "headers",
                "request_headers",
                "http_request",
                "http_response",
                "transport",
            }
        }
    if isinstance(result.get("id"), str):
        result["id"] = safe_text(result["id"], limit=256, secrets=secrets)
    # Error-body extensions can be SDK/transport debug objects. They are not
    # agent output. Preserve only the documented diagnostic scalar fields.
    for field, allowed in (
        ("error", {"code", "message", "type", "param"}),
        ("incomplete_details", {"reason"}),
    ):
        details = body.get(field)
        if isinstance(details, dict):
            result[field] = {
                key: safe_text(value, secrets=secrets)
                for key, value in details.items()
                if key in allowed and (value is None or isinstance(value, str))
            }
    return result


def provider_cause(body, *, http_status=None, request_id=None):
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    incomplete = (
        body.get("incomplete_details")
        if isinstance(body.get("incomplete_details"), dict)
        else {}
    )
    cause = {"type": "ProviderResponse", "code": "provider_incomplete"}
    error_code = safe_code(error.get("code"), fallback="")
    incomplete_reason = safe_code(incomplete.get("reason"), fallback="")
    if error_code:
        cause["provider_error_code"] = error_code
    if incomplete_reason:
        cause["incomplete_reason"] = incomplete_reason
    cause["code"] = error_code or incomplete_reason or "provider_incomplete"
    status = safe_code(body.get("status"), "failed")
    reason = f"Provider response {status}: {cause['code']}"
    if incomplete_reason == "max_output_tokens":
        reason += "; maximum output token budget reached"
    message = safe_text(error.get("message"))
    if message:
        reason += "; " + message
    if http_status is not None:
        cause["http_status"] = http_status
        reason += f" (HTTP {http_status})"
    for field, value in (
        ("provider_request_id", request_id),
        ("response_id", body.get("id")),
    ):
        if isinstance(value, str):
            cause[field] = safe_text(value, limit=256)
    cause["reason"] = safe_text(reason)
    return cause
