"""Model-neutral response and timing contract."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .contracts import load_json


@dataclass
class ModelResponse:
    body: dict
    request_id: str | None = None
    http_status: int | None = None
    native_trace: dict | None = None
    rate_limits: dict | None = None


async def model_response(adapter, request, *, timeout_seconds, timing, monotonic=None):
    """Admission waits do not consume the pinned model-response budget."""
    if hasattr(adapter, "respond_with_budget"):
        return await adapter.respond_with_budget(
            request, timeout_seconds=timeout_seconds, timing=timing
        )
    clock = monotonic or time.monotonic
    started = clock()
    try:
        return await asyncio.wait_for(
            adapter.respond(request, timeout_seconds=timeout_seconds),
            timeout=timeout_seconds,
        )
    finally:
        timing["model_seconds"] += clock() - started


def validate_model(value):
    """Validate an explicit model profile without a model-name allowlist."""
    from .model_config import validate_model as validate

    return validate(value)


def load_model(path):
    return validate_model(load_json(path))


def load_judge_model(path):
    return load_model(path)
