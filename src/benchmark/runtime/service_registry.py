"""Explicit runtime services in public tool order. No discovery or storage owner."""

from __future__ import annotations


from benchmark.prompts import load_text

from . import (
    slack_service,
    linear_service,
    notion_service,
    hubspot_service,
    stripe_service,
    calendar_service,
    posthog_service,
    analytics_service,
    search_console_service,
    bigquery_service,
    snowflake_service,
)

# Slack owns initial context and the only business writes. New evidence services
# join this tuple and supply the small module interface in SERVICE_REGISTRY.md.
SERVICES = (
    slack_service,
    linear_service,
    notion_service,
    hubspot_service,
    stripe_service,
    calendar_service,
    posthog_service,
    analytics_service,
    search_console_service,
    bigquery_service,
    snowflake_service,
)
CONTEXT_SERVICE = "slack"


def service_names():
    return tuple(service.NAME for service in SERVICES)


def read_names():
    return frozenset(name for service in SERVICES for name in service.READS)


def write_names():
    # A new adapter cannot grant business writes.
    return slack_service.WRITES


def tool_names():
    return read_names() | write_names()


def service_for(name):
    return next((service for service in SERVICES if service.NAME == name), None)


def tool_service(name):
    return next(
        (
            service
            for service in SERVICES
            if name in service.READS
            or (service.NAME == CONTEXT_SERVICE and name in write_names())
        ),
        None,
    )


def enabled_services(scope):
    return tuple(
        service
        for service in SERVICES
        if set(scope)
        & (
            service.READS
            | (write_names() if service.NAME == CONTEXT_SERVICE else set())
        )
    )


def definitions(scope, cap, *, read_only=False):
    selected = set(scope) & (read_names() if read_only else tool_names())
    return [
        definition
        for service in SERVICES
        for definition in service.definitions(selected, cap)
        if definition["name"] in selected
    ]


def reader_definitions(readers, *, read_only=False):
    allowed = read_names() if read_only else tool_names()
    return [
        definition
        for service in SERVICES
        for definition in readers[service.NAME].tool_definitions
        if definition["name"] in allowed
    ]


def agent_definition(definition):
    # Only the tested agent opts into compact PostHog discovery. Native reads,
    # Saved evidence reads retain their full default.
    return posthog_service.agent_definition(definition)


def agent_arguments(name, arguments):
    return posthog_service.agent_arguments(name, arguments)


def dispatch(readers, name, arguments, *, read_only=False):
    if read_only and name not in read_names():
        return {
            "ok": False,
            "error": "forbidden_write",
            "message": "Only configured reads are allowed",
        }
    service = tool_service(name)
    if service is not None:
        return readers[service.NAME].call(name, arguments)
    # The context environment also owns internal judge fragment reads and the
    # existing unknown-tool error. Other service prefixes never reach it.
    if name.startswith("_") or name.startswith(CONTEXT_SERVICE + "_"):
        return readers[CONTEXT_SERVICE].call(name, arguments)
    return {
        "ok": False,
        "error": "unknown_tool",
        "message": "Unknown or disallowed tool",
    }


def required_files(baseline, scope):
    return tuple(
        path
        for service in SERVICES
        if service.NAME == CONTEXT_SERVICE or service in enabled_services(scope)
        for path in service.required_files(baseline)
    )


def local_roots(roots=None, **aliases):
    """Normalize explicit input aliases, rejecting conflicting source roots."""
    from pathlib import Path
    from .failure_causes import RuntimeInputError

    if roots is not None and not isinstance(roots, dict):
        raise RuntimeInputError("Local service roots must be a map")
    result = {k: v for k, v in (roots or {}).items() if v is not None}
    for name, path in aliases.items():
        if path is None:
            continue
        if name in result and Path(result[name]).resolve() != Path(path).resolve():
            raise RuntimeInputError("Conflicting local roots for " + name)
        result[name] = path
    return result


def validate_local_inputs(scope, mode, roots):
    from .failure_causes import RuntimeInputError

    if set(roots) - set(service_names()):
        raise RuntimeInputError("Local roots must name registered services")
    for service in SERVICES:
        if service.NAME != CONTEXT_SERVICE:
            service.validate_local(
                mode, service in enabled_services(scope), roots.get(service.NAME)
            )


def discovery_guide(scope):
    guides = [load_text("guides/shared/read_rules.txt")]
    guides.extend(
        text for service in SERVICES if (text := service.discovery_guide(scope))
    )
    return "\n".join(guides)


def render_discovery_guides(instruction, scope):
    # Insert the configured service guidance once.
    markers = ("{service_guide}",)
    positions = [
        (instruction.index(marker), marker)
        for marker in markers
        if marker in instruction
    ]
    guide = discovery_guide(scope)
    if not positions:
        return instruction + "\n\n" + guide
    position, marker = min(positions)
    instruction = instruction[:position] + guide + instruction[position + len(marker) :]
    for marker in markers:
        instruction = instruction.replace(marker, "")
    return instruction


render_guides = render_discovery_guides


def supports_delta(path):
    return any(path.startswith("delta/" + name + "/") for name in service_names())


def native_success(name, result):
    service = tool_service(name)
    return service is not None and service.native_success(name, result)


def native_error(name, result):
    service = tool_service(name)
    return service is not None and getattr(service, "native_error", lambda *_: False)(
        name, result
    )


def evidence_entries(name, result):
    service = tool_service(name) if name is not None else service_for(CONTEXT_SERVICE)
    return service.evidence_entries(result) if service is not None else ()


def trace_facts(name, result):
    service = tool_service(name)
    return service.trace_facts(name, result) if service is not None else ()


def records_native_trace(name):
    # Existing Linear field locators refer to a separate, exact native event.
    service = tool_service(name)
    return service is not None and service.NAME != CONTEXT_SERVICE


def trusted_error(error):
    return any(service.trusted_error(error) for service in SERVICES)


def receipt_paths_valid(baseline, paths):
    required = set(service_for(CONTEXT_SERVICE).required_files(baseline))
    allowed = set(required)
    for service in SERVICES:
        if service.NAME != CONTEXT_SERVICE and any(
            path.startswith(service.NAME + "/") for path in paths
        ):
            allowed.update(
                service.receipt_paths(baseline, paths)
                if hasattr(service, "receipt_paths")
                else getattr(service, "receipt_files", service.required_files)(baseline)
            )
            required.update(service.required_files(baseline))
            # Linked artifacts must exist in the same pinned dataset inventory.
            linked = getattr(service, "linked_file", None)
            if linked is not None:
                for path in paths:
                    if linked(path):
                        baseline.file_inventory((path,))
                        allowed.add(path)
    return required.issubset(paths) and set(paths) <= allowed
