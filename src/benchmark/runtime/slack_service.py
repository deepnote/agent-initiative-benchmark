"""Service-owned runtime wiring. Native libraries own storage and queries."""

from __future__ import annotations


NAME = "slack"
READS = frozenset(
    (
        "slack_list_channels",
        "slack_list_users",
        "slack_search_messages",
        "slack_get_thread",
    )
)
WRITES = frozenset(("slack_post_message", "slack_reply_to_thread", "slack_send_dm"))
DELTA_FILE = "delta/slack/seed.json"


def definitions(scope, cap):
    from .slack_environment import tool_definitions

    return tool_definitions(scope, cap)


def required_files(baseline):
    return baseline.plan_service(NAME).paths


def local_file(path):
    return path.endswith(".parquet")


def discovery_guide(scope):
    from benchmark.prompts import load_text

    return load_text("guides/slack/data_overview.txt") if set(scope) & READS else ""


def evidence_entries(result):
    if not result.get("ok"):
        return
    for message in result.get("messages", []):
        record = {
            key: message.get(key)
            for key in ("channel_id", "user_id", "ts", "thread_ts", "text")
        }
        yield (
            "slack:" + record["channel_id"] + ":" + record["ts"],
            "snapshot",
            {
                "service": NAME,
                "record": {"channel_id": record["channel_id"], "ts": record["ts"]},
            },
            record,
        )
    for key, entity in (("channels", "channel"), ("users", "user")):
        for row in result.get(key, []):
            yield f"slack:{entity}:{row['id']}", "snapshot", {
                "service": NAME,
                "entity": entity,
                "id": row["id"],
            }, row
    if isinstance(result.get("channel"), dict):
        row = result["channel"]
        yield f"slack:channel:{row['id']}", "snapshot", {
            "service": NAME,
            "entity": "channel",
            "id": row["id"],
        }, row


def native_success(name, result):
    return False  # Slack uses the ordinary explicit ok envelope.


def trace_facts(name, result):
    return ()


def trusted_error(error):
    from .slack_environment import SlackEnvironmentError

    return isinstance(error, SlackEnvironmentError)
