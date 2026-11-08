"""Read saved scores and messages without data, credentials or model calls."""

import csv
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from statistics import mean

from .asset_download import require
from .configuration import ROLES
from .runtime.artifact_store import PAYLOAD
from .runtime.judge_rubrics import criterion_labels
from .runtime.runtime_identity import write_json
from .scoring import combine_judgments


def display(value):
    text = str(value)
    return "".join(
        (
            f"\\x{ord(c):02x}"
            if (ord(c) < 32 and c not in "\n\t") or 127 <= ord(c) < 160
            else c
        )
        for c in text
    )


def score(value):
    return f"{value:.2f}" if value is not None else "-"


def build_summary(session):
    rows = []
    for ident, task in session.tasks.items():
        values = {r: session.payload(ident, r) for r in ROLES}
        agent = values["agent"]
        judges = [values[r] for r in ROLES[1:]]
        combined = dict(status="incomplete", overall=None, absolute_difference=None)
        error = None
        if agent is None:
            status = "interrupted" if session.unit(ident, "agent") else "pending"
        elif agent["stop_reason"] == "cancelled":
            status = "interrupted"
        elif agent["outcome"] != "completed" and agent["stop_reason"] not in {
            "model_budget",
            "model_timeout",
        }:
            status = "agent_error"
            error = (agent.get("error") or {}).get("reason") or agent["stop_reason"]
        elif ident in session.progress["evaluation_errors"]:
            status = "evaluation_error"
            error = session.progress["evaluation_errors"][ident]
        elif any(
            j is not None and (j.get("error") or {}).get("code") == "cancelled"
            for j in judges
        ):
            status = "interrupted"
        elif all(j is not None for j in judges):
            combined = combine_judgments(*judges)
            status = {
                "complete": "scored",
                "task_fault": "task_error",
                "needs_repair": "judge_error",
                "incomplete": "unscored",
            }[combined["status"]]
            error = (
                " | ".join(
                    (j.get("error") or {}).get("reason") or j.get("reason", "")
                    for j in judges
                    if j["status"] != "scored"
                )
                or None
            )
        else:
            status = "judging"
        row = dict(
            id=ident,
            title=task["title"],
            category=task["category"],
            intent=task["intent"],
            status=status,
            overall=combined["overall"],
            disagreement=combined["absolute_difference"],
            judge_scores=[
                j["overall"] if j and j["status"] == "scored" else None for j in judges
            ],
            judge_status=[j["status"] if j else "pending" for j in judges],
            agent_stop=agent["stop_reason"] if agent else None,
            messages=len(agent["message_refs"]) if agent else None,
            attempts={r: len(session.progress["tasks"][ident][r]) for r in ROLES},
            error=error,
        )
        rows.append(row)
    scored = [r for r in rows if r["status"] == "scored"]
    categories = {}
    for name in dict.fromkeys(r["category"] for r in rows):
        members = [r for r in rows if r["category"] == name]
        scores = [r["overall"] for r in members if r["overall"] is not None]
        categories[name] = dict(
            scored=len(scores),
            selected=len(members),
            mean=mean(scores) if scores else None,
        )
    silence = [r for r in rows if r["intent"] == "silence"]
    silence_scored = [r["overall"] for r in silence if r["overall"] is not None]
    return dict(
        schema_version=1,
        run_id=session.settings["run_id"],
        created_at=session.settings["created_at"],
        scope="Public example set",
        models={r: m["model_id"] for r, m in session.settings["models"].items()},
        tasks=rows,
        selected=len(rows),
        scored=len(scored),
        overall=mean(r["overall"] for r in scored) if scored else None,
        errors=sum(
            r["status"].endswith("_error") or r["status"] == "unscored" for r in rows
        ),
        pending=sum(r["status"] in {"pending", "judging", "interrupted"} for r in rows),
        categories=categories,
        silence=dict(
            selected=len(silence),
            scored=len(silence_scored),
            mean=mean(silence_scored) if silence_scored else None,
            messages=sum(r["messages"] or 0 for r in silence),
        ),
    )


def save_summary(session, summary):
    write_json(session.root / "summary.json", summary)


def format_summary(summary):
    rows = [["Task", "Category", "Status", "Judge 1", "Judge 2", "Mean", "Messages"]]
    rows.extend(
        [
            [
                r["id"],
                r["category"],
                r["status"].replace("_", " "),
                score(r["judge_scores"][0]),
                score(r["judge_scores"][1]),
                score(r["overall"]),
                str(r["messages"]) if r["messages"] is not None else "-",
            ]
            for r in summary["tasks"]
        ]
    )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        f'{summary["scope"]} | Run {summary["run_id"]}',
        f'Agent: {summary["models"]["agent"]} | Judges: {summary["models"]["judge-1"]}, {summary["models"]["judge-2"]}',
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )
    lines.append(
        f'Scored {summary["scored"]}/{summary["selected"]} | Errors {summary["errors"]} | Pending {summary["pending"]}'
    )
    lines.append(
        f'Mean of scored tasks: {score(summary["overall"])}/10 ({summary["scored"]}/{summary["selected"]})'
    )
    lines.append(
        "Categories: "
        + ", ".join(
            f'{name} {score(value["mean"])} ({value["scored"]}/{value["selected"]})'
            for name, value in summary["categories"].items()
        )
    )
    silence = summary["silence"]
    if silence["selected"]:
        lines.append(
            f'Silence case: {silence["scored"]}/{silence["selected"]} scored, mean {score(silence["mean"])}, {silence["messages"]} delivered messages'
        )
    lines.extend(f'{r["id"]}: {r["error"]}' for r in summary["tasks"] if r["error"])
    return display("\n".join(lines))


def export_csv(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "category",
        "status",
        "judge_1",
        "judge_2",
        "mean",
        "disagreement",
        "messages",
        "agent_stop",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["tasks"]:
            writer.writerow(
                dict(
                    task=row["id"],
                    category=row["category"],
                    status=row["status"],
                    judge_1=row["judge_scores"][0],
                    judge_2=row["judge_scores"][1],
                    mean=row["overall"],
                    disagreement=row["disagreement"],
                    messages=row["messages"],
                    agent_stop=row["agent_stop"],
                )
            )
    return path


def message_texts(session, ident):
    run = session.payload(ident, "agent")
    if not run or not run["message_refs"]:
        return []
    unit = session.unit(ident, "agent")
    marker = session.store.read_completion(unit)
    path = "final_state/slack/overlay.sqlite"
    record = next((e for e in marker["files"] if e["path"] == path), None)
    require(record is not None, "Saved delivered messages are missing")
    session.store.store.verify_object(
        "local",
        session.store._key(unit, PAYLOAD + "/" + path),
        record["bytes"],
        record["sha256"],
        full=True,
    )
    result = []
    with closing(
        sqlite3.connect(
            (session.bundle(unit) / path).as_uri() + "?mode=ro&immutable=1", uri=True
        )
    ) as db:
        for ref in run["message_refs"]:
            row = db.execute(
                "SELECT text FROM messages WHERE channel_id=? AND ts=?",
                (ref["channel_id"], ref["ts"]),
            ).fetchone()
            require(
                row is not None, "A delivered message is missing from the frozen state"
            )
            result.append(
                dict(
                    destination=ref.get("destination")
                    or dict(channel_id=ref["channel_id"]),
                    text=row[0],
                )
            )
    return result


def inspect_task(session, ident):
    require(ident in session.tasks, "Task is not part of this run")
    task = session.tasks[ident]
    row = next(r for r in build_summary(session)["tasks"] if r["id"] == ident)
    lines = [
        f'{ident} | {task["category"]} | {task["title"]}',
        f'Status: {row["status"]} | Mean: {score(row["overall"])}/10',
    ]
    messages = message_texts(session, ident)
    if messages:
        lines.append("\nDelivered messages:")
        for message in messages:
            lines.extend(
                [
                    json.dumps(message["destination"], ensure_ascii=False),
                    message["text"],
                    "",
                ]
            )
    else:
        lines.append("\nNo delivered messages.")
    for role in ROLES[1:]:
        judgment = session.payload(ident, role)
        lines.append(f'\n{role}: {session.settings["models"][role]["model_id"]}')
        if judgment is None:
            lines.append("Not completed.")
            continue
        lines.append(f'{judgment["status"]} | Score: {score(judgment["overall"])}')
        lines.append(judgment["reason"])
        for key, label in criterion_labels(judgment["judge"]["rubric_version"]).items():
            item = judgment["criteria"].get(key)
            if item:
                lines.append(f'  {label}: {score(item["score"])}. {item["reason"]}')
        if judgment.get("error"):
            lines.append(
                "Error: " + (judgment["error"].get("reason") or "Technical error")
            )
    if row["error"]:
        lines.append("\nError: " + row["error"])
    lines.append(f"\nSaved artifacts: {session.root/session.store.prefix/ident}")
    return display("\n".join(lines))
