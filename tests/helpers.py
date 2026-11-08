import copy
import gzip
import json
from pathlib import Path
import shutil

from benchmark.runtime.packages import hash_file
from benchmark.runtime.responses_adapter import ModelResponse
from benchmark.runtime.slack_environment import _TOOLS

FIXTURES = Path(__file__).parent / "fixtures"


def write_manifest(package, *, config_path):
    task = json.loads((package / "task.json").read_text())
    files = [
        dict(
            path=p.relative_to(package).as_posix(),
            bytes=hash_file(p)[0],
            sha256=hash_file(p)[1],
        )
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    manifest = dict(
        schema_version=1,
        build_ref=task["meta"]["build_ref"],
        config_ref=task["meta"]["config_ref"],
        files=files,
    )
    (package / "manifest.json").write_text(json.dumps(manifest))


def call(name, arguments=None, call_id="call-1"):
    return dict(
        type="function_call",
        id="fc-" + call_id,
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments or {}),
        status="completed",
    )


def response(*items):
    return dict(
        id="response-test",
        status="completed",
        output=list(items),
        usage=dict(input_tokens=12, output_tokens=8),
    )


class ScriptedAdapter:
    route_id = "scripted-offline"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def respond(self, request, *, timeout_seconds):
        self.requests.append(copy.deepcopy(request))
        value = self.responses.pop(0)
        return ModelResponse(
            value(request) if callable(value) else value, "scripted-request"
        )


def fixture(root, *, verified=False):
    root = Path(root)
    baseline = root / "data"
    shutil.copytree(FIXTURES / "slack", baseline / "slack")
    config = json.loads((FIXTURES / "config.json").read_text())
    config["agent_instructions"] = (
        "Inspect recent activity and finish explicitly. {service_guide}"
    )
    config["tool_scope"] = list(_TOOLS)
    config["limits"].update(
        agent_model_seconds=180,
        tool_timeout_seconds=600,
        environment_setup_timeout_seconds=600,
    )
    config["storage"] = {}
    if verified:
        rows = [
            dict(
                path=str(p.relative_to(baseline)),
                size_bytes=p.stat().st_size,
                sha256=hash_file(p)[1],
            )
            for p in sorted((baseline / "slack").rglob("*"))
            if p.is_file()
        ]
        with gzip.open(baseline / "file-index.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        manifest = dict(
            source_snapshot_sha256=config["build_ref"]["manifest_sha256"],
            file_count=len(rows),
            file_index=dict(
                path="file-index.jsonl.gz",
                sha256=hash_file(baseline / "file-index.jsonl.gz")[1],
            ),
        )
        (baseline / "manifest.json").write_text(json.dumps(manifest))
        config["storage"] = {
            "baseline": dict(
                manifest_sha256=hash_file(baseline / "manifest.json")[1],
                cache_dir=str(root / "cache"),
                max_cache_bytes=64 * 1024**2,
                max_file_bytes=32 * 1024**2,
                min_free_bytes=0,
                lock_wait_seconds=10,
            )
        }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    task = json.loads((FIXTURES / "task.json").read_text())
    task["meta"]["config_ref"]["sha256"] = hash_file(config_path)[1]
    package = root / "package"
    (package / "delta").mkdir(parents=True)
    (package / "task.json").write_text(json.dumps(task))
    write_manifest(package, config_path=config_path)
    model = root / "model.json"
    model.write_text(
        json.dumps(
            dict(
                schema_version=1,
                provider="scripted",
                model_id="arbitrary-test-model",
                reasoning_effort="high",
                max_output_tokens=16384,
            )
        )
    )
    return dict(
        package=package,
        config_path=config_path,
        model_path=model,
        baseline_dir=baseline if verified else baseline / "slack",
        baseline_build_ref=None if verified else config["build_ref"],
        baseline_mode="verified_local" if verified else "local_declared",
    )
