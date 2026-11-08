import asyncio
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark.configuration import load_project, init_config, task_catalog
from benchmark.execution import Session, start_run, resume_run, run_lock
from benchmark.reporting import build_summary, inspect_task, export_csv
from benchmark.runtime.artifact_store import ArtifactStore, StorageError
from benchmark.runtime.judge_contracts import CRITERIA
from benchmark.runtime.packages import hash_file
from benchmark.runtime.responses_adapter import ModelResponse
from helpers import fixture, write_manifest, call, response


def project_fixture(root, count=1):
    original = root / "fixture"
    original.mkdir()
    args = fixture(original, verified=True)
    tasks_dir = root / "tasks"
    tasks = []
    for number in range(1, count + 1):
        ident = f"task-{number:02d}"
        folder = tasks_dir / ident
        shutil.copytree(args["package"], folder / "package")
        shutil.copyfile(args["config_path"], folder / "config.json")
        task = json.loads((folder / "package/task.json").read_text())
        task["meta"]["task_id"] = task["brief"]["task_id"] = ident
        (folder / "package/task.json").write_text(json.dumps(task))
        write_manifest(folder / "package", config_path=folder / "config.json")
        tasks.append(
            dict(
                id=ident,
                category="Analyze",
                title="Handled release check",
                version=1,
                package=f"{ident}/package",
                config=f"{ident}/config.json",
                manifest_sha256=hash_file(folder / "package/manifest.json")[1],
            )
        )
    (tasks_dir / "index.json").write_text(
        json.dumps(dict(schema_version=1, tasks=tasks))
    )
    config = root / "benchmark.json"
    init_config(config)
    value = json.loads(config.read_text())
    value["data_dir"] = str(args["baseline_dir"])
    for name in ["cache_dir", "results_dir", "tasks_dir"]:
        value[name] = str(root / name.replace("_dir", ""))
    for model, name in zip(
        [value["agent"], *value["judges"]],
        ["offline-agent", "offline-judge-one", "offline-judge-two"],
    ):
        model.update(provider="scripted", model_id=name, api_key_env=None)
    config.write_text(json.dumps(value))
    return load_project(config), config


def verdict(request, value):
    inp = json.loads(request["input"][1]["content"])
    source = next(s["id"] for s in inp["sources"] if s["kind"] == "snapshot")
    return response(
        call(
            "submit_judgment",
            dict(
                status="scored",
                overall=value,
                criteria={
                    name: dict(
                        score=value if i == 0 or inp["delivered_messages"] else None,
                        reason="Offline CLI check",
                        evidence=(
                            [source] if i == 0 or inp["delivered_messages"] else []
                        ),
                    )
                    for i, name in enumerate(CRITERIA)
                },
                reason="Offline CLI check, not model-quality evidence",
                evidence=[source],
                reference_difference=dict(
                    material=False,
                    reason="Test only",
                    evidence=[],
                    task_review_reason=None,
                ),
            ),
        )
    )


class OfflineFactory:
    def __init__(self):
        self.calls = []
        self.behavior = {}
        self.closed = []

    def __call__(self, model):
        factory = self

        class Adapter:
            route_id = (
                "offline-"
                + hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                factory.closed.append(model["model_id"])

            def check(self):
                pass

            async def respond(self, request, *, timeout_seconds):
                factory.calls.append((model["model_id"], copy.deepcopy(request)))
                queue = factory.behavior.get(model["model_id"], [])
                behavior = queue.pop(0) if queue else None
                if behavior == "cancel-task":
                    asyncio.current_task().cancel()
                    await asyncio.sleep(0)
                if behavior == "cancel":
                    raise asyncio.CancelledError
                if behavior == "wait":
                    await asyncio.sleep(10)
                if behavior == "error":
                    return ModelResponse(
                        dict(
                            status="failed",
                            output=[],
                            error=dict(
                                code="offline_http_error", message="Mock API failure"
                            ),
                        ),
                        http_status=503,
                    )
                if callable(behavior):
                    body = behavior(request)
                elif model["model_id"] == "offline-agent":
                    body = response(call("finish_task"))
                else:
                    body = verdict(
                        request, 0 if model["model_id"] == "offline-judge-one" else 10
                    )
                return ModelResponse(body, "offline-request")

        return Adapter()

    def counts(self):
        return Counter(name for name, _ in self.calls)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.project, self.config = project_fixture(self.root)
        self.factory = OfflineFactory()
        self.check = patch(
            "benchmark.execution.check_inputs",
            return_value=dict(manifest_sha256="test-dataset"),
        )
        self.check.start()
        self.addCleanup(self.check.stop)
        self.options = dict(factory=self.factory, progress=lambda text: None)

    def run_new(self, **changes):
        return asyncio.run(
            start_run(self.project, run_id="cli-test", **{**self.options, **changes})
        )

    def test_model_check_reads_project_dotenv_and_keeps_exported_keys(self):
        profile = self.root / "model.json"
        profile.write_text(
            json.dumps(
                dict(
                    provider="responses",
                    model_id="docs-model",
                    endpoint="https://model.invalid/v1/responses",
                    max_output_tokens=1024,
                    api_key_env="DOCS_MODEL_KEY",
                )
            )
        )
        folder = self.root / "settings"
        folder.mkdir()
        cases = [
            (self.config, self.root, "DOCS_MODEL_KEY=from-file\n", None),
            (
                folder / "custom.json",
                folder,
                "DOCS_MODEL_KEY=from-custom-file\n",
                None,
            ),
            (
                self.config,
                self.root,
                'DOCS_MODEL_KEY="invalid\nheader"\n',
                "from-shell",
            ),
        ]
        for config, dotenv_folder, content, exported in cases:
            with self.subTest(config=config.name, exported=bool(exported)):
                for dotenv_file in [self.root / ".env", folder / ".env"]:
                    dotenv_file.unlink(missing_ok=True)
                (dotenv_folder / ".env").write_text(content)
                environment = dict(os.environ)
                environment.pop("DOCS_MODEL_KEY", None)
                if exported:
                    environment["DOCS_MODEL_KEY"] = exported
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmark",
                        "--config",
                        str(config),
                        "model",
                        "check",
                        str(profile),
                    ],
                    cwd=self.root,
                    env=environment,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Valid model profile: docs-model", result.stdout)
                self.assertIn("No API call made.", result.stdout)

    def resume(self, session, **changes):
        return asyncio.run(
            resume_run(Session(session.root), **{**self.options, **changes})
        )

    def test_complete_pair_zero_score_and_offline_resume(self):
        session = self.run_new()
        summary = build_summary(session)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["tasks"][0]["judge_scores"], [0, 10])
        self.assertEqual(summary["overall"], 5)
        self.assertEqual(summary["silence"]["messages"], 0)
        self.assertEqual(
            self.factory.counts(),
            {"offline-agent": 1, "offline-judge-one": 1, "offline-judge-two": 1},
        )
        with patch(
            "benchmark.execution.check_inputs",
            side_effect=AssertionError("No data needed"),
        ):
            self.resume(session)
        self.assertEqual(len(self.factory.calls), 3)
        self.assertFalse(list((session.root / ".work").glob("*/bundle")))
        self.assertNotIn("PRIVATE_NOTES_SENTINEL", json.dumps(self.factory.calls[0][1]))
        for name, request in self.factory.calls[1:]:
            self.assertNotIn("offline-agent", json.dumps(request))
        self.assertIn(
            "Public example set",
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark",
                    "results",
                    "--results-dir",
                    str(session.root.parent),
                    "--run",
                    "cli-test",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout,
        )

    def test_one_failed_judge_retries_only_that_judge(self):
        self.factory.behavior["offline-judge-two"] = ["error"]
        session = self.run_new()
        summary = build_summary(session)
        self.assertEqual(summary["tasks"][0]["status"], "judge_error")
        self.assertIsNone(summary["overall"])
        self.resume(session)
        self.assertEqual(len(self.factory.calls), 3)
        session = self.resume(session, retry_errors=True)
        self.assertEqual(
            self.factory.counts(),
            {"offline-agent": 1, "offline-judge-one": 1, "offline-judge-two": 2},
        )
        self.assertEqual(build_summary(session)["overall"], 5)
        self.assertTrue(session.bundle("task-01/judge-2-001").is_dir())
        self.assertEqual(len(session.progress["tasks"]["task-01"]["judge-2"]), 2)

    def test_verified_model_time_budget_stop_is_scored_and_kept(self):
        folder = self.project["tasks_dir"] / "task-01"
        config = json.loads((folder / "config.json").read_text())
        config["limits"]["agent_model_seconds"] = 0.03
        (folder / "config.json").write_text(json.dumps(config))
        task = json.loads((folder / "package/task.json").read_text())
        task["meta"]["config_ref"]["sha256"] = hash_file(folder / "config.json")[1]
        (folder / "package/task.json").write_text(json.dumps(task))
        write_manifest(folder / "package", config_path=folder / "config.json")
        index = json.loads((self.project["tasks_dir"] / "index.json").read_text())
        index["tasks"][0]["manifest_sha256"] = hash_file(
            folder / "package/manifest.json"
        )[1]
        (self.project["tasks_dir"] / "index.json").write_text(json.dumps(index))
        self.factory.behavior["offline-agent"] = ["wait"]
        session = self.run_new()
        self.assertEqual(
            session.payload("task-01", "agent")["stop_reason"], "model_timeout"
        )
        self.assertEqual(build_summary(session)["scored"], 1)
        self.resume(session, retry_errors=True)
        self.assertEqual(len(self.factory.calls), 3)

    def test_task_fault_is_distinct_and_never_becomes_a_zero(self):
        from benchmark.runtime.judge_contracts import error_result

        def fault(request):
            inp = json.loads(request["input"][1]["content"])
            source = next(s["id"] for s in inp["sources"] if s["kind"] == "snapshot")
            value = error_result("Offline fixture-defect check")
            value.update(status="task_error", evidence=[source])
            return response(call("submit_judgment", value))

        self.factory.behavior["offline-judge-two"] = [fault]
        session = self.run_new()
        summary = build_summary(session)
        self.assertEqual(summary["tasks"][0]["status"], "task_error")
        self.assertIsNone(summary["overall"])
        self.resume(session, retry_errors=True)
        self.assertEqual(len(self.factory.calls), 3)

    def test_agent_failure_is_not_scored_or_retried_implicitly(self):
        self.factory.behavior["offline-agent"] = ["error"]
        session = self.run_new()
        self.assertEqual(build_summary(session)["tasks"][0]["status"], "agent_error")
        self.assertEqual(len(self.factory.calls), 1)
        self.resume(session)
        self.assertEqual(len(self.factory.calls), 1)
        session = self.resume(session, retry_errors=True)
        self.assertEqual(build_summary(session)["scored"], 1)
        self.assertEqual(self.factory.counts()["offline-agent"], 2)

    def test_cancellation_stops_run_and_preserves_trace_for_resume(self):
        self.factory.behavior["offline-agent"] = ["cancel-task"]
        with self.assertRaises(asyncio.CancelledError):
            self.run_new()
        session = Session(self.project["results_dir"] / "cli-test")
        self.assertEqual(build_summary(session)["tasks"][0]["status"], "interrupted")
        self.assertTrue((session.bundle("task-01/agent-001") / "trace.jsonl").is_file())
        session = self.resume(session)
        self.assertEqual(build_summary(session)["scored"], 1)
        self.assertEqual(self.factory.counts()["offline-agent"], 2)
        self.assertTrue(session.bundle("task-01/agent-001").exists())

    def test_resume_checks_credentials_only_for_missing_model_work(self):
        self.factory.behavior["offline-judge-two"] = ["cancel"]
        session = self.run_new()
        checked = []

        def factory(model):
            adapter = self.factory(model)

            def check():
                checked.append(model["model_id"])
                if model["model_id"] != "offline-judge-two":
                    raise AssertionError("Saved model credentials must not be required")

            adapter.check = check
            return adapter

        session = self.resume(session, factory=factory)
        self.assertEqual(checked, ["offline-judge-two"])
        self.assertEqual(build_summary(session)["scored"], 1)
        self.assertEqual(self.factory.counts()["offline-agent"], 1)

    def test_interrupted_second_judge_keeps_completed_agent_and_first_judge(self):
        self.factory.behavior["offline-judge-two"] = ["cancel"]
        session = self.run_new()
        self.assertEqual(build_summary(session)["tasks"][0]["status"], "interrupted")
        session = self.resume(session)
        self.assertEqual(build_summary(session)["overall"], 5)
        self.assertEqual(
            self.factory.counts(),
            {"offline-agent": 1, "offline-judge-one": 1, "offline-judge-two": 2},
        )

    def test_lost_in_progress_attempt_is_preserved(self):
        from benchmark.configuration import check_models

        tasks = task_catalog(self.project["tasks_dir"])
        routes = asyncio.run(check_models(self.project["models"], factory=self.factory))
        session = Session.create(self.project, tasks, routes, dict(), run_id="cli-test")
        unit = session.start("task-01", "agent")
        work = session.work(unit) / "bundle"
        work.mkdir(parents=True)
        (work / "trace.jsonl").write_text("unfinished request")
        session = self.resume(session)
        self.assertEqual((work / "trace.jsonl").read_text(), "unfinished request")
        self.assertEqual(len(session.progress["tasks"]["task-01"]["agent"]), 2)
        self.assertEqual(build_summary(session)["scored"], 1)

    def test_publication_recovery_does_not_repeat_any_api_call(self):
        real = ArtifactStore.publish_directory

        def publish(store, bundle, unit, metadata=None):
            if unit == "task-01/judge-2-001":
                path = Path(bundle) / "judgment.json"
                store.store.put_file_immutable(
                    "local",
                    store._key(unit, "_files/judgment.json"),
                    path,
                    hash_file(path)[1],
                )
                raise OSError("Interrupted publication")
            return real(store, bundle, unit, metadata)

        with patch.object(ArtifactStore, "publish_directory", publish):
            with self.assertRaises(Exception):
                self.run_new()
        session = Session(self.project["results_dir"] / "cli-test")
        self.assertEqual(len(self.factory.calls), 3)
        with patch(
            "benchmark.execution.check_inputs",
            side_effect=AssertionError("No model or data check should repeat"),
        ):
            session = self.resume(session)
        self.assertEqual(len(self.factory.calls), 3)
        self.assertEqual(build_summary(session)["scored"], 1)

    def test_runtime_task_and_adapter_changes_stop_before_calls(self):
        self.factory.behavior["offline-judge-two"] = ["cancel"]
        session = self.run_new()
        before = len(self.factory.calls)
        with patch("benchmark.execution.fingerprint", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Runtime changed"):
                self.resume(session)
        config = self.project["tasks_dir"] / "task-01/config.json"
        config.write_text(config.read_text() + " ")
        with self.assertRaises(ValueError):
            self.resume(session)
        self.assertEqual(len(self.factory.calls), before)

    def test_corrupted_result_is_not_accepted(self):
        session = self.run_new()
        path = session.bundle("task-01/judge-1-001") / "judgment.json"
        path.write_text("{}")
        with self.assertRaises(StorageError):
            build_summary(session)
        with self.assertRaises(StorageError):
            self.resume(session)

    def test_viewing_results_does_not_modify_saved_files(self):
        session = self.run_new()
        before = {
            str(path): path.stat().st_mtime_ns
            for path in session.root.rglob("*")
            if path.is_file()
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark",
                "results",
                "--results-dir",
                str(session.root.parent),
                "--run",
                "cli-test",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["scored"], 1)
        after = {
            str(path): path.stat().st_mtime_ns
            for path in session.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_completed_resume_checks_saved_trace_without_data_or_keys(self):
        session = self.run_new()
        (session.bundle("task-01/agent-001") / "trace.jsonl").write_text(
            "changed trace"
        )
        with self.assertRaises(StorageError):
            self.resume(session)
        self.assertEqual(len(self.factory.calls), 3)

    def test_overlapping_cache_and_results_are_rejected_before_a_run(self):
        config = json.loads(self.config.read_text())
        config["results_dir"] = config["cache_dir"] + "/results"
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "separate folders"):
            load_project(self.config)
        self.assertFalse(self.factory.calls)

    def test_saved_trace_is_checked_before_new_model_work(self):
        self.factory.behavior["offline-judge-two"] = ["cancel"]
        session = self.run_new()
        (session.bundle("task-01/agent-001") / "trace.jsonl").write_text(
            "changed trace"
        )
        before = len(self.factory.calls)
        with self.assertRaises(StorageError):
            self.resume(session)
        self.assertEqual(len(self.factory.calls), before)

    def test_inspect_messages_and_csv_need_no_source_data(self):
        self.factory.behavior["offline-agent"] = [
            lambda req: response(
                call(
                    "slack_post_message",
                    dict(channel_id="C_PUBLIC", text="A test message"),
                )
            ),
            lambda req: response(call("finish_task", call_id="finish")),
        ]
        session = self.run_new()
        shutil.rmtree(self.project["data_dir"])
        shutil.rmtree(self.project["tasks_dir"])
        text = inspect_task(Session(session.root), "task-01")
        self.assertIn("A test message", text)
        self.assertIn("Offline CLI check", text)
        path = export_csv(build_summary(session), self.root / "table.csv")
        self.assertIn("task-01,Analyze,scored,0,10,5", path.read_text())
        self.resume(session)
        self.assertEqual(self.factory.counts()["offline-agent"], 2)

    def test_selected_task_and_full_set(self):
        shutil.rmtree(self.project["tasks_dir"])
        # A separate fixture supplies several tasks sharing the same checked data.
        folder = self.root / "multi"
        folder.mkdir()
        project, _ = project_fixture(folder, 3)
        selected = asyncio.run(
            start_run(project, "task-02", run_id="one", **self.options)
        )
        self.assertEqual(list(selected.tasks), ["task-02"])
        session = asyncio.run(start_run(project, "all", run_id="all", **self.options))
        self.assertEqual(build_summary(session)["scored"], 3)

    def test_run_lock_and_existing_id_are_clear(self):
        session = self.run_new()
        with run_lock(session.root):
            with self.assertRaisesRegex(ValueError, "already active"):
                with run_lock(session.root):
                    pass
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_new()

    def test_credentials_are_rejected_before_any_settings_file_is_written(self):
        from benchmark.configuration import check_models

        secret = "unique-private-model-key-123456789"
        self.project["models"]["agent"]["parameters"]["metadata"] = secret
        with patch.dict(os.environ, {"TEST_API_KEY": secret}):
            routes = asyncio.run(
                check_models(self.project["models"], factory=self.factory)
            )
            with self.assertRaisesRegex(ValueError, "credential"):
                Session.create(
                    self.project,
                    task_catalog(self.project["tasks_dir"]),
                    routes,
                    dict(),
                    run_id="secret",
                )
        self.assertFalse((self.project["results_dir"] / "secret").exists())

    def test_cli_init_check_run_results_and_inspect(self):
        from benchmark.__main__ import main

        output = io.StringIO()
        with patch("benchmark.configuration.check_inputs", return_value=dict()), patch(
            "benchmark.runtime.provider_adapters.create_adapter", self.factory
        ), patch.object(
            sys, "argv", ["benchmark", "check", "--config", str(self.config)]
        ), redirect_stdout(
            output
        ), redirect_stderr(
            io.StringIO()
        ):
            main()
        self.assertIn("No API call made", output.getvalue())
        self.assertFalse(self.factory.calls)
        with patch(
            "benchmark.runtime.provider_adapters.create_adapter", self.factory
        ), patch.object(
            sys,
            "argv",
            ["benchmark", "run", "--config", str(self.config), "--run", "command"],
        ), redirect_stdout(
            io.StringIO()
        ), redirect_stderr(
            io.StringIO()
        ):
            main()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark",
                "results",
                "--config",
                str(self.config),
                "--run",
                "command",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["scored"], 1)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark",
                "inspect",
                "--config",
                str(self.config),
                "--run",
                "command",
                "--task",
                "task-01",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Score: 0.00", result.stdout)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark",
                "init",
                "--config",
                str(self.root / "new.json"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Created", result.stdout)
        with self.assertRaises(ValueError):
            load_project(self.root / "new.json")


if __name__ == "__main__":
    unittest.main()
