import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.runtime.agent_runner import run_attempt
from benchmark.runtime.artifact_store import ArtifactStore, StorageError
from benchmark.runtime.judge_contracts import CRITERIA
from benchmark.runtime.judge_runner import run_judgment, prepare_anonymous_answer
from benchmark.scoring import combine_judgments
from helpers import fixture, ScriptedAdapter, call, response


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.args = fixture(self.root)
        self.store = ArtifactStore(self.root / "results")

    def agent(self, adapter, **changes):
        return run_attempt(
            **{**self.args, **changes},
            work_dir=self.root / "attempt",
            run_id="test",
            attempt_id="attempt-1",
            unit_key="runs/test/attempt-1",
            adapter=adapter,
            artifact_store=self.store,
        )

    def test_silence_privacy_and_resume_without_another_model_call(self):
        adapter = ScriptedAdapter([response(call("finish_task"))])
        result = self.agent(adapter)
        self.assertEqual(result["run"]["stop_reason"], "finish_task")
        self.assertEqual(result["run"]["message_refs"], [])
        visible = json.dumps(adapter.requests)
        for private in [
            "PRIVATE_NOTES_SENTINEL",
            "PRIVATE_REFERENCE_SENTINEL",
            "PRIVATE_SCENARIO_SENTINEL",
            "HIDDEN_EMPLOYEE_DM",
            "HIDDEN_OLD_AGENT_DM",
        ]:
            self.assertNotIn(private, visible)
        again = ScriptedAdapter([])
        self.assertTrue(self.agent(again)["skipped"])
        self.assertEqual(again.requests, [])

    def test_delivered_message_and_separate_finish(self):
        adapter = ScriptedAdapter(
            [
                response(
                    call(
                        "slack_post_message",
                        dict(channel_id="C_PUBLIC", text="A useful test message"),
                    )
                ),
                response(call("finish_task", call_id="finish")),
            ]
        )
        result = self.agent(adapter)
        self.assertEqual(result["run"]["outcome"], "completed")
        self.assertEqual(len(result["run"]["message_refs"]), 1)

    def test_mixed_write_and_finish_has_no_side_effect(self):
        adapter = ScriptedAdapter(
            [
                response(
                    call(
                        "slack_post_message",
                        dict(channel_id="C_PUBLIC", text="Must not be delivered"),
                    ),
                    call("finish_task", call_id="mixed-finish"),
                ),
                response(call("finish_task", call_id="finish")),
            ]
        )
        result = self.agent(adapter)
        self.assertEqual(result["run"]["message_refs"], [])
        self.assertEqual(result["run"]["stop_reason"], "finish_task")

    def test_verified_local_dataset_and_changed_source_rejection(self):
        other = self.root / "verified"
        other.mkdir()
        args = fixture(other, verified=True)
        adapter = ScriptedAdapter(
            [
                response(call("slack_list_channels")),
                response(call("finish_task", call_id="finish")),
            ]
        )
        result = run_attempt(
            **args,
            work_dir=other / "attempt",
            run_id="local",
            attempt_id="1",
            unit_key="runs/local/1",
            adapter=adapter,
            artifact_store=self.store,
        )
        self.assertEqual(
            result["run"]["stop_reason"], "finish_task", result["run"].get("error")
        )
        self.assertEqual(result["run"]["baseline_ref"]["mode"], "verified_local")
        data = other / "data/slack/users.parquet"
        data.write_bytes(data.read_bytes() + b"changed")
        rejected = run_attempt(
            **args,
            work_dir=other / "bad-attempt",
            run_id="changed",
            attempt_id="1",
            unit_key="runs/changed/1",
            adapter=ScriptedAdapter([]),
            artifact_store=self.store,
        )
        self.assertEqual(rejected["run"]["outcome"], "technical_error")

    def test_restored_result_rejects_corrupted_payload(self):
        self.agent(ScriptedAdapter([response(call("finish_task"))]))
        file = self.root / "results/benchmark/runs/test/attempt-1/_files/run.json"
        file.write_bytes(b"corrupted")
        with self.assertRaises(StorageError):
            self.store.download_directory("runs/test/attempt-1", self.root / "restored")

    def test_independent_anonymous_judges_and_compatible_mean(self):
        result = self.agent(ScriptedAdapter([response(call("finish_task"))]))
        prepared = prepare_anonymous_answer(
            package=self.args["package"],
            config_path=self.args["config_path"],
            run_bundle=result["bundle"],
        )

        def verdict(request):
            inp = json.loads(request["input"][1]["content"])
            source = next(s["id"] for s in inp["sources"] if s["kind"] == "snapshot")
            value = dict(
                status="scored",
                overall=10,
                criteria={
                    name: dict(
                        score=10 if i == 0 else None,
                        reason="Scripted plumbing check",
                        evidence=[source] if i == 0 else [],
                    )
                    for i, name in enumerate(CRITERIA)
                },
                reason="Scripted plumbing check, not an actual model judgment",
                evidence=[source],
                reference_difference=dict(
                    material=False,
                    reason="No difference",
                    evidence=[],
                    task_review_reason=None,
                ),
            )
            return response(call("submit_judgment", value))

        judgments = []
        for n in range(2):
            model = self.root / f"judge-{n}.json"
            settings = json.loads(self.args["model_path"].read_text())
            settings["model_id"] = f"independent-judge-{n}"
            settings["reasoning_effort"] = "xhigh"
            model.write_text(json.dumps(settings))
            adapter = ScriptedAdapter([verdict])
            kwargs = {**self.args, "model_path": model}
            judged = run_judgment(
                **kwargs,
                run_bundle=result["bundle"],
                work_dir=self.root / f"judge-work-{n}",
                judgment_id=f"judge-{n}",
                unit_key=f"judgments/{n}",
                adapter=adapter,
                artifact_store=self.store,
                anonymous_preparation=prepared,
            )
            self.assertEqual(
                judged["judgment"]["status"], "scored", judged["judgment"].get("error")
            )
            judgments.append(judged["judgment"])
            self.assertNotIn("arbitrary-test-model", json.dumps(adapter.requests))
        self.assertEqual(combine_judgments(*judgments)["overall"], 10)
        changed = copy.deepcopy(judgments[1])
        changed["judge"]["input_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            combine_judgments(judgments[0], changed)

    def test_inconsistent_judgment_gets_one_correction(self):
        attempt = self.agent(ScriptedAdapter([response(call("finish_task"))]))
        prepared = prepare_anonymous_answer(
            package=self.args["package"],
            config_path=self.args["config_path"],
            run_bundle=attempt["bundle"],
        )

        def verdict(overall, call_id):
            def respond(request):
                inp = json.loads(request["input"][1]["content"])
                source = next(
                    s["id"] for s in inp["sources"] if s["kind"] == "snapshot"
                )
                value = dict(
                    status="scored",
                    overall=overall,
                    criteria={
                        name: dict(
                            score=10 if i == 0 else None,
                            reason="Scripted consistency check",
                            evidence=[source] if i == 0 else [],
                        )
                        for i, name in enumerate(CRITERIA)
                    },
                    reason="Scripted check, not a model-quality result",
                    evidence=[source],
                    reference_difference=dict(
                        material=False,
                        reason="Test only",
                        evidence=[],
                        task_review_reason=None,
                    ),
                )
                return response(call("submit_judgment", value, call_id=call_id))

            return respond

        for corrected in (True, False):
            with self.subTest(corrected=corrected):
                adapter = ScriptedAdapter(
                    [
                        verdict(6, "first"),
                        verdict(10 if corrected else 6, "second"),
                        verdict(10, "third"),
                    ]
                )
                judged = run_judgment(
                    **self.args,
                    run_bundle=attempt["bundle"],
                    work_dir=self.root / f"correction-{corrected}",
                    judgment_id=f"correction-{corrected}",
                    unit_key=f"judgments/correction-{corrected}",
                    adapter=adapter,
                    artifact_store=self.store,
                    anonymous_preparation=prepared,
                )["judgment"]
                self.assertEqual(len(adapter.requests), 2)
                feedback = json.loads(adapter.requests[1]["input"][-1]["output"])
                self.assertEqual(feedback["error"], "inconsistent_scores")
                self.assertEqual(
                    judged["status"], "scored" if corrected else "technical_error"
                )
                self.assertEqual(judged["overall"], 10 if corrected else None)
                if not corrected:
                    self.assertEqual(judged["error"]["code"], "inconsistent_scores")

    def test_judge_reconstructs_verified_local_source(self):
        root = self.root / "verified-judge"
        root.mkdir()
        args = fixture(root, verified=True)
        attempt = run_attempt(
            **args,
            work_dir=root / "attempt",
            run_id="verified",
            attempt_id="1",
            unit_key="runs/verified/1",
            adapter=ScriptedAdapter([response(call("finish_task"))]),
            artifact_store=self.store,
        )
        prepared = prepare_anonymous_answer(
            package=args["package"],
            config_path=args["config_path"],
            run_bundle=attempt["bundle"],
        )

        def verdict(request):
            inp = json.loads(request["input"][1]["content"])
            source = next(s["id"] for s in inp["sources"] if s["kind"] == "snapshot")
            value = dict(
                status="scored",
                overall=10,
                criteria={
                    name: dict(
                        score=10 if i == 0 else None,
                        reason="Scripted check",
                        evidence=[source] if i == 0 else [],
                    )
                    for i, name in enumerate(CRITERIA)
                },
                reason="Scripted check",
                evidence=[source],
                reference_difference=dict(
                    material=False,
                    reason="No difference",
                    evidence=[],
                    task_review_reason=None,
                ),
            )
            return response(call("submit_judgment", value))

        judged = run_judgment(
            **args,
            run_bundle=attempt["bundle"],
            work_dir=root / "judge",
            judgment_id="local-judge",
            unit_key="judgments/local",
            adapter=ScriptedAdapter([verdict]),
            artifact_store=self.store,
            anonymous_preparation=prepared,
        )
        self.assertEqual(
            judged["judgment"]["status"], "scored", judged["judgment"].get("error")
        )


if __name__ == "__main__":
    unittest.main()
