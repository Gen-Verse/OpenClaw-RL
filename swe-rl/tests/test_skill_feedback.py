import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SWE_RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SWE_RL))

import skill_feedback


class TestSkillFeedback(unittest.TestCase):
    def test_retrieval_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            message, skills = skill_feedback.augment_instance_message("issue text", "issue text")
        self.assertEqual(message, "issue text")
        self.assertEqual(skills, [])

    def test_retrieval_injects_matching_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "pytest-fix.md").write_text(
                "# Pytest recovery\nRun the failing pytest target before editing.\n",
                encoding="utf-8",
            )
            env = {
                "SWE_SKILL_RETRIEVAL": "1",
                "SWE_SKILLS_DIR": str(skill_dir),
                "SWE_SKILL_TOP_K": "2",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                message, skills = skill_feedback.augment_instance_message(
                    "base", "pytest target fails with assertion"
                )
        self.assertIn("Retrieved Recovery Skills", message)
        self.assertEqual(len(skills), 1)

    def test_event_schema_and_failure_evidence(self):
        event = skill_feedback.build_swe_event(
            instance={
                "repo": "astropy/astropy",
                "instance_id": "astropy__astropy-12907",
                "base_commit": "abc",
            },
            data_source="swe-gym",
            resolved=False,
            run_info={
                "error": "eval timeout",
                "exit_status": "submitted",
                "policy": {"reasons": ["eval_test_file_modified"]},
                "eval_result": {"grading_error": "", "resolved_by": "harness"},
                "retrieved_skills": [{"path": "x", "score": 0.1}],
            },
        )
        for key in ("repo_id", "task_id", "final_status", "command_results"):
            self.assertIn(key, event)
        self.assertEqual(event["final_status"], "failed")
        self.assertIn("eval timeout", event["command_results"][0]["stderr"])

    def test_publish_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "traj.jsonl"
            with mock.patch.dict(os.environ, {"SWE_TRAJECTORY_LOG": str(log_path)}, clear=True):
                asyncio.run(
                    skill_feedback.publish_swe_trajectory(
                        instance={"repo": "r", "instance_id": "i"},
                        data_source="swe-gym",
                        resolved=True,
                        run_info={"n_steps": 3},
                    )
                )
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(events[0]["final_status"], "success")


if __name__ == "__main__":
    unittest.main()
