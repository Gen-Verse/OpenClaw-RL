import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolve import grouper, judge
from skill_evolve.evolver import evolve_group
from skill_evolve.sessions import build_session
from skill_evolve.store import EvolvingSkillStore
from skill_evolve.verifier import verify


def make_run_dir(root: Path, score: float) -> Path:
    run_dir = root / "openclaw" / "01_Productivity_Flow" / "task_1" / "run_abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "chat.jsonl").write_text(
        json.dumps({"role": "user", "content": "do the task"}) + "\n"
        + json.dumps({"role": "assistant", "content": "try", "tool_calls": [
            {"function": {"name": "bash", "arguments": "ls /root/skills/pdf-help/SKILL.md"}}
        ]}) + "\n"
        + json.dumps({"role": "tool", "content": "No such file"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "score.json").write_text(json.dumps({"overall_score": score}), encoding="utf-8")
    return run_dir


class TestSkillEvolve(unittest.TestCase):
    def test_session_building_and_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = make_run_dir(root, 0.5)
            skills_root = root / "skills" / "pdf-help"
            skills_root.mkdir(parents=True)
            (skills_root / "SKILL.md").write_text("# pdf help", encoding="utf-8")

            session = build_session(run_dir, ["pdf-help"])
            self.assertEqual(session["score"], 0.5)
            self.assertIn("pdf-help", session["skills_referenced"])

            groups = grouper.group_sessions([session, {"session_id": "x", "skills_referenced": set()}])
            self.assertIn("pdf-help", groups)
            self.assertIn(grouper.NO_SKILL, groups)

    def test_judge_skips_scored_sessions(self):
        session = {"score": 0.7, "trajectory": "x"}
        with mock.patch.dict(os.environ, {"SKILL_EVOLVE_JUDGE": "1"}, clear=True):
            self.assertEqual(judge.maybe_judge(session), 0.7)

    def test_heuristic_create_requires_two_failures(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            one_fail = [{"score": 0.0, "task_id": "t1", "trajectory": "err"}]
            self.assertIsNone(evolve_group("g", one_fail, None, []))
            two_fail = one_fail + [{"score": 0.2, "task_id": "t2", "trajectory": "err2"}]
            result = evolve_group("g", two_fail, None, [])
            self.assertEqual(result["action"], "create")

    def test_store_version_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvolvingSkillStore(Path(tmp) / "skills")
            store.publish("alpha", "# v1", "ev1")
            store.publish("alpha", "# v2", "ev2")
            self.assertEqual(store.current("alpha"), "# v2\n")
            history = store.history("alpha")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["version"], "v1")

    def test_verifier_heuristic(self):
        candidate = {"skill_md": "run pytest case.py then fix import", "evidence": "pytest fails"}
        sessions = [{"trajectory": "pytest case.py failed with import error"}]
        with mock.patch.dict(os.environ, {}, clear=True):
            gate = verify(candidate, sessions)
        self.assertTrue(gate["accepted"])


if __name__ == "__main__":
    unittest.main()
