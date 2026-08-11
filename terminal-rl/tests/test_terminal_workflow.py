import sys
import tempfile
import unittest
from pathlib import Path


TERMINAL_RL = Path(__file__).resolve().parents[1]
DATA_UTILS = TERMINAL_RL / "data_utils"
sys.path[:0] = [str(TERMINAL_RL), str(DATA_UTILS)]

from prepare_terminal_splits import stable_split
from skill_context import retrieve_skills
from terminal_report import summarize


class TestTerminalWorkflow(unittest.TestCase):
    def test_stable_split_is_disjoint_and_repeatable(self):
        names = [f"task-{index}" for index in range(10)]
        train_a, heldout_a = stable_split(names, 0.2, "seed")
        train_b, heldout_b = stable_split(names, 0.2, "seed")
        self.assertEqual((train_a, heldout_a), (train_b, heldout_b))
        self.assertFalse(set(train_a) & set(heldout_a))
        self.assertEqual(set(train_a) | set(heldout_a), set(names))

    def test_skill_retrieval_and_task_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "pytest-import.md").write_text(
                "# Import failure\nRun pytest after installing the missing dependency.\n",
                encoding="utf-8",
            )
            skills = retrieve_skills("pytest fails with missing dependency", skill_dir)
        self.assertEqual(len(skills), 1)

        report = summarize(
            [
                {"task_id": "a", "attempt_id": "0", "success": False, "steps": 2},
                {"task_id": "a", "attempt_id": "1", "success": True, "steps": 3},
                {"task_id": "b", "attempt_id": "0", "success": True, "steps": 1, "skill_retrieval": [{"path": "x"}]},
            ],
            pass_at_k=2,
        )
        self.assertEqual(report["pass_at_1"], 0.5)
        self.assertEqual(report["pass_at_2"], 1.0)
        self.assertEqual(report["resolve_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
