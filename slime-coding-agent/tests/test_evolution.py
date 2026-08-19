import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from agent_core.evolution import EvolutionCoordinator, GPUInfo


class StaticProbe:
    def __init__(self, gpus):
        self.gpus = gpus

    def probe(self):
        return self.gpus


def build_config(tmpdir):
    return {
        "schedule": {"enabled": True, "start_hour": 0, "end_hour": 0},
        "data": {
            "trajectory_log": "trajectories.jsonl",
            "training_batch_dir": "training/batches",
            "state_path": "state.json",
        },
        "training": {
            "execute": False,
            "min_gpu_count": 2,
            "min_free_vram_gb_per_gpu": 20,
            "command": ["bash", "openclaw-combine/run_qwen3_4b_openclaw_combine.sh"],
        },
        "skill_fallback": {
            "max_failures_per_cycle": 10,
            "skills_dir": "skills",
            "index_path": "skills/index.json",
            "summarizer": {"mode": "heuristic"},
        },
    }


class TestEvolutionCoordinator(unittest.TestCase):
    def test_low_gpu_capacity_distills_failure_to_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = EvolutionCoordinator(
                build_config(root),
                root,
                resource_probe=StaticProbe([GPUInfo(0, "small", 8192, 6000)]),
            )
            coordinator.ingest(
                {
                    "repo_id": "demo/repo",
                    "task_id": "broken-test",
                    "final_status": "failed",
                    "command_results": [{"command": "pytest tests", "exit_code": 1, "stderr": "AssertionError"}],
                }
            )
            result = coordinator.run_cycle(now=datetime(2026, 1, 1, 1))

            self.assertEqual(result["mode"], "skill_accumulation")
            self.assertTrue(result["skills"][0]["created"])
            self.assertTrue(Path(result["skills"][0]["path"]).exists())

    def test_sufficient_gpu_prepares_combined_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = EvolutionCoordinator(
                build_config(root),
                root,
                resource_probe=StaticProbe(
                    [
                        GPUInfo(0, "gpu0", 32768, 25000),
                        GPUInfo(1, "gpu1", 32768, 25000),
                    ]
                ),
            )
            coordinator.ingest({"repo_id": "demo/repo", "task_id": "broken-test", "final_status": "failed"})
            result = coordinator.run_cycle(force=True)

            self.assertEqual(result["mode"], "binary_rl_opd_training")
            self.assertEqual(result["action"], "ready")
            self.assertTrue(Path(result["failure_batch"]).exists())

    def test_matching_failures_share_a_skill_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = EvolutionCoordinator(
                build_config(root),
                root,
                resource_probe=StaticProbe([]),
            )
            for task_id in ("task-a", "task-b"):
                coordinator.ingest(
                    {
                        "repo_id": "demo/repo",
                        "task_id": task_id,
                        "final_status": "failed",
                        "command_results": [
                            {"command": "pytest case.py", "exit_code": 1, "stderr": "ModuleNotFoundError: foo"}
                        ],
                    }
                )
            result = coordinator.run_cycle(force=True)
            self.assertEqual(result["mode"], "skill_accumulation")
            self.assertTrue(result["skills"][0]["created"])
            self.assertFalse(result["skills"][1]["created"])
            self.assertEqual(result["skills"][1]["occurrences"], 2)


if __name__ == "__main__":
    unittest.main()
