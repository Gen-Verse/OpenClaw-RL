import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import wcb_to_rl_dataset as conv
import make_split


class StubTokenizer:
    """Deterministic word-level tokenizer stub for converter tests."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        msg = messages[0]
        words = f"{msg.get('role','?')}:{msg.get('content','')}".split()
        return [hash(w) % 10000 for w in words] or [0]


def make_run(root: Path, task: str, run: str, score: float) -> Path:
    run_dir = root / "openclaw" / "01_Productivity_Flow" / task / run
    run_dir.mkdir(parents=True)
    (run_dir / "chat.jsonl").write_text(
        json.dumps({"role": "user", "content": "do it"}) + "\n"
        + json.dumps({"role": "assistant", "content": "done now"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "score.json").write_text(json.dumps({"overall_score": score}), encoding="utf-8")
    return run_dir


class TestSplit(unittest.TestCase):
    def test_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cat in ["01_Productivity_Flow", "03_Social_Interaction"]:
                d = root / "tasks" / cat
                d.mkdir(parents=True)
                for i in range(5):
                    (d / f"{cat}_task_{i}.md").write_text("x", encoding="utf-8")
            out1 = root / "s1.json"
            out2 = root / "s2.json"
            for out in (out1, out2):
                sys.argv = ["make_split.py", "--wcb-root", str(root), "--output", str(out)]
                make_split.main()
            a = json.loads(out1.read_text())
            b = json.loads(out2.read_text())
            self.assertEqual(a["train"], b["train"])
            self.assertFalse(set(a["train"]) & set(a["eval"]))


class TestConverter(unittest.TestCase):
    def test_grouping_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            make_run(raw, "task_a", "r1", 1.0)
            make_run(raw, "task_a", "r2", 0.0)
            make_run(raw, "task_b", "r1", 0.5)

            samples = []
            groups = {}
            tok = StubTokenizer()
            for run_dir, chat_path, score in conv.iter_runs(raw):
                messages = [json.loads(x) for x in chat_path.read_text().splitlines() if x.strip()]
                tokens, masks = conv.encode_messages(tok, messages)
                prompt_len, response_mask = conv.split_prompt_response(tokens, masks)
                group = groups.setdefault(run_dir.parent.name, len(groups))
                samples.append((prompt_len, response_mask, group, score))

            self.assertEqual(len(samples), 3)
            # task_a 两条进同一组（GRPO 需要组内 >1）
            self.assertEqual(samples[0][2], samples[1][2])
            self.assertNotEqual(samples[0][2], samples[2][2])
            # assistant 段 mask 全 1
            for prompt_len, response_mask, _, _ in samples:
                self.assertTrue(all(response_mask))
                self.assertGreater(prompt_len, 0)


if __name__ == "__main__":
    unittest.main()
