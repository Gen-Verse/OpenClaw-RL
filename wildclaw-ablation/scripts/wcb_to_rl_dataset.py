#!/usr/bin/env python3
"""Convert WildClawBench trajectory outputs into slime offline-RL .pt data.

Reads results/<collect>/raw/**/{chat.jsonl,score.json} and writes one
torch file compatible with ``--load-debug-rollout-data``:
  {"rollout_id": 0, "samples": [Sample.to_dict(), ...]}

Each task forms one GRPO group (``group_index``), so run the collect phase
with ROLLOUTS_PER_TASK > 1 to get non-degenerate group-relative advantages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _torch():
    import torch
    return torch


def encode_messages(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Per-message chat-template encode. Returns (tokens, assistant_mask)."""
    tokens: list[int] = []
    masks: list[int] = []
    for msg in messages:
        role = msg.get("role", "user")
        toks = tokenizer.apply_chat_template([msg], tokenize=True, add_generation_prompt=False)
        tokens.extend(toks)
        masks.extend([1 if role == "assistant" else 0] * len(toks))
    return tokens, masks


def split_prompt_response(tokens: list[int], masks: list[int]) -> tuple[int, list[int]]:
    """First assistant token marks the response start. Returns (prompt_len, response_mask)."""
    try:
        first_asst = masks.index(1)
    except ValueError:
        return len(tokens), []
    return first_asst, masks[first_asst:]


def iter_runs(raw_dir: Path):
    for chat_path in sorted(raw_dir.rglob("chat.jsonl")):
        run_dir = chat_path.parent
        score_path = run_dir / "score.json"
        if not score_path.is_file():
            continue
        try:
            score = json.loads(score_path.read_text(encoding="utf-8")).get("overall_score")
        except json.JSONDecodeError:
            continue
        if score is None:
            continue
        yield run_dir, chat_path, float(score)


def compute_response_logprobs(
    model_path: str,
    samples: list[dict[str, Any]],
    device: str,
    chunk: int = 2048,
) -> None:
    """Fill rollout_log_probs in-place via HF forward (0.6B-friendly, chunked lm_head)."""
    from transformers import AutoModelForCausalLM
    torch = _torch()

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    backbone = model.model
    lm_head = model.lm_head

    with torch.no_grad():
        for sample in samples:
            tokens = sample["tokens"]
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            hidden = backbone(input_ids=ids).last_hidden_state[0]  # [seq, d]

            logps = torch.zeros(len(tokens) - 1, dtype=torch.float32)
            for start in range(0, len(tokens) - 1, chunk):
                end = min(start + chunk, len(tokens) - 1)
                logits = lm_head(hidden[start:end]).float()
                logp = torch.log_softmax(logits, dim=-1)
                targets = ids[0, start + 1 : end + 1]
                logps[start:end] = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu()

            prompt_len = len(tokens) - sample["response_length"]
            # response token j (abs index prompt_len + j) is predicted at position prompt_len + j - 1
            sample["rollout_log_probs"] = logps[prompt_len - 1 :].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--model", required=True, help="HF model path for tokenizer + logprob scoring")
    parser.add_argument("--output", required=True, help="output .pt path (use {rollout_id} template or plain path)")
    parser.add_argument("--reward-mode", choices=["raw", "centered"], default="raw")
    parser.add_argument("--skip-logprobs", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples: list[dict[str, Any]] = []
    group_of_task: dict[str, int] = {}

    for run_dir, chat_path, score in iter_runs(Path(args.raw_dir)):
        messages = []
        for line in chat_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("role") in {"system", "user", "assistant", "tool"}:
                messages.append(msg)
        if not messages:
            continue

        tokens, masks = encode_messages(tokenizer, messages)
        prompt_len, response_mask = split_prompt_response(tokens, masks)
        if not response_mask or not any(response_mask):
            continue

        task_id = run_dir.parent.name
        group_index = group_of_task.setdefault(task_id, len(group_of_task))
        reward = 2.0 * score - 1.0 if args.reward_mode == "centered" else score

        samples.append(
            {
                "group_index": group_index,
                "index": len(samples),
                "prompt": "",
                "tokens": tokens,
                "response": "",
                "response_length": len(tokens) - prompt_len,
                "reward": {"score": reward},
                "loss_mask": response_mask,
                "status": "completed",
                "metadata": {"task_id": task_id, "run_dir": str(run_dir), "overall_score": score},
            }
        )

    if not samples:
        raise SystemExit("no usable samples found")

    if not args.skip_logprobs:
        torch = _torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_response_logprobs(args.model, samples, device)

    out_path = Path(args.output.format(rollout_id=0)) if "{rollout_id}" in args.output else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    torch.save({"rollout_id": 0, "samples": samples}, out_path)

    groups = len(group_of_task)
    print(f"[convert] samples={len(samples)} groups={groups} -> {out_path}")
    if groups and len(samples) == groups:
        print("[convert] WARNING: 1 sample per group; GRPO group-relative advantage degenerates. "
              "Collect with ROLLOUTS_PER_TASK>1 or use --disable-rewards-normalization semantics.")


if __name__ == "__main__":
    main()
