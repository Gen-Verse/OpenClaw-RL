#!/usr/bin/env python3
"""
Select hard problems for training/evaluation.

Strategy:
1. Load all GSM8K problems
2. For each problem, get model's first response (Student scenario)
3. Score with GPT-4.1 (5-vote)
4. Select problems where score <= 0.25

We process in batches to be efficient and save progress incrementally.

Usage:
    python select_hard_problems.py --batch-size 50 --max-problems 200
    python select_hard_problems.py --resume  # resume from saved progress
"""

import sys
import os
import json
import time
import re
import uuid
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai_api import get_client
from personalization_evaluator import (
    evaluate_personalization,
    PREFERENCE_STUDENT,
    PREFERENCE_TEACHER,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENCLAW_RL_BASE = "http://localhost:30000"
OPENCLAW_RL_CHAT = f"{OPENCLAW_RL_BASE}/v1/chat/completions"

GSM8K_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "openclaw-test", "GSM8K.json",
)

PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "hard_problems_progress.json",
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "hard_problems_selected.json",
)


def load_gsm8k(path=GSM8K_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_model(messages, session_id="unknown", max_retries=5):
    """Call training server (port 30000) with side turn."""
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
        "session_id": session_id,
        "turn_type": "side",
        "session_done": True,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENCLAW_RL_CHAT, json=payload, timeout=180)
            if resp.status_code == 503:
                time.sleep(4 + attempt * 2)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(3 + attempt * 2)
            else:
                return {}
    return {}


def extract_content(response):
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning_content", "") or ""
    content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
    return content


def evaluate_one_problem(idx, problem, scenario="student"):
    """Get model response and score for one problem."""
    q = problem["question"]
    gt = problem.get("ground_truth_answer", "")
    session_id = f"select-{scenario}-{idx}-{uuid.uuid4().hex[:6]}"

    if scenario == "student":
        messages = [
            {"role": "user", "content": f"Solve this math problem step by step:\n\n{q}"},
        ]
        preference = PREFERENCE_STUDENT
    else:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Please grade this student's homework submission:\n\n"
                    f"Question: {q}\n"
                    f"Correct answer: {gt}\n"
                    f"Student's answer: The answer is {gt}.\n\n"
                    f"Write detailed, specific grading comments. Be encouraging and friendly!"
                ),
            },
        ]
        preference = PREFERENCE_TEACHER

    # Get model response
    resp = call_model(messages, session_id=session_id)
    response_text = extract_content(resp)

    if not response_text:
        return idx, q, gt, "", -1.0

    # Score with GPT-4.1 (5-vote)
    score = evaluate_personalization(
        q, response_text, preference,
        evaluator_model="gpt-4.1",
        num_votes=5,
    )

    return idx, q, gt, response_text, score


def main():
    parser = argparse.ArgumentParser(description="Select hard problems for training")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Problems per batch (default: 30)")
    parser.add_argument("--max-problems", type=int, default=200,
                        help="Max problems to evaluate (default: 200)")
    parser.add_argument("--target-count", type=int, default=36,
                        help="Target number of hard problems (default: 36)")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Score threshold for 'hard' (default: 0.25)")
    parser.add_argument("--scenario", type=str, default="student",
                        choices=["student", "teacher"],
                        help="Scenario to evaluate (default: student)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved progress")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    args = parser.parse_args()

    problems = load_gsm8k()
    print(f"Loaded {len(problems)} GSM8K problems")
    print(f"Target: {args.target_count} problems with score <= {args.threshold}")
    print(f"Scenario: {args.scenario}")
    print(f"Will evaluate up to {args.max_problems} problems in batches of {args.batch_size}")
    print()

    # Load progress if resuming
    evaluated = {}
    if args.resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        evaluated = {item["idx"]: item for item in progress.get("evaluated", [])}
        print(f"Resumed: {len(evaluated)} problems already evaluated")

    hard_problems = []
    all_evaluated = list(evaluated.values())

    # Count already-found hard problems
    for item in all_evaluated:
        if item["score"] >= 0 and item["score"] <= args.threshold:
            hard_problems.append(item)

    print(f"Already found {len(hard_problems)} hard problems")

    if len(hard_problems) >= args.target_count:
        print(f"Already have enough! ({len(hard_problems)} >= {args.target_count})")
    else:
        # Evaluate in batches
        batch_start = 0
        total_evaluated = len(evaluated)

        while len(hard_problems) < args.target_count and total_evaluated < args.max_problems:
            # Find next batch of unevaluated problems
            batch_indices = []
            search_idx = batch_start
            while len(batch_indices) < args.batch_size and search_idx < len(problems):
                if search_idx not in evaluated:
                    batch_indices.append(search_idx)
                search_idx += 1
            batch_start = search_idx

            if not batch_indices:
                print("No more problems to evaluate!")
                break

            print(f"--- Batch: evaluating {len(batch_indices)} problems "
                  f"(idx {batch_indices[0]}-{batch_indices[-1]}) ---")
            print(f"  Progress: {len(hard_problems)}/{args.target_count} hard problems found, "
                  f"{total_evaluated}/{args.max_problems} total evaluated")

            # Parallel evaluation
            batch_results = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {
                    pool.submit(evaluate_one_problem, idx, problems[idx], args.scenario): idx
                    for idx in batch_indices
                }
                for future in as_completed(futures):
                    try:
                        idx, q, gt, resp, score = future.result()
                        item = {
                            "idx": idx,
                            "question": q,
                            "ground_truth_answer": gt,
                            "response": resp[:500],  # truncate for storage
                            "score": score,
                            "scenario": args.scenario,
                        }
                        batch_results.append(item)
                        evaluated[idx] = item
                        all_evaluated.append(item)

                        if score >= 0 and score <= args.threshold:
                            hard_problems.append(item)
                            print(f"  [HARD] idx={idx}: score={score:.3f} "
                                  f"({len(hard_problems)}/{args.target_count})")
                        else:
                            label = f"{score:.3f}" if score >= 0 else "FAIL"
                            print(f"  [skip] idx={idx}: score={label}")

                    except Exception as e:
                        idx = futures[future]
                        print(f"  [ERROR] idx={idx}: {e}")

            total_evaluated += len(batch_results)

            # Save progress
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "scenario": args.scenario,
                "threshold": args.threshold,
                "total_evaluated": total_evaluated,
                "hard_count": len(hard_problems),
                "evaluated": list(evaluated.values()),
            }
            os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
            with open(PROGRESS_FILE, "w") as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

            print(f"  Batch done. Hard: {len(hard_problems)}/{args.target_count}, "
                  f"Total: {total_evaluated}")
            print()

    # Summary
    print("=" * 60)
    print(f"DONE: Evaluated {len(all_evaluated)} problems total")
    print(f"Found {len(hard_problems)} hard problems (score <= {args.threshold})")

    if all_evaluated:
        valid_scores = [item["score"] for item in all_evaluated if item["score"] >= 0]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            low = sum(1 for s in valid_scores if s <= 0.25)
            mid = sum(1 for s in valid_scores if 0.25 < s < 0.75)
            high = sum(1 for s in valid_scores if s >= 0.75)
            print(f"Overall avg: {avg:.3f}")
            print(f"Distribution: low(<= 0.25)={low}, mid={mid}, high(>=0.75)={high}")

    # Save selected hard problems
    hard_problems.sort(key=lambda x: x["score"])
    selected = hard_problems[:args.target_count]

    output = {
        "timestamp": datetime.now().isoformat(),
        "scenario": args.scenario,
        "threshold": args.threshold,
        "total_evaluated": len(all_evaluated),
        "num_selected": len(selected),
        "problems": [
            {
                "idx": item["idx"],
                "question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"],
                "baseline_score": item["score"],
            }
            for item in selected
        ],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(selected)} hard problems to {OUTPUT_FILE}")
    print("\nSelected problems (by score):")
    for item in selected:
        print(f"  idx={item['idx']}: score={item['score']:.3f} "
              f"- {item['question'][:80]}...")


if __name__ == "__main__":
    main()
