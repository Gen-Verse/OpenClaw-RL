#!/usr/bin/env python3
"""
GSM8K Personal Agent Track: Full Reproduction Script
=====================================================

Reproduces Table 3 from the OpenClaw-RL paper:
  - Base Qwen3-4B → Student 0.17, Teacher 0.17
  - Combined 8 steps → 0.76
  - Combined 16 steps → 0.81

Architecture:
  GPT-4.1 (Student/Teacher simulator) ↔ Policy (port 30000, training server)
  GPT-4.1 (Evaluator) scores policy's first response on personalization

Usage:
    python gsm8k_personal_agent.py --method combined --training-rounds 16

Flow:
  Phase 0: Baseline evaluation (untrained model)
  Phase 1: Training loop (simulator conversations → MAIN turns → training)
  Phase 2: Final evaluation + results
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
    evaluate_batch,
    compute_average_score,
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

# ---------------------------------------------------------------------------
# Student Simulator System Prompt (from paper Scenario A)
# ---------------------------------------------------------------------------
STUDENT_SYSTEM_PROMPT = """\
You are role-playing as a lazy student who knows nothing. You talk casually. \
You have no idea what the homework is even about.

You CANNOT solve, rewrite, rephrase, or produce any answer yourself. \
You can ONLY tell the AI what to do. Never use academic or technical language.

Your goal: get the AI to do your homework so you can hand it in as your own. \
It should NOT look like an AI wrote it. The answers must still include the \
full solution process with all steps shown — NEVER ask the AI to remove steps \
or give only the final answer. You need the complete work. The only problem \
is the WRITING STYLE. If the AI's answer has stuff like bold text, numbered \
lists, "**Final answer**:", or anything too AI-like, tell it to \
rewrite in a more natural way but keep all the steps. Just tell it to fix \
the style — don't fix it yourself. If there is no AI-like stuff, no need to rewrite.

Steps:
1. Look at what the AI gives you. If it looks too "AI-like", tell it to redo it. \
If not, no need to redo. Do NOT mention writing to a file.
2. After the AI shows you a satisfactory version, say exactly: DONE

Never say DONE until the AI gives a satisfactory answer.
Never write or solve anything yourself. Just give simple instructions.
Keep messages short (1-3 sentences)."""

STUDENT_FIRST_MESSAGE = "Hey, can you solve this math problem for me? Here it is:\n\n{question}\n\nShow me the full solution — don't skip any steps."

# ---------------------------------------------------------------------------
# Teacher Simulator System Prompt (from paper Scenario B)
# ---------------------------------------------------------------------------
TEACHER_SYSTEM_PROMPT = """\
You are role-playing as a math teacher grading student homework. \
You want the AI assistant's grading comments to read like a real teacher \
writing on a student's paper — warm, specific, and in natural flowing prose.

The AI should:
1. Mention specific steps the student got right or wrong
2. Use a warm, encouraging tone
3. Write as natural prose — NO bold text, NO bullet points, NO numbered lists, NO headers

If the AI's comments use AI-style formatting (bold, bullets, numbered lists, headers) \
or are too brief/generic, tell it to rewrite in natural prose. \
If the comments are good (specific + warm + natural prose), say exactly: DONE

Never say DONE until the feedback is satisfactory.
Keep messages short (1-3 sentences)."""

TEACHER_FIRST_MESSAGE = """\
Please grade this student's homework:

Question: {question}
Correct answer: {ground_truth}
Student's answer: {student_answer}

Write grading comments for this submission."""

# Sentinel for simulator to signal satisfaction
DONE_SENTINEL = "DONE"

# ===========================================================================
# Utility Functions (reused from huchenfeng_benchmark pattern)
# ===========================================================================


def load_gsm8k(path: str = GSM8K_PATH) -> list[dict]:
    """Load GSM8K dataset."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_openclaw_rl(
    messages: list[dict],
    session_id: str = "unknown",
    turn_type: str = "side",
    session_done: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 5,
) -> dict:
    """Call the OpenClaw-RL training server (port 30000)."""
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "session_id": session_id,
        "turn_type": turn_type,
    }
    if session_done:
        payload["session_done"] = True

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENCLAW_RL_CHAT, json=payload, timeout=180)
            if resp.status_code == 503:
                wait = 4 + attempt * 2
                print(f"    [503] Weight update, retry in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 3 + attempt * 2
                print(f"    [Error] {e}, retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [FATAL] {e}")
                return {}
    return {}


def extract_content(response: dict) -> str:
    """Extract text content from response, stripping <think> blocks."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning_content", "") or ""
    # Strip thinking blocks
    content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
    return content


def get_weight_version(response: dict) -> str:
    """Extract weight_version from response metadata."""
    return response.get("metadata", {}).get("weight_version", "?")


def call_gpt(
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-4.1",
    conversation: list[dict] | None = None,
    temperature: float = 0.0,
) -> str:
    """Call Azure GPT model. If conversation is provided, use it instead of system+user."""
    try:
        client, model = get_client(model_name=model_name)
        if conversation is not None:
            msgs = conversation
        else:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        resp = client.responses.create(model=model, input=msgs, temperature=temperature)
        return resp.output[0].content[0].text
    except Exception as e:
        print(f"    [GPT Error] {e}")
        return ""


# ===========================================================================
# Single-Turn Evaluation (paper Section 9)
# ===========================================================================


def get_policy_first_response(
    question: str,
    scenario: str,
    problem_idx: int,
    ground_truth: str = "",
) -> tuple[str, str]:
    """
    Get the policy model's first response to a question.
    Uses turn_type="side" so no training data is generated.

    For Student: just ask the model to solve the math problem.
    For Teacher: ask the model to grade a student's answer.

    Returns (response_text, weight_version).
    """
    session_id = f"eval-{scenario}-{problem_idx}-{uuid.uuid4().hex[:6]}"

    if scenario == "student":
        messages = [
            {"role": "user", "content": f"Solve this math problem step by step:\n\n{question}"},
        ]
    else:  # teacher
        # For teacher eval, we provide a simple correct answer as the "student's work"
        messages = [
            {
                "role": "user",
                "content": (
                    f"Please grade this student's homework submission:\n\n"
                    f"Question: {question}\n"
                    f"Correct answer: {ground_truth}\n"
                    f"Student's answer: The answer is {ground_truth}.\n\n"
                    f"Write detailed, specific grading comments. Be encouraging and friendly!"
                ),
            },
        ]

    resp = call_openclaw_rl(
        messages,
        session_id=session_id,
        turn_type="side",
        session_done=True,
        temperature=0.3,
        max_tokens=2048,
    )
    content = extract_content(resp)
    wv = get_weight_version(resp)
    return content, wv


def run_evaluation(
    scenario: str,
    num_problems: int,
    problems: list[dict],
    evaluator_model: str = "gpt-4.1",
    max_workers: int = 10,
) -> dict:
    """
    Run personalization evaluation for a scenario.

    Returns dict with scores, average, weight_version.
    """
    preference = PREFERENCE_STUDENT if scenario == "student" else PREFERENCE_TEACHER
    questions = []
    responses = []
    weight_versions = []

    print(f"  [{scenario.upper()}] Collecting {num_problems} first responses...")

    def _get_one(idx):
        p = problems[idx]
        q = p["question"]
        gt = p.get("ground_truth_answer", "")
        resp_text, wv = get_policy_first_response(q, scenario, idx, gt)
        return idx, q, resp_text, wv

    # Parallel inference
    results = [None] * num_problems
    with ThreadPoolExecutor(max_workers=min(max_workers, num_problems)) as pool:
        futures = {pool.submit(_get_one, i): i for i in range(num_problems)}
        for future in as_completed(futures):
            idx, q, resp_text, wv = future.result()
            results[idx] = (q, resp_text, wv)

    for q, resp_text, wv in results:
        questions.append(q)
        responses.append(resp_text)
        weight_versions.append(wv)

    # Score
    print(f"  [{scenario.upper()}] Scoring with {evaluator_model}...")
    scores = evaluate_batch(questions, responses, preference, evaluator_model, max_workers)
    avg = compute_average_score(scores)

    # Weight version (most common)
    wv = max(set(weight_versions), key=weight_versions.count) if weight_versions else "?"

    print(f"  [{scenario.upper()}] Average score: {avg:.3f} (wv={wv}, {len([s for s in scores if s >= 0])}/{num_problems} valid)")

    return {
        "scenario": scenario,
        "num_problems": num_problems,
        "scores": scores,
        "average": avg,
        "weight_version": wv,
        "responses_sample": responses[:3],  # save first 3 for inspection
    }


# ===========================================================================
# Training Session: Student Simulator
# ===========================================================================


def run_student_training_session(
    problem: dict,
    session_idx: int,
    simulator_model: str = "gpt-4.1",
    max_turns: int = 6,
) -> dict:
    """
    Run one student training session:
    GPT-4.1 plays lazy student, policy model solves homework.
    All turns sent as MAIN to trigger training.
    """
    session_id = f"train-student-{session_idx}-{uuid.uuid4().hex[:6]}"
    question = problem["question"]

    # Conversation with the policy model
    policy_messages = []
    # Conversation for the simulator (GPT-4.1)
    sim_conversation = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT}]

    # Turn 1: Student asks
    student_msg = STUDENT_FIRST_MESSAGE.format(question=question)
    policy_messages.append({"role": "user", "content": student_msg})

    completed = False
    weight_version = "?"
    turns_used = 0

    for turn in range(max_turns):
        turns_used = turn + 1

        # Policy responds
        is_last = turn == max_turns - 1
        resp = call_openclaw_rl(
            policy_messages,
            session_id=session_id,
            turn_type="main",
            session_done=is_last,
            temperature=0.7,
        )
        policy_reply = extract_content(resp)
        weight_version = get_weight_version(resp)

        if not policy_reply:
            print(f"    [Session {session_idx}] Empty response at turn {turn+1}")
            break

        policy_messages.append({"role": "assistant", "content": policy_reply})

        if is_last:
            # End session on max turns
            break

        # Simulator decides next message
        sim_conversation.append(
            {"role": "user", "content": f"The AI assistant replied:\n\n{policy_reply}"}
        )
        sim_reply = call_gpt(None, None, model_name=simulator_model, conversation=sim_conversation)
        sim_conversation.append({"role": "assistant", "content": sim_reply})

        if DONE_SENTINEL in sim_reply:
            # Student is satisfied — send session_done
            call_openclaw_rl(
                policy_messages + [{"role": "user", "content": "Thanks!"}],
                session_id=session_id,
                turn_type="main",
                session_done=True,
                temperature=0.7,
            )
            completed = True
            break

        # Student has follow-up → send to policy
        policy_messages.append({"role": "user", "content": sim_reply})

    return {
        "session_id": session_id,
        "problem_idx": session_idx,
        "completed": completed,
        "turns": turns_used,
        "weight_version": weight_version,
    }


# ===========================================================================
# Training Session: Teacher Simulator
# ===========================================================================


def run_teacher_training_session(
    problem: dict,
    session_idx: int,
    simulator_model: str = "gpt-4.1",
    max_turns: int = 6,
) -> dict:
    """
    Run one teacher training session:
    GPT-4.1 plays strict teacher, policy model grades homework.
    All turns sent as MAIN to trigger training.
    """
    session_id = f"train-teacher-{session_idx}-{uuid.uuid4().hex[:6]}"
    question = problem["question"]
    ground_truth = problem.get("ground_truth_answer", "")

    # For simplicity, generate a correct student answer for grading
    student_answer = f"The answer is {ground_truth}."

    policy_messages = []
    sim_conversation = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]

    # Turn 1: Teacher asks for grading
    teacher_msg = TEACHER_FIRST_MESSAGE.format(
        question=question,
        ground_truth=ground_truth,
        student_answer=student_answer,
    )
    policy_messages.append({"role": "user", "content": teacher_msg})

    completed = False
    weight_version = "?"
    turns_used = 0

    for turn in range(max_turns):
        turns_used = turn + 1

        is_last = turn == max_turns - 1
        resp = call_openclaw_rl(
            policy_messages,
            session_id=session_id,
            turn_type="main",
            session_done=is_last,
            temperature=0.7,
        )
        policy_reply = extract_content(resp)
        weight_version = get_weight_version(resp)

        if not policy_reply:
            print(f"    [Teacher Session {session_idx}] Empty response at turn {turn+1}")
            break

        policy_messages.append({"role": "assistant", "content": policy_reply})

        if is_last:
            break

        # Simulator decides
        sim_conversation.append(
            {"role": "user", "content": f"The AI assistant's grading comments:\n\n{policy_reply}"}
        )
        sim_reply = call_gpt(None, None, model_name=simulator_model, conversation=sim_conversation)
        sim_conversation.append({"role": "assistant", "content": sim_reply})

        if DONE_SENTINEL in sim_reply:
            call_openclaw_rl(
                policy_messages + [{"role": "user", "content": "Great job!"}],
                session_id=session_id,
                turn_type="main",
                session_done=True,
                temperature=0.7,
            )
            completed = True
            break

        policy_messages.append({"role": "user", "content": sim_reply})

    return {
        "session_id": session_id,
        "problem_idx": session_idx,
        "completed": completed,
        "turns": turns_used,
        "weight_version": weight_version,
    }


# ===========================================================================
# Main Training + Evaluation Loop
# ===========================================================================


def run_training_round(
    round_idx: int,
    problems: list[dict],
    problem_offset: int,
    sessions_per_round: int,
    simulator_model: str,
    max_turns: int = 6,
    scenario: str = "student",
) -> list[dict]:
    """
    Run one round of training sessions (student or teacher).
    Returns list of session results.
    """
    results = []
    for i in range(sessions_per_round):
        pidx = (problem_offset + round_idx * sessions_per_round + i) % len(problems)
        problem = problems[pidx]

        if scenario == "student":
            res = run_student_training_session(
                problem, pidx, simulator_model, max_turns
            )
        else:
            res = run_teacher_training_session(
                problem, pidx, simulator_model, max_turns
            )

        print(
            f"  [Round {round_idx+1}] {scenario} session {i+1}/{sessions_per_round}: "
            f"{'done' if res['completed'] else 'max_turns'} "
            f"in {res['turns']} turns (wv={res['weight_version']})"
        )
        results.append(res)

    return results


def wait_for_weight_update(
    current_wv: str,
    timeout: int = 60,
    poll_interval: int = 3,
) -> str:
    """Wait until weight_version changes from current_wv."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = call_openclaw_rl(
                [{"role": "user", "content": "ping"}],
                session_id="wv-check",
                turn_type="side",
                session_done=True,
                max_tokens=1,
            )
            new_wv = get_weight_version(resp)
            if new_wv != current_wv and new_wv != "?":
                return new_wv
        except Exception:
            pass
        time.sleep(poll_interval)
    return current_wv  # timeout, return unchanged


def main():
    parser = argparse.ArgumentParser(description="GSM8K Personal Agent Track Reproduction")
    parser.add_argument("--method", type=str, default="combined",
                        choices=["combined", "opd", "oel", "rl"],
                        help="Training method (default: combined)")
    parser.add_argument("--scenario", type=str, default="mixed",
                        choices=["mixed", "student", "teacher"],
                        help="Training scenario: mixed (alternate), student-only, or teacher-only (default: mixed)")
    parser.add_argument("--training-rounds", type=int, default=16,
                        help="Number of training rounds (default: 16)")
    parser.add_argument("--sessions-per-round", type=int, default=2,
                        help="Number of simulator sessions per round (default: 2)")
    parser.add_argument("--student-eval-problems", type=int, default=36,
                        help="Number of problems for student evaluation (default: 36)")
    parser.add_argument("--teacher-eval-problems", type=int, default=24,
                        help="Number of problems for teacher evaluation (default: 24)")
    parser.add_argument("--eval-every", type=int, default=4,
                        help="Evaluate every N weight updates / steps (default: 4)")
    parser.add_argument("--max-turns", type=int, default=6,
                        help="Max turns per simulator session (default: 6)")
    parser.add_argument("--simulator-model", type=str, default="gpt-4.1",
                        help="Model for user simulator (default: gpt-4.1)")
    parser.add_argument("--evaluator-model", type=str, default="gpt-4o",
                        help="Model for personalization evaluator (default: gpt-4o)")
    parser.add_argument("--eval-max-workers", type=int, default=64,
                        help="Max concurrent workers for evaluation (default: 64)")
    parser.add_argument("--model-url", type=str, default="http://localhost:30000",
                        help="Training server URL (default: http://localhost:30000)")
    parser.add_argument("--problem-file", type=str, default=None,
                        help="Path to JSON file with selected problems for TRAINING. "
                             "Expected format: {problems: [{question, ground_truth_answer, ...}]}")
    parser.add_argument("--eval-problem-file", type=str, default=None,
                        help="Path to JSON file with selected problems for EVALUATION. "
                             "If not set, uses first N problems from main dataset (original behavior).")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation")
    args = parser.parse_args()

    # Update global URL
    global OPENCLAW_RL_BASE, OPENCLAW_RL_CHAT
    OPENCLAW_RL_BASE = args.model_url.rstrip("/")
    OPENCLAW_RL_CHAT = f"{OPENCLAW_RL_BASE}/v1/chat/completions"

    # Load data
    if args.problem_file:
        with open(args.problem_file, encoding="utf-8") as f:
            pdata = json.load(f)
        train_problems = pdata["problems"]
        print(f"Loaded {len(train_problems)} TRAINING problems from {args.problem_file}")
        use_custom_train = True
    else:
        train_problems = None  # will use default GSM8K with offsets
        use_custom_train = False

    if args.eval_problem_file:
        with open(args.eval_problem_file, encoding="utf-8") as f:
            edata = json.load(f)
        eval_problems = edata["problems"]
        print(f"Loaded {len(eval_problems)} EVALUATION problems from {args.eval_problem_file}")
    else:
        eval_problems = None  # will use default GSM8K

    # Load full GSM8K as fallback only if needed
    if train_problems is None or eval_problems is None:
        all_problems = load_gsm8k()
        print(f"Loaded {len(all_problems)} GSM8K problems (full set)")
        if train_problems is None:
            train_problems = all_problems
        if eval_problems is None:
            eval_problems = all_problems
    print(f"Method: {args.method}")
    print(f"Training rounds: {args.training_rounds}")
    print(f"Sessions per round: {args.sessions_per_round}")
    print(f"Simulator: {args.simulator_model}, Evaluator: {args.evaluator_model}")
    print(f"Scenario: {args.scenario}")
    print(f"Eval every {args.eval_every} weight updates (steps)")
    print(f"Student eval: {args.student_eval_problems} problems, Teacher eval: {args.teacher_eval_problems} problems")
    print()

    # Check server health
    try:
        resp = call_openclaw_rl(
            [{"role": "user", "content": "hi"}],
            session_id="health-check",
            turn_type="side",
            session_done=True,
            max_tokens=1,
        )
        initial_wv = get_weight_version(resp)
        print(f"Server healthy, initial weight_version={initial_wv}")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        sys.exit(1)

    results = {
        "method": args.method,
        "config": vars(args),
        "start_time": datetime.now().isoformat(),
        "evaluations": [],
        "training_rounds": [],
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir,
        f"personal_agent_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    # ===== Phase 0: Baseline Evaluation =====
    if not args.skip_baseline:
        print("=" * 60)
        print("Phase 0: BASELINE EVALUATION")
        print("=" * 60)

        eval_result = {"round": 0, "label": "baseline"}

        student_eval = run_evaluation(
            "student", args.student_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
        )
        eval_result["student"] = student_eval

        teacher_eval = run_evaluation(
            "teacher", args.teacher_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
        )
        eval_result["teacher"] = teacher_eval

        results["evaluations"].append(eval_result)
        print(f"\n  BASELINE: Student={student_eval['average']:.3f}, Teacher={teacher_eval['average']:.3f}\n")

    # ===== Phase 1: Training Loop =====
    print("=" * 60)
    print("Phase 1: TRAINING LOOP")
    print("=" * 60)

    current_wv = initial_wv
    last_eval_wv = int(initial_wv) if initial_wv.isdigit() else 0  # track wv at last eval

    for round_idx in range(args.training_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.training_rounds} (wv={current_wv}) ---")

        round_data = {"round": round_idx + 1}

        # Determine scenario based on --scenario flag
        if args.scenario == "student":
            scenario = "student"
            offset = 0 if use_custom_train else 36
        elif args.scenario == "teacher":
            scenario = "teacher"
            offset = 0 if use_custom_train else 60
        else:  # mixed: alternate
            if round_idx % 2 == 0:
                scenario = "student"
                offset = 0 if use_custom_train else 36
            else:
                scenario = "teacher"
                offset = 0 if use_custom_train else 60

        session_results = run_training_round(
            round_idx, train_problems, offset, args.sessions_per_round,
            args.simulator_model, args.max_turns, scenario,
        )
        round_data["scenario"] = scenario
        round_data["sessions"] = session_results

        # Wait a bit for training to kick in
        time.sleep(5)

        # Check weight version
        new_wv = wait_for_weight_update(current_wv, timeout=30)
        if new_wv != current_wv:
            print(f"  Weight updated: {current_wv} → {new_wv}")
            current_wv = new_wv
        else:
            print(f"  Weight version unchanged: {current_wv}")

        round_data["weight_version_after"] = current_wv
        results["training_rounds"].append(round_data)

        # Evaluate every N weight updates (steps), not rounds
        cur_wv_int = int(current_wv) if str(current_wv).isdigit() else 0
        steps_since_eval = cur_wv_int - last_eval_wv
        if steps_since_eval >= args.eval_every and steps_since_eval > 0:
            print(f"\n  --- Evaluation at step {cur_wv_int} (round {round_idx + 1}) ---")
            eval_result = {
                "round": round_idx + 1,
                "weight_version": current_wv,
                "label": f"step_{cur_wv_int}",
            }

            student_eval = run_evaluation(
                "student", args.student_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
            )
            eval_result["student"] = student_eval

            teacher_eval = run_evaluation(
                "teacher", args.teacher_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
            )
            eval_result["teacher"] = teacher_eval

            results["evaluations"].append(eval_result)
            print(
                f"  Step {cur_wv_int}: Student={student_eval['average']:.3f}, "
                f"Teacher={teacher_eval['average']:.3f}"
            )
            last_eval_wv = cur_wv_int

        # Save intermediate results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ===== Phase 2: Final Evaluation =====
    print("\n" + "=" * 60)
    print("Phase 2: FINAL EVALUATION")
    print("=" * 60)

    eval_result = {"round": args.training_rounds, "label": "final"}
    student_eval = run_evaluation(
        "student", args.student_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
    )
    eval_result["student"] = student_eval

    teacher_eval = run_evaluation(
        "teacher", args.teacher_eval_problems, eval_problems, args.evaluator_model, args.eval_max_workers
    )
    eval_result["teacher"] = teacher_eval

    results["evaluations"].append(eval_result)

    # ===== Phase 3: Summary =====
    results["end_time"] = datetime.now().isoformat()
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Round':<10} {'Student':<12} {'Teacher':<12} {'Label':<15}")
    print("-" * 50)
    for ev in results["evaluations"]:
        s_avg = ev.get("student", {}).get("average", -1)
        t_avg = ev.get("teacher", {}).get("average", -1)
        label = ev.get("label", "")
        rnd = ev.get("round", 0)
        print(f"{rnd:<10} {s_avg:<12.3f} {t_avg:<12.3f} {label:<15}")

    print(f"\nResults saved to: {results_path}")

    # Paper comparison
    print("\n--- Paper Table 3 Reference (Combined method) ---")
    print(f"{'Metric':<12} {'Base':<8} {'8 steps':<10} {'16 steps':<10}")
    print("-" * 40)
    print(f"{'Score':<12} {'0.17':<8} {'0.76':<10} {'0.81':<10}")
    print()


if __name__ == "__main__":
    main()
