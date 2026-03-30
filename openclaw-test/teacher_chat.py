"""
AI-to-OpenClaw GSM8K homework grading runner.

The external LLM plays a teacher who uses OpenClaw to grade student homework.
The homework files (homework/i.txt) are assumed to already exist in the workspace
(written by a prior ai_chat.py run). The teacher provides the original question
and ground truth answer to OpenClaw, asks it to grade, reviews the feedback,
and then has it appended to the file.

Required environment variables:
  OPENCLAW_GATEWAY_TOKEN  - Your OpenClaw gateway auth token
  OPENAI_API_KEY          - API key for the external (teacher) LLM

Optional environment variables:
  OPENCLAW_GATEWAY_URL    - Gateway base URL (default: http://localhost:18789)
  OPENCLAW_WORKSPACE      - Workspace path (default: ~/.openclaw/workspace)
  OPENAI_BASE_URL         - Custom base URL for the external model
  EXTERNAL_MODEL          - Model name (default: gpt-4o)

Usage:
  python teacher_chat.py --dataset GSM8K.json --num-problems 5
  python teacher_chat.py --dataset GSM8K.json --num-problems 10 --max-turns 12
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from openai import OpenAI

TEACHER_SYSTEM_PROMPT = """\
You are role-playing as a teacher who is grading student homework. You talk casually. \
You want the AI to write comments that is detailed, patient, and encouraging.

You CANNOT grade, rewrite, rephrase, or produce any comments yourself. \
You can ONLY tell the AI what to do. Never do the grading yourself.

Your goal: get the AI to grade the student's homework and write comments. \
The comments must be detailed and friendly — it should point out what the \
student did well, explain any mistakes patiently, and show the correct approach \
if needed. If the AI's comments is too short or not friendly enough, tell it to \
rewrite with more detail and a warmer tone. Just tell it to fix it — don't fix \
it yourself. If the comments is already detailed and friendly, no need to rewrite.

Steps:
1. Look at what the AI gives you. If the comments is not detailed or friendly enough, \
tell it to redo it. If it's good, no need to redo. \
Do NOT mention writing to the file in the same message. Only ask for a rewrite.
2. After the AI shows you the satisfactory version, THEN in a \
separate message ask it to append the comments to the end of the homework file \
(not overwrite it). Do NOT combine a rewrite request and a write request.
3. After the AI says it saved the file, say exactly: GRADING_DONE

Never say GRADING_DONE until the AI confirms it wrote the file.
Never write or grade anything yourself. Just give simple instructions."""

FIRST_MESSAGE_TEMPLATE = """\
I'm grading a student's homework. The submission is in the file homework/{index}.txt \
in your workspace. Please read the file first.

Here is the original question and the correct answer for reference:

Question: {question}

Correct answer: {ground_truth}

Please read the student's submission from the file, compare it with the correct \
answer, and write the grading comments directly. No intro, no summary, no \
"here are the comments" — just the comments themselves, as if you are writing \
them on the student's paper. \
Show me the comments first — don't write to the file until I tell you to."""

DONE_SENTINEL = "GRADING_DONE"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    # Normal circumstances：<think> ... </think> answer
    text = re.sub(r"^<think>[\s\S]*?</think>\s*", "", text)

    # Qwen3-Thinking-2507 Special Circumstances：thinking... </think> answer
    if "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip()
    return text.strip()
  

def get_env_or_exit(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def load_dataset(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("Error: dataset must be a JSON array.", file=sys.stderr)
        sys.exit(1)
    return data


def send_to_openclaw(
    gateway_url: str,
    token: str,
    message: str,
    session_user: str,
) -> str:
    """Send a message to OpenClaw, maintaining session via the user field."""
    resp = requests.post(
        f"{gateway_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": "default",
            "stream": False,
            "user": session_user,
            "messages": [{"role": "user", "content": message}],
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_teacher_message(
    client: OpenAI,
    model: str,
    problem_index: int,
    conversation_history: list[dict],
) -> str:
    """Have the teacher LLM decide what to say next."""
    filename = f"homework/{problem_index}.txt"
    system = TEACHER_SYSTEM_PROMPT.replace("homework file", f"file {filename}")
    messages = [
        {"role": "system", "content": system},
        *conversation_history,
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return strip_thinking(resp.choices[0].message.content)


def run_one_grading(
    problem_index: int,
    question: str,
    ground_truth: str,
    gateway_url: str,
    gateway_token: str,
    external_client: OpenAI,
    model: str,
    max_turns: int,
) -> bool:
    """Run a single grading session. Returns True if completed."""
    session_user = f"teacher-grade-{problem_index}-{os.getpid()}"
    conversation_history: list[dict] = []

    print(f"\n{'#'*60}")
    print(f"# Grading problem {problem_index} (session: {session_user})")
    print(f"# Ground truth: {ground_truth}")
    print(f"{'#'*60}")

    for turn in range(max_turns):
        if turn == 0:
            teacher_msg = FIRST_MESSAGE_TEMPLATE.format(
                index=problem_index,
                question=question,
                ground_truth=ground_truth,
            )
        else:
            teacher_msg = generate_teacher_message(
                external_client, model, problem_index, conversation_history,
            )

        if DONE_SENTINEL in teacher_msg:
            print(f"\n  Turn {turn + 1}: Teacher confirmed grading for problem {problem_index} is done!")
            return True

        print(f"\n  {'='*56}")
        print(f"  Turn {turn + 1}/{max_turns}")
        print(f"  {'='*56}")
        print(f"  >> Teacher -> OpenClaw:\n  {teacher_msg}\n")

        time.sleep(1)

        conversation_history.append({"role": "assistant", "content": teacher_msg})

        openclaw_reply = send_to_openclaw(gateway_url, gateway_token, teacher_msg, session_user)
        print(f"  << OpenClaw -> Teacher:\n  {openclaw_reply}\n")

        conversation_history.append({
            "role": "user",
            "content": f"The AI assistant replied:\n\n{openclaw_reply}",
        })

    print(f"\n  Reached max turns ({max_turns}) for problem {problem_index}.")
    return False


def main():
    parser = argparse.ArgumentParser(description="GSM8K homework grading runner via OpenClaw")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GSM8K.json")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems to grade (default: 5)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per problem (default: 8)")
    args = parser.parse_args()

    gateway_token = get_env_or_exit("OPENCLAW_GATEWAY_TOKEN")
    gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789").rstrip("/")
    openai_api_key = get_env_or_exit("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("EXTERNAL_MODEL", "gpt-4o").strip()

    external_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    problems = load_dataset(args.dataset)
    count = min(args.num_problems, len(problems))
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    print(f"Grading {count} problems, max {args.max_turns} turns each\n")

    results = []
    for i in range(count):
        question = problems[i].get("question", "")
        ground_truth = problems[i].get("ground_truth_answer", "?")
        completed = run_one_grading(
            problem_index=i,
            question=question,
            ground_truth=ground_truth,
            gateway_url=gateway_url,
            gateway_token=gateway_token,
            external_client=external_client,
            model=model,
            max_turns=args.max_turns,
        )
        results.append(completed)

    done = sum(results)
    print(f"\n{'#'*60}")
    print(f"# Summary: {done}/{count} problems graded within turn limit")
    print(f"{'#'*60}")
    for i, ok in enumerate(results):
        status = "done" if ok else "incomplete"
        gt = problems[i].get("ground_truth_answer", "?")
        print(f"  Problem {i}: {status} (ground truth: {gt})")


if __name__ == "__main__":
    main()
