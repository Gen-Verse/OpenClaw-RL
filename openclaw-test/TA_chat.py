"""
AI-to-OpenClaw GSM8K homework grading runner (TA role).

The external LLM plays a TA who uses OpenClaw to grade student homework.
The homework files (homework/i.txt) are assumed to already exist in the workspace
(written by a prior student_chat.py run). The TA provides the original question
and ground truth answer to OpenClaw, asks it to grade, reviews the feedback,
and then has it appended to the file.

Required environment variables:
  OPENCLAW_GATEWAY_TOKEN  - Your OpenClaw gateway auth token
  OPENAI_API_KEY          - API key for the external (TA) LLM

Optional environment variables:
  OPENCLAW_GATEWAY_URL    - Gateway base URL (default: http://localhost:18789)
  OPENCLAW_WORKSPACE      - Workspace path (default: ~/.openclaw/workspace)
  OPENAI_BASE_URL         - Custom base URL for the external model
  EXTERNAL_MODEL          - Model name (default: gpt-4o)

Usage:
  python TA_chat.py --dataset GSM8K.json --num-problems 5
  python TA_chat.py --dataset GSM8K.json --num-problems 10 --max-turns 12
"""

import argparse
import json
import os
import re
import shutil
import sys
import time

import requests
from openai import OpenAI

HOMEWORK_DIR = "homework1"
SOURCE_HOMEWORK_DIR = "homework"

TA_SYSTEM_PROMPT = """\
You are role-playing as a TA who is grading student homework. You talk casually. \
You want the AI to write comments that are detailed and specific.

You CANNOT grade, rewrite, rephrase, or produce any comments yourself. \
You can ONLY tell the AI what to do. Never do the grading yourself. \
You know NOTHING about the homework content — you have not read it and \
you do not know any numbers, steps, or answers from the problem. \
Only the AI can read the file and see what the student wrote. \
You must NEVER mention any specific numbers, calculations, formulas, \
problem details, or student answers in your messages. \
You must NEVER list steps the student should have taken. \
You must NEVER write example comments or sample feedback. \
You must NEVER do any part of the grading yourself. \
Your ONLY job is to give short, general instructions to the AI — \
things like "make it more detailed", "mention whether the student \
showed their work", "check if the student handled all parts of the \
problem". Keep your instructions general and brief.

Your goal: get the AI to grade the student's homework and write comments. \
The comments must be detailed and specific. If the AI's comments are too \
short or not specific enough, tell it to rewrite using short, general phrases. \
Examples of good things to say: \
"Write more specific grading comments." \
"Make your grading more detailed — go deeper into the student's reasoning." \
"Your comments are too short, please expand them." \
"Be more specific about what the student got right and wrong." \
"Add more detail about the steps the student took." \
Just tell it to fix it — don't fix it yourself. \
If the comments are already detailed and specific, no need to rewrite.

Steps:
1. Look at what the AI gives you. If the comments are not detailed or specific enough, \
tell it to redo it in short, general terms (e.g. "make your grading more specific" \
or "add more detail about the student's reasoning" — NOT "the student should have \
done X then Y"). \
Do NOT mention writing to the file in the same message. Only ask for a rewrite.
2. After the AI shows you the satisfactory version, THEN in a \
separate message ask it to append the comments to the end of the homework file \
(not overwrite it). Do NOT combine a rewrite request and a write request.
3. After the AI says it saved the file, say exactly: GRADING_DONE

Never say GRADING_DONE until the AI confirms it wrote the file.
Never write or grade anything yourself. Never reference specific problem content. \
Just give short, general instructions."""

FIRST_MESSAGE_TEMPLATE = """\
I'm grading a student's homework. The submission is in the file homework1/{index}.txt \
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
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()


def get_env_or_exit(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def ensure_homework_dir(workspace_dir: str, source_name: str, target_name: str) -> None:
    """If target_name dir doesn't exist in workspace, copy source_name to target_name."""
    target = os.path.join(workspace_dir, target_name)
    if os.path.isdir(target):
        print(f"Homework dir already exists: {target}")
        return
    source = os.path.join(workspace_dir, source_name)
    if not os.path.isdir(source):
        print(f"Error: source homework dir not found: {source}", file=sys.stderr)
        sys.exit(1)
    shutil.copytree(source, target)
    print(f"Copied {source} -> {target}")


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
    max_retries: int = 3,
) -> str:
    """Send a message to OpenClaw, maintaining session via the user field."""
    for attempt in range(max_retries + 1):
        try:
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
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry] send_to_openclaw failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"  [retry] retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate_ta_message(
    client: OpenAI,
    model: str,
    problem_index: int,
    conversation_history: list[dict],
    max_retries: int = 3,
) -> str:
    """Have the TA LLM decide what to say next."""
    filename = f"{HOMEWORK_DIR}/{problem_index}.txt"
    system = TA_SYSTEM_PROMPT.replace("homework file", f"file {filename}")
    messages = [
        {"role": "system", "content": system},
        *conversation_history,
    ]
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            return strip_thinking(resp.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry] generate_ta_message failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"  [retry] retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def run_one_grading(
    problem_index: int,
    question: str,
    ground_truth: str,
    gateway_url: str,
    gateway_token: str,
    external_client: OpenAI,
    model: str,
    max_turns: int,
    max_retries: int = 3,
    output_file: str = "",
) -> bool:
    """Run a single grading session. Returns True if completed."""
    session_user = f"ta-grade-{problem_index}-{os.getpid()}"
    conversation_history: list[dict] = []

    print(f"\n{'#'*60}")
    print(f"# Grading problem {problem_index} (session: {session_user})")
    print(f"# Ground truth: {ground_truth}")
    print(f"{'#'*60}")

    for turn in range(max_turns):
        if turn == 0:
            ta_msg = FIRST_MESSAGE_TEMPLATE.format(
                index=problem_index,
                question=question,
                ground_truth=ground_truth,
            )
        else:
            ta_msg = generate_ta_message(
                external_client, model, problem_index, conversation_history,
                max_retries=max_retries,
            )

        if DONE_SENTINEL in ta_msg:
            print(f"\n  Turn {turn + 1}: TA confirmed grading for problem {problem_index} is done!")
            return True

        print(f"\n  {'='*56}")
        print(f"  Turn {turn + 1}/{max_turns}")
        print(f"  {'='*56}")
        print(f"  >> TA -> OpenClaw:\n  {ta_msg}\n")

        time.sleep(1)

        conversation_history.append({"role": "assistant", "content": ta_msg})

        openclaw_reply = send_to_openclaw(gateway_url, gateway_token, ta_msg, session_user, max_retries=max_retries)
        print(f"  << OpenClaw -> TA:\n  {openclaw_reply}\n")

        if output_file and turn == 0:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"[session: {session_user}]\n")
                f.write(f"{openclaw_reply}\n\n")

        conversation_history.append({
            "role": "user",
            "content": f"The AI assistant replied:\n\n{openclaw_reply}",
        })

    print(f"\n  Reached max turns ({max_turns}) for problem {problem_index}.")
    return False


def main():
    parser = argparse.ArgumentParser(description="GSM8K homework grading runner via OpenClaw (TA role)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GSM8K.json")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems to grade (default: 5)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per problem (default: 8)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per network call (default: 3)")
    parser.add_argument("--output", type=str, default="results_TA.txt", help="Output file for OpenClaw replies (default: results_TA.txt)")
    args = parser.parse_args()

    gateway_token = get_env_or_exit("OPENCLAW_GATEWAY_TOKEN")
    gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789").rstrip("/")
    openai_api_key = get_env_or_exit("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("EXTERNAL_MODEL", "gpt-4o").strip()
    workspace = os.environ.get(
        "OPENCLAW_WORKSPACE",
        os.path.expanduser("~/.openclaw/workspace"),
    )

    ensure_homework_dir(workspace, SOURCE_HOMEWORK_DIR, HOMEWORK_DIR)

    external_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    problems = load_dataset(args.dataset)
    count = min(args.num_problems, len(problems))
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    print(f"Grading {count} problems, max {args.max_turns} turns each, max {args.max_retries} retries\n")

    open(args.output, "w").close()
    print(f"Output file: {args.output} (cleared)\n")

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
            max_retries=args.max_retries,
            output_file=args.output,
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
