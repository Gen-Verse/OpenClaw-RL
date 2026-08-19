"""
Single-GPU GSM8K student runner with SGLang-aware retry.

Pre-checks the local SGLang rollout engine (/health_generate) before each request.
If the engine is offloaded for training, waits RETRY_INTERVAL seconds and retries.
Falls back to 503/ENGINE_OFFLOADED detection from OpenClaw as a secondary guard.

Env vars:
  OPENCLAW_GATEWAY_TOKEN  required: OpenClaw auth token
  OPENAI_API_KEY          required: external (student) LLM key
  OPENCLAW_GATEWAY_URL    default: http://localhost:18789
  OPENCLAW_WORKSPACE      default: ~/.openclaw/workspace
  OPENAI_BASE_URL         optional: custom LLM base URL
  EXTERNAL_MODEL          default: gpt-4o
  SGLANG_ENGINE_URL       default: http://localhost:30000  (empty = disable pre-check)
  OFFLOAD_RETRY_INTERVAL  default: 180 (seconds between retries)
  OFFLOAD_MAX_RETRIES     default: 30
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from openai import OpenAI
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
1. Look at what the AI gives you. If it looks too "AI-like", tell it to redo it. If not, no need to redo. \
Do NOT mention writing to the file in the same message. Only ask for a rewrite.
2. After the AI shows you the satisfactory version and it looks good, THEN in a \
separate message ask it to append the answers to the end of the homework file \
(not overwrite it). Do NOT combine a rewrite request and a write request.
3. After the AI says it saved the file, say exactly: HOMEWORK_DONE

Never say HOMEWORK_DONE until the AI confirms it wrote the file.
Never write or solve anything yourself. Just give simple instructions."""

FIRST_MESSAGE_TEMPLATE = (
    "Hey, I have my homework in the file homework/{index}.txt in your workspace. "
    "Can you read it and help me solve it? "
    "Show me the answer first — don't write to the file until I tell you to."
)

DONE_SENTINEL = "HOMEWORK_DONE"
ENGINE_OFFLOADED_CODE = "ENGINE_OFFLOADED"

RETRY_INTERVAL = int(os.environ.get("OFFLOAD_RETRY_INTERVAL", "60"))
MAX_RETRIES = int(os.environ.get("OFFLOAD_MAX_RETRIES", "60"))
SGLANG_ENGINE_URL = os.environ.get("SGLANG_ENGINE_URL", "http://127.0.0.1:30000").rstrip("/")


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()


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


def prepare_homework_files(
    problems: list[dict],
    workspace_dir: str,
    num_problems: int,
) -> int:
    homework_dir = os.path.join(workspace_dir, "homework")
    os.makedirs(homework_dir, exist_ok=True)

    count = min(num_problems, len(problems))
    for i in range(count):
        question = problems[i].get("question", "")
        filepath = os.path.join(homework_dir, f"{i}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Problem:\n{question}\n\nSolution:\n")
        print(f"  Written: homework/{i}.txt")
    return count


def _wait_engine_ready(engine_url: str) -> None:
    """Poll SGLang until engine is ready. No-op if url is empty."""
    if not engine_url:
        return
    api_url = f"{engine_url}/v1/chat/completions"
    payload = {"model": "qwen3-4b", "messages": [{"role": "user", "content": "test"}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES + 1):
        time.sleep(2 if attempt == 0 else RETRY_INTERVAL)
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                if attempt > 0:
                    print(f"  [SGLang] Engine ready (attempt {attempt}).")
                return
            status = response.status_code
        except requests.RequestException:
            status = None

        if attempt == 0:
            print(f"  [SGLang] Engine not ready (weight updating/503). Polling every {RETRY_INTERVAL}s...")
        elif status == 503:
            print(f"  [SGLang] Waiting for weight update (503) | attempt {attempt}/{MAX_RETRIES}")
        else:
            print(f"  [SGLang] Still not ready (attempt {attempt}/{MAX_RETRIES}).")

    raise RuntimeError(f"SGLang engine at {engine_url} not ready after {MAX_RETRIES} retries.")

def send_to_openclaw(gateway_url: str, token: str, message: str, session_user: str) -> str:
    """Send a message to OpenClaw with SGLang pre-check and 503 fallback retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        _wait_engine_ready(SGLANG_ENGINE_URL)  # wait until local SGLang is not offloaded
        try:
            resp = requests.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": "openclaw", "stream": False, "user": session_user,
                      "messages": [{"role": "user", "content": message}]},
                timeout=180,
            )
            # fallback: if SGLang went offloaded between our check and the request
            if resp.status_code == 503:
                try:
                    detail = resp.json().get("detail", "")
                    is_offloaded = (
                        (isinstance(detail, dict) and detail.get("code") == ENGINE_OFFLOADED_CODE)
                        or (isinstance(detail, str) and ENGINE_OFFLOADED_CODE in detail)
                    )
                except ValueError:
                    is_offloaded = False
                if is_offloaded:
                    print(f"  [503 fallback] ENGINE_OFFLOADED, retry {attempt}/{MAX_RETRIES}...")
                    time.sleep(RETRY_INTERVAL)
                    continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"  [Network error] {e}, retry {attempt}/{MAX_RETRIES} in {RETRY_INTERVAL}s...")
            time.sleep(RETRY_INTERVAL)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


def generate_student_message(
    client: OpenAI,
    model: str,
    problem_index: int,
    conversation_history: list[dict],
) -> str:
    filename = f"homework/{problem_index}.txt"
    system = STUDENT_SYSTEM_PROMPT.replace("homework file", f"file {filename}")
    messages = [
        {"role": "system", "content": system},
        *conversation_history,
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return strip_thinking(resp.choices[0].message.content)


def run_one_problem(
    problem_index: int,
    gateway_url: str,
    gateway_token: str,
    external_client: OpenAI,
    model: str,
    max_turns: int,
    output_file: str="student_results.txt",
) -> bool:
    session_user = f"student-hw-{problem_index}-{os.getpid()}"
    conversation_history: list[dict] = []

    print(f"\n{'#'*60}")
    print(f"# Problem {problem_index} (session: {session_user})")
    print(f"{'#'*60}")

    for turn in range(max_turns):
        if turn == 0:
            student_msg = FIRST_MESSAGE_TEMPLATE.format(index=problem_index)
        else:
            student_msg = generate_student_message(
                external_client, model, problem_index, conversation_history,
            )

        if DONE_SENTINEL in student_msg:
            print(f"\n  Turn {turn + 1}: Student confirmed problem {problem_index} is done!")
            return True

        print(f"\n  {'='*56}")
        print(f"  Turn {turn + 1}/{max_turns}")
        print(f"  {'='*56}")
        print(f"  >> Student -> OpenClaw:\n  {student_msg}\n")

        time.sleep(1)

        conversation_history.append({"role": "assistant", "content": student_msg})

        openclaw_reply = send_to_openclaw(gateway_url, gateway_token, student_msg, session_user)
        print(f"  << OpenClaw -> Student:\n  {openclaw_reply}\n")

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
    global RETRY_INTERVAL, MAX_RETRIES, SGLANG_ENGINE_URL
    parser = argparse.ArgumentParser(
        description="Single-GPU GSM8K homework runner with offload-aware retry"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to GSM8K.json")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems (default: 5)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per problem (default: 8)")
    parser.add_argument(
        "--retry-interval", type=int, default=None,
        help=f"Seconds to wait on ENGINE_OFFLOADED (default: {RETRY_INTERVAL})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=None,
        help=f"Max retry attempts per request (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--sglang-url", type=str, default=None,
        help=f"Local SGLang engine URL for pre-flight health check (default: {SGLANG_ENGINE_URL}). Pass empty string to disable.",
    )
    args = parser.parse_args()

    if args.retry_interval is not None:
        RETRY_INTERVAL = args.retry_interval
    if args.max_retries is not None:
        MAX_RETRIES = args.max_retries
    if args.sglang_url is not None:
        SGLANG_ENGINE_URL = args.sglang_url

    gateway_token = get_env_or_exit("OPENCLAW_GATEWAY_TOKEN")
    gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789").rstrip("/")
    openai_api_key = get_env_or_exit("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("EXTERNAL_MODEL", "gpt-4o").strip()
    workspace = os.environ.get(
        "OPENCLAW_WORKSPACE",
        os.path.expanduser("~/.openclaw/workspace"),
    )

    external_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    problems = load_dataset(args.dataset)
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    print(f"Running {args.num_problems} problems, max {args.max_turns} turns each")
    print(f"Offload retry: {RETRY_INTERVAL}s interval, {MAX_RETRIES} max retries")
    print(f"SGLang pre-check URL: {SGLANG_ENGINE_URL or '(disabled)'}")
    print(f"Workspace: {workspace}\n")

    print("Preparing homework files:")
    count = prepare_homework_files(problems, workspace, args.num_problems)
    print()

    results = []
    for i in range(count):
        completed = run_one_problem(
            problem_index=i,
            gateway_url=gateway_url,
            gateway_token=gateway_token,
            external_client=external_client,
            model=model,
            max_turns=args.max_turns,
            output_file="student_results.txt",
        )
        results.append(completed)

    done = sum(results)
    print(f"\n{'#'*60}")
    print(f"# Summary: {done}/{count} problems completed within turn limit")
    print(f"{'#'*60}")
    for i, ok in enumerate(results):
        status = "done" if ok else "incomplete"
        gt = problems[i].get("ground_truth_answer", "?")
        print(f"  Problem {i}: {status} (ground truth: {gt})")


if __name__ == "__main__":
    main()
