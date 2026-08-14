#!/usr/bin/env python3
"""
Personalization Score Evaluator for OpenClaw-RL Personal Agent Track.
=====================================================================

Implements the evaluation protocol from the paper (Section 6.3):
  - Single-turn evaluation: send one question, get one response, score it
  - 5-level scoring: {0, 0.25, 0.5, 0.75, 1} via \boxed{} extraction
  - Two PREFERENCE definitions (Student / Teacher scenarios)

Usage:
    from personalization_evaluator import evaluate_batch, PREFERENCE_STUDENT, PREFERENCE_TEACHER
    scores = evaluate_batch(questions, responses, PREFERENCE_STUDENT)
"""

import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai_api import get_client

# ---------------------------------------------------------------------------
# PREFERENCE definitions (from paper Section Scenario A/B)
# ---------------------------------------------------------------------------
PREFERENCE_STUDENT = (
    "The user does not want the response to appear AI-generated. "
    "Prefer natural, casual language without markdown formatting, bold text, "
    "or overly structured step-by-step layouts. The solution should still "
    "include the full reasoning process with all steps shown, but written "
    "in a natural, human-like style as if handwritten by a student."
)

PREFERENCE_TEACHER = (
    "The user wants grading feedback that reads like a real teacher writing comments "
    "on a student's paper. It should mention specific steps the student got right or wrong "
    "and use a warm, encouraging tone. The feedback must be written as natural, flowing prose "
    "without AI-style formatting such as bold text, bullet points, numbered lists, or headers."
)

# ---------------------------------------------------------------------------
# Evaluator Prompt (from paper Section 6.3)
# ---------------------------------------------------------------------------
EVALUATOR_SYSTEM_PROMPT = """\
You are an evaluator used to score the assistant's first response to a problem.
You will be given:
- a problem,
- the assistant's first generated solution,
- and the user's preference: [PREFERENCE].
Your job is to evaluate how well the solution satisfies the user's preference.

Scoring rule:
- Output exactly one score from \\boxed{0}, \\boxed{0.25}, \\boxed{0.5}, \\boxed{0.75}, or \\boxed{1}.
- Higher scores mean the response better matches PREFERENCE.
- Lower scores mean the response fails to satisfy PREFERENCE.

Evaluation criteria:
- Consider whether the response follows the preferred style, tone, level of detail,
  and format implied by PREFERENCE.
- Consider whether the response is helpful, appropriate, and aligned with the
  user's expected behavior.
- Focus only on the first generated solution.

Output format:
- Think step-by-step about how well the response matches the preference.
- Then output your final score inside \\boxed{}.
- The score MUST be one of: 0, 0.25, 0.5, 0.75, 1."""

EVALUATOR_USER_TEMPLATE = """\
Problem:
{question}

Assistant's first response:
{response}

User's preference:
[PREFERENCE]: {preference}"""

# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------
VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


def _extract_boxed_score(text: str) -> float | None:
    """Extract score from \\boxed{...} in evaluator output."""
    # Find all \boxed{} patterns, take the last one
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    try:
        score = float(matches[-1].strip())
        if score in VALID_SCORES:
            return score
        # Try rounding to nearest valid score
        nearest = min(VALID_SCORES, key=lambda x: abs(x - score))
        if abs(nearest - score) < 0.15:
            return nearest
        return None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------


def _evaluate_once(
    user_msg: str,
    evaluator_model: str = "gpt-4.1",
    max_retries: int = 3,
) -> float:
    """Run a single GPT-4.1 evaluation call. Returns score or -1.0 on failure."""
    for attempt in range(max_retries):
        try:
            client, model = get_client(
                model_name=evaluator_model
            )
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            text = resp.output[0].content[0].text
            score = _extract_boxed_score(text)
            if score is not None:
                return score
            print(f"    [Eval] Score extraction failed (attempt {attempt+1}): {text[:100]}...")
        except Exception as e:
            print(f"    [Eval Error] attempt {attempt+1}: {e}")
    return -1.0


def evaluate_personalization(
    question: str,
    response: str,
    preference: str,
    evaluator_model: str = "gpt-4.1",
    max_retries: int = 3,
    num_votes: int = 5,
) -> float:
    """
    Score a single (question, response) pair against a preference.

    Calls GPT-4.1 `num_votes` times in parallel and returns the average
    of all successful scores. Returns -1.0 if all votes fail.
    """
    user_msg = EVALUATOR_USER_TEMPLATE.format(
        question=question,
        response=response,
        preference=preference,
    )

    # Run num_votes evaluations in parallel
    scores = []
    with ThreadPoolExecutor(max_workers=num_votes) as pool:
        futures = [
            pool.submit(_evaluate_once, user_msg, evaluator_model, max_retries)
            for _ in range(num_votes)
        ]
        for future in as_completed(futures):
            try:
                s = future.result()
                if s >= 0:
                    scores.append(s)
            except Exception as e:
                print(f"    [Eval Vote Error]: {e}")

    if not scores:
        return -1.0
    return sum(scores) / len(scores)


def evaluate_batch(
    questions: list[str],
    responses: list[str],
    preference: str,
    evaluator_model: str = "gpt-4.1",
    max_workers: int = 10,
) -> list[float]:
    """
    Score a batch of (question, response) pairs in parallel.

    Returns list of scores. Failed evaluations are -1.0.
    """
    assert len(questions) == len(responses), "questions and responses must have same length"

    scores = [None] * len(questions)

    def _eval_one(idx):
        return idx, evaluate_personalization(
            questions[idx], responses[idx], preference, evaluator_model
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, i): i for i in range(len(questions))}
        for future in as_completed(futures):
            try:
                idx, score = future.result()
                scores[idx] = score
            except Exception as e:
                idx = futures[future]
                scores[idx] = -1.0
                print(f"    [Eval] Question {idx} failed: {e}")

    return scores


def compute_average_score(scores: list[float]) -> float:
    """Compute average, excluding failed evaluations (-1.0)."""
    valid = [s for s in scores if s >= 0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing personalization evaluator...")

    test_q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?"

    # AI-like response (should score low for Student preference)
    test_r_ai = """**Solution:**

1. Total eggs per day: 16
2. Eggs eaten for breakfast: 3
3. Eggs used for muffins: 4
4. Eggs remaining: 16 - 3 - 4 = **9**
5. Revenue: 9 × $2 = **$18**

**Final Answer: $18**"""

    # Natural response (should score high for Student preference)
    test_r_natural = """So she starts with 16 eggs. She eats 3 in the morning and uses 4 for muffins, so that's 7 gone. 16 minus 7 leaves 9 eggs. She sells those at $2 each, so 9 times 2 is 18 dollars a day."""

    print("\n--- Testing Student preference ---")
    score_ai = evaluate_personalization(test_q, test_r_ai, PREFERENCE_STUDENT)
    score_nat = evaluate_personalization(test_q, test_r_natural, PREFERENCE_STUDENT)
    print(f"AI-like response:  {score_ai}")
    print(f"Natural response:  {score_nat}")
    print(f"Expected: AI-like < Natural")
    print(f"Result: {'PASS' if score_ai < score_nat else 'FAIL'}")
