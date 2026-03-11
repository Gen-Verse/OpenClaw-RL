"""Context management for SWE RL rollout and training.

Prevents context window overflow during multi-turn agent rollouts by
keeping a head+tail subset of turn pairs when the full messages exceed
the token budget.  The omitted middle is replaced by a single user-role
marker so that the chat structure stays valid.

Usage (rollout):
    from swe_context_manager import get_context_messages

    ctx = get_context_messages(messages, tokenizer, max_input_tokens)
    resp = await acompletion(model=..., messages=ctx, ...)
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

OMIT_TEMPLATE = "[... {n} turn(s) of interaction history omitted due to context window limit ...]"


def _count_tokens(messages: List[dict], tokenizer: Any) -> int:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def _count_tokens_for_turn(turn: List[dict], tokenizer: Any) -> int:
    """Approximate token cost of a single turn pair.

    We wrap in a minimal conversation so that the chat template produces
    the correct role headers / delimiters, then subtract the wrapper cost
    to isolate the turn's own tokens.
    """
    wrapper = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    base = _count_tokens(wrapper, tokenizer)
    full = _count_tokens(wrapper + turn, tokenizer)
    return max(0, full - base)


def _split_into_turn_pairs(messages: List[dict]) -> List[List[dict]]:
    """Split messages[2:] into (assistant, user) pairs.

    The last turn may contain only an assistant message (e.g. a submit
    action with no subsequent observation).
    """
    body = messages[2:]
    turns: List[List[dict]] = []
    i = 0
    while i < len(body):
        if i + 1 < len(body) and body[i]["role"] == "assistant" and body[i + 1]["role"] == "user":
            turns.append([body[i], body[i + 1]])
            i += 2
        else:
            turns.append([body[i]])
            i += 1
    return turns


def get_context_messages(
    messages: List[dict],
    tokenizer: Any,
    max_input_tokens: int,
    head_ratio: float = 0.3,
) -> List[dict]:
    """Return a context-managed copy of *messages*.

    If the full messages fit within *max_input_tokens* the original list
    is returned unchanged.  Otherwise a head (earliest turns) + tail
    (most recent turns) subset is selected so that the total token count
    stays within budget, with a ``[... N turn(s) omitted ...]`` marker
    inserted between head and tail.

    Parameters
    ----------
    messages:
        Full conversation ``[system, problem, (assistant, user)*, ...]``.
    tokenizer:
        HuggingFace-compatible tokenizer with ``apply_chat_template``
        and ``encode``.
    max_input_tokens:
        Maximum number of input tokens (= context_limit - max_new_tokens).
    head_ratio:
        Fraction of the *available* token budget allocated to head turns
        (default 0.3 → 30 % head, 70 % tail).
    """
    total_tokens = _count_tokens(messages, tokenizer)
    if total_tokens <= max_input_tokens:
        return messages

    fixed = messages[:2]
    turns = _split_into_turn_pairs(messages)

    if len(turns) <= 1:
        return messages

    omit_placeholder = [{"role": "user", "content": OMIT_TEMPLATE.format(n=0)}]
    fixed_tokens = _count_tokens(fixed + omit_placeholder, tokenizer)
    available = max_input_tokens - fixed_tokens
    if available <= 0:
        logger.warning(
            "Context budget exhausted by fixed messages alone "
            "(fixed=%d, budget=%d). Returning fixed + omit only.",
            fixed_tokens, max_input_tokens,
        )
        return fixed + [{"role": "user", "content": OMIT_TEMPLATE.format(n=len(turns))}]

    head_budget = int(available * head_ratio)
    tail_budget = available - head_budget

    # --- greedy fill from the front (head) --------------------------------
    head_end = 0
    used_head = 0
    for i, turn in enumerate(turns):
        cost = _count_tokens_for_turn(turn, tokenizer)
        if used_head + cost <= head_budget:
            used_head += cost
            head_end = i + 1
        else:
            break

    # --- greedy fill from the back (tail) ---------------------------------
    tail_start = len(turns)
    used_tail = 0
    for i in range(len(turns) - 1, -1, -1):
        if i < head_end:
            break
        cost = _count_tokens_for_turn(turns[i], tokenizer)
        if used_tail + cost <= tail_budget:
            used_tail += cost
            tail_start = i
        else:
            break

    head_turns = turns[:head_end]
    tail_turns = turns[tail_start:]
    n_omitted = tail_start - head_end

    if n_omitted <= 0:
        return messages

    omit_msg = {"role": "user", "content": OMIT_TEMPLATE.format(n=n_omitted)}

    result: List[dict] = list(fixed)
    for turn in head_turns:
        result.extend(turn)
    result.append(omit_msg)
    for turn in tail_turns:
        result.extend(turn)

    managed_tokens = _count_tokens(result, tokenizer)
    logger.info(
        "Context management: %d tokens -> %d tokens "
        "(head=%d tail=%d omitted=%d turns, budget=%d)",
        total_tokens, managed_tokens,
        len(head_turns), len(tail_turns), n_omitted, max_input_tokens,
    )
    return result
