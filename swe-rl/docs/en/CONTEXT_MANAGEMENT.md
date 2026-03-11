# Context Management

SWE RL's multi-turn agent rollouts continuously accumulate messages — each step adds an assistant reply (THOUGHT + bash) and a user observation (returncode + stdout).
Once the model's context window is exceeded, inference fails. `swe_context_manager.py` implements automatic truncation.

## Head + Tail Strategy

When total message tokens exceed the budget, keep both ends and drop the middle:

```
[system, problem]                          ← always kept
[turn_0, turn_1]                           ← head (30% of budget): early exploration
[... 10 turn(s) omitted ...]              ← omission marker
[turn_12, turn_13, ..., turn_19]           ← tail (70% of budget): recent work
```

- **Unit of operation**: message pairs `(assistant, user)` — kept or dropped as a whole, preserving semantic completeness
- **Trigger**: only activates when total tokens > `max_input_tokens`; short rollouts are unaffected
- **Omission marker**: `[... {n} turn(s) of interaction history omitted due to context window limit ...]`, inserted with `user` role

## Token Budget Calculation

```
rollout_max_context_len = 16384
max_new_tokens          = 4096
max_input_tokens        = 16384 - 4096 = 12288

fixed_cost  = tokens(system + problem + omission marker) ≈ 500-800
available   = 12288 - fixed_cost ≈ 11500
head_budget = available × 0.3 ≈ 3450
tail_budget = available × 0.7 ≈ 8050
```

## Greedy Fill Algorithm

Core logic of `get_context_messages()`:

1. If all messages fit within `max_input_tokens`, return the original messages
2. Split `messages[2:]` into turn pairs (each pair is `[assistant, user]`; the last pair may be assistant-only)
3. Compute fixed cost: `system + problem + omission placeholder`
4. **Front-to-back** fill head: pack as many complete turn pairs as possible into head_budget
5. **Back-to-front** fill tail: pack as many complete turn pairs as possible into tail_budget (no overlap with head)
6. If middle turns were omitted, insert the omission marker; otherwise return the original messages

## Training Alignment (dynamic_history)

Each rollout step's truncated context is recorded in the `managed_contexts` list.
With `--dynamic_history` enabled, training sample construction becomes **one sample per step**:

```
Step i training sample:
  prompt   = managed_contexts[i]   ← what the model actually saw during rollout
  response = assistant_texts[i]    ← what the model actually generated
  loss_mask = 1 only on response tokens
```

**Training and rollout are exactly aligned** — the model trains on the same (possibly truncated) context it saw during rollout.

Without `--dynamic_history`, the default single-sample path is used: full messages concatenated then tail-truncated to `rollout_max_context_len`.

## How to Enable

```bash
ROLLOUT_ARGS=(
  --rollout-max-response-len 4096      # max tokens per LLM generation
  --rollout-max-context-len 16384      # total context window (auto-triggers CM)
)

GRPO_ARGS=(
  --dynamic_history                    # one training sample per step, aligned with CM
)
```

Optional tuning: `--swe-cm-head-ratio 0.3` (fraction of budget for head turns; default 0.3, i.e. 30% head / 70% tail).

## Related Files

| File | Role |
|------|------|
| `swe_context_manager.py` | Core logic: `get_context_messages()` |
| `generate_with_swe_remote.py` | Rollout integration + dynamic_history sample building |
