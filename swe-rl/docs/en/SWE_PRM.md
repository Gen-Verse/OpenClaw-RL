# SWE PRM (Process Reward Model)

SWE RL defaults to outcome reward: resolved → +1, failed → -1. Every step in the trajectory receives the same reward.
PRM scores each step independently, enabling RL training to learn more efficient strategies.

## How It Works

After the agent executes a bash command at each rollout step, PRM **asynchronously** evaluates step quality:

```
Agent generates THOUGHT + bash → Docker exec → returncode + output
                                                    │
                                                    └─ SweRewardAgent.submit_step_judge() (non-blocking)
                                                         ├─ Build PRM prompt
                                                         ├─ m parallel samples to PRM LLM
                                                         └─ Vote: +1 (good step) / -1 (bad step)
```

After all steps complete, PRM scores are collected and combined with outcome reward:

```
final_score = outcome_reward + prm_step_coef × mean(step_prm_scores)
```

## PRM Prompt

### System

```
You are a strict evaluator for a software engineering agent that fixes GitHub issues.
You are given:
1) The issue description (problem statement).
2) The agent's recent action history.
3) The agent's most recent step (THOUGHT + bash command) to evaluate.
4) The execution result of that command (returncode + stdout/stderr).
```

### User

```
## Issue Description
{problem_statement}

## Recent History ({n_history} steps)
Step 1: $ find /testbed -name "models.py"
  returncode=0
  output: /testbed/moto/awslambda/models.py...

## Current Step to Evaluate (step 3)

Agent's full response:
{policy_response}

Execution result (returncode={returncode}):
{command_output}
```

### Scoring Criteria

**+1** (all must hold):
- Command executed without unexpected errors
- Step clearly advances issue diagnosis or fix
- Output provides useful information / edit is logically correct

**-1** (any holds):
- Command fails (wrong path, syntax error)
- Step is irrelevant to the issue
- Repeating a previously failed approach
- Edit introduces a bug
- Wasting steps (e.g. re-reading already examined files)

PRM output format: chain-of-thought reasoning + `\boxed{+1}` or `\boxed{-1}`.

## m-voting

Each step is sampled m times in parallel (default m=3, different seeds), extracting the score from `\boxed{}`:

```
valid_votes = [v for v in votes if v in (-1, +1)]
step_score  = mean(valid_votes) if valid_votes else 0.0
```

Range [-1, 1]. Invalid outputs (parse failures) score 0 and are excluded from the mean.

## Output Truncation

Each step's bash output is saved as `output_head` (first 2000 chars) and `output_tail` (last 2000 chars).
PRM uses `_format_output()` to deduplicate and combine:

| Total output length | Handling |
|---------------------|----------|
| ≤ 2000 | Use head directly |
| 2001 – 3999 | head + tail (remove overlap) |
| ≥ 4000 | head + `... (N chars omitted) ...` + tail |

Current step output capped at `max_output_len=4000` chars; history step output capped at `max_history_output_len=1000` chars.

## Reward Calculation

### Total Reward

```
final_score = outcome_reward + prm_step_coef × mean(all step_prm_scores)
```

- `outcome_reward`: resolved → +1, not resolved → -1
- `prm_step_coef`: default 1.0

### Step-wise Advantage

Per-step combined reward:
```
step_reward[k] = step_prm_score[k] + outcome_reward
```

The framework's `_post_process_step_wise_rewards()` applies z-score normalisation within each `(group_index, step_index)` bucket.

## Metadata Structure

### Default Path

```python
sample.metadata["prm"] = {
    "enabled": True,
    "step_scores": [0.33, -1.0, 1.0, ...],
    "step_mean_score": 0.11,
    "step_details": [{...}, ...],
}

sample.metadata["step_wise"] = {
    "step_scores": [0.33, -1.0, 1.0],
    "step_indices": [0, 1, 2],
    "step_token_spans": [[start0, end0], ...],
    "step_scores_with_outcome": [1.33, 0.0, 2.0],
    "outcome_reward": 1.0,
}
```

### dynamic_history Path

Each dynamic sample contains only its own step:

```python
child.metadata["step_wise"] = {
    "step_scores": [prm_score],
    "step_indices": [step_idx],
    "step_token_spans": [[0, len(response_ids)]],
    "step_scores_with_outcome": [prm_score + outcome_reward],
    "outcome_reward": outcome_reward,
}
```

## How to Enable

```bash
PRM_ARGS=(
  --prm-enable
  --prm-model-path /path/to/prm_model
  --prm-m 3                             # votes per step
  --prm-num-gpus 8
  --prm-num-gpus-per-engine 8
  --prm-step-coef 1.0                   # PRM reward weight
  --prm-temperature 1.0
  --prm-max-new-tokens 4096
)

GRPO_ARGS=(
  --advantage-estimator step_wise        # step_wise advantage
  --dynamic_history                      # one sample per step
)
```

### SWE-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--swe-prm-max-history-steps` | 8 | Max history steps visible to PRM |
| `--swe-prm-max-problem-len` | 8000 | Max issue description chars |
| `--swe-prm-max-output-len` | 4000 | Max current step output chars |
| `--swe-prm-max-history-output-len` | 1000 | Max history step output chars |
| `--swe-prm-skip-submit` | True | Skip evaluation for submit steps |

## Example

From a 20-step failed trajectory:

**step 0 (+1)**: `find /testbed -type f -exec grep -l 'delete_layer_version' {} \;` → successfully located relevant files

**step 3-7 (-1)**: `nl -ba responses.py | sed -n '101,150p'` for 5 consecutive steps browsing the same file 50 lines at a time → should use grep to locate directly

**step 12 (+1)**: `sed -i 's/layer_name = self.path.../...' responses.py` → correctly modified the ARN parsing logic

## Related Files

| File | Role |
|------|------|
| `swe_prm.py` | `SweRewardAgent` class: prompt building, PRM requests, m-voting, result collection |
| `generate_with_swe_remote.py` | PRM agent init, per-step async dispatch, metadata writing, reward_func composition |
