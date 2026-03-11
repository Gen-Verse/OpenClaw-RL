# SWE PRM（Process Reward Model）

SWE RL 默认使用 outcome reward：resolved → +1，failed → -1。整条轨迹所有步骤拿到相同的奖励。
PRM 对每一步独立评分，让 RL 训练学到更高效的策略。

## 工作原理

每步 rollout 中 agent 执行 bash 命令后，PRM **异步**评估该步质量：

```
Agent 生成 THOUGHT + bash → Docker 执行 → returncode + output
                                              │
                                              └─ SweRewardAgent.submit_step_judge()（不阻塞 rollout）
                                                   ├─ 构造 PRM prompt
                                                   ├─ m 次并行采样到 PRM LLM
                                                   └─ 投票：+1（好步骤）/ -1（差步骤）
```

所有步骤完成后，收集 PRM 分数，与 outcome reward 组合：

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

### 评分标准

**+1**（以下全部满足）：
- 命令执行无意外错误
- 步骤明确推进了问题诊断或修复
- 输出提供了有用信息 / 编辑逻辑正确

**-1**（以下任一满足）：
- 命令失败（路径错误、语法错误）
- 步骤与 issue 无关
- 重复之前失败的做法
- 编辑引入 bug
- 浪费步骤（如重复读已看过的文件）

PRM 输出格式：思考过程 + `\boxed{+1}` 或 `\boxed{-1}`。

## m-voting

每步 m 次并行采样（默认 m=3，不同 seed），提取 `\boxed{}` 中的分数：

```
valid_votes = [v for v in votes if v in (-1, +1)]
step_score  = mean(valid_votes) if valid_votes else 0.0
```

范围 [-1, 1]。无效输出（解析失败）得 0，不计入均值。

## 输出截断处理

每步的 bash 输出保存为 `output_head`（前 2000 字符）和 `output_tail`（后 2000 字符）。
PRM 使用 `_format_output()` 去重拼接：

| output 总长 | 处理方式 |
|-------------|---------|
| ≤ 2000 | 直接使用 head |
| 2001 – 3999 | head + tail（去重叠部分） |
| ≥ 4000 | head + `... (N chars omitted) ...` + tail |

当前步输出限制 `max_output_len=4000` 字符，历史步输出限制 `max_history_output_len=1000` 字符。

## 奖励计算

### 总奖励

```
final_score = outcome_reward + prm_step_coef × mean(all step_prm_scores)
```

- `outcome_reward`：resolved → +1，not resolved → -1
- `prm_step_coef`：默认 1.0

### step_wise advantage

每步的组合奖励：
```
step_reward[k] = step_prm_score[k] + outcome_reward
```

框架的 `_post_process_step_wise_rewards()` 对同一 `(group_index, step_index)` 桶做 z-score 归一化。

## Metadata 结构

### 默认路径

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

### dynamic_history 路径

每个动态样本只包含自己那一步：

```python
child.metadata["step_wise"] = {
    "step_scores": [prm_score],
    "step_indices": [step_idx],
    "step_token_spans": [[0, len(response_ids)]],
    "step_scores_with_outcome": [prm_score + outcome_reward],
    "outcome_reward": outcome_reward,
}
```

## 启用方式

```bash
PRM_ARGS=(
  --prm-enable
  --prm-model-path /path/to/prm_model
  --prm-m 3                             # 每步投票次数
  --prm-num-gpus 8
  --prm-num-gpus-per-engine 8
  --prm-step-coef 1.0                   # PRM 奖励权重
  --prm-temperature 1.0
  --prm-max-new-tokens 4096
)

GRPO_ARGS=(
  --advantage-estimator step_wise        # step_wise advantage
  --dynamic_history                      # 每步一个样本
)
```

### SWE 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--swe-prm-max-history-steps` | 8 | PRM 可见的历史步数 |
| `--swe-prm-max-problem-len` | 8000 | issue 描述最大字符数 |
| `--swe-prm-max-output-len` | 4000 | 当前步输出最大字符数 |
| `--swe-prm-max-history-output-len` | 1000 | 历史步输出最大字符数 |
| `--swe-prm-skip-submit` | True | 跳过 submit 步的评估 |

## 示例

以一个 20 步失败轨迹为例：

**step 0（+1）**：`find /testbed -type f -exec grep -l 'delete_layer_version' {} \;` → 成功定位相关文件

**step 3-7（-1）**：`nl -ba responses.py | sed -n '101,150p'` 连续 5 步逐段 50 行浏览同一文件 → 应该用 grep 直接定位

**step 12（+1）**：`sed -i 's/layer_name = self.path.../...' responses.py` → 正确修改了 ARN 解析逻辑

## 相关文件

| 文件 | 职责 |
|------|------|
| `swe_prm.py` | `SweRewardAgent` 类：prompt 构造、PRM 请求、m-voting、结果收集 |
| `generate_with_swe_remote.py` | 初始化 PRM agent、每步异步派发、写入 metadata、reward_func 组合 |
