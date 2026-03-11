# 上下文管理

SWE RL 的多轮 agent rollout 会不断累积消息——每步包含 assistant 回复（THOUGHT + bash）和 user 观测（returncode + stdout）。
超出模型上下文窗口后推理会失败。`swe_context_manager.py` 实现了自动截断机制。

## Head + Tail 策略

当消息总 token 数超出预算时，保留两端、丢弃中间：

```
[system, problem]                          ← 始终保留
[turn_0, turn_1]                           ← head（30% 预算）：早期探索
[... 10 turn(s) omitted ...]              ← 省略标记
[turn_12, turn_13, ..., turn_19]           ← tail（70% 预算）：近期工作
```

- **操作单位**：消息对 `(assistant, user)`——整对保留或整对丢弃，语义完整
- **触发条件**：仅当总 token 数 > `max_input_tokens` 时激活，短 rollout 不受影响
- **省略标记**：`[... {n} turn(s) of interaction history omitted due to context window limit ...]`，以 `user` 角色插入

## Token 预算计算

```
rollout_max_context_len = 16384
max_new_tokens          = 4096
max_input_tokens        = 16384 - 4096 = 12288

fixed_cost  = tokens(system + problem + 省略标记) ≈ 500-800
available   = 12288 - fixed_cost ≈ 11500
head_budget = available × 0.3 ≈ 3450
tail_budget = available × 0.7 ≈ 8050
```

## 贪心填充算法

`get_context_messages()` 的核心逻辑：

1. 如果全部消息 ≤ `max_input_tokens`，直接返回原始消息
2. 将 `messages[2:]` 拆分成 turn pairs（每对是 `[assistant, user]`，最后一对可能只有 assistant）
3. 计算固定开销：`system + problem + 省略占位符`
4. **从前往后**填充 head：尽可能多地把完整 turn pair 放入 head_budget
5. **从后往前**填充 tail：尽可能多地把完整 turn pair 放入 tail_budget（不与 head 重叠）
6. 若有被省略的中间 turns，插入省略标记；否则返回原始消息

## 训练对齐（dynamic_history）

每步 rollout 使用的截断后上下文被记录在 `managed_contexts` 列表中。
启用 `--dynamic_history` 后，训练样本构建改为**每步一个样本**：

```
Step i 训练样本:
  prompt  = managed_contexts[i]   ← 模型在 rollout 时实际看到的上下文
  response = assistant_texts[i]    ← 模型实际生成的内容
  loss_mask = 1 仅在 response tokens 上
```

**训练与 rollout 完全对齐**——模型在训练时看到的上下文与 rollout 时完全一致。

不启用 `--dynamic_history` 时，使用默认的单样本路径：完整消息拼接后尾部截断至 `rollout_max_context_len`。

## 启用方式

```bash
ROLLOUT_ARGS=(
  --rollout-max-response-len 4096      # 每次 LLM 生成的最大 token 数
  --rollout-max-context-len 16384      # 总上下文窗口（自动触发 CM）
)

GRPO_ARGS=(
  --dynamic_history                    # 每步一个训练样本，与 CM 对齐
)
```

可选调优：`--swe-cm-head-ratio 0.3`（head 占预算比例，默认 0.3，即 30% head / 70% tail）。

## 相关文件

| 文件 | 职责 |
|------|------|
| `swe_context_manager.py` | 核心逻辑：`get_context_messages()` |
| `generate_with_swe_remote.py` | Rollout 集成 + dynamic_history 样本构建 |
