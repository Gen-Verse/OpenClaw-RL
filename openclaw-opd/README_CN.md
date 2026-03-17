# 基于后见之明提示的 On-Policy Distillation (OPD)

用于智能体工具调用的在线蒸馏：利用下一轮反馈提取后见之明提示（Hindsight Hints），构建更强的教师信号，并在策略上训练学生模型。

## 核心流程

对于每个主线路轮次：

1. 使用当前策略服务响应并保留 Rollout 对数概率。
2. 当下一状态到达时（用户回复 / 环境反馈），判断 `(响应, 下一状态)` 的后见之明价值。
3. 运行 `m` 次判断投票；每次投票返回 `+1/-1` 和可选的 Hint。
4. 保留最长的非平凡正向 Hint；如果不存在，则丢弃该样本。
5. 将 Hint 追加到提示词中，在原始响应 Token 上查询教师对数概率。
6. 将训练样本提交到 SLIME。

这将延迟反馈转化为 Token 级监督，无需手工标注轨迹。

## 选项 A（默认）：Token 级 OPD

每个 Token 的教师信号：

$$A_t=\log\pi_{\text{teacher}}(a_t\mid s+\text{hint})-\log\pi_\theta(a_t\mid s)$$

训练使用带有上述 Token 级优势的 PPO 裁剪策略损失，加上 KL 损失：

$$\mathcal{L}=\mathcal{L}_{pg}+\beta_{KL}\mathcal{L}_{KL}$$

默认脚本：

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

## 选项 B：Top-K Logits 蒸馏（SDFT/SDPO 风格）

遵循 [SDFT](https://arxiv.org/abs/2601.19897) 和 [SDPO](https://arxiv.org/abs/2601.20802)，不使用单 Token 教师目标，而是蒸馏每个位置的 Top-K 教师分布。但请注意，我们使用教师 Top-K 而非学生 Top-K（他们原论文中的设置），详见 Issue #7。我们稍后会比较教师 Top-K 和学生 Top-K。

- 教师查询：`input_top_logprobs`（每个位置 `K` 个 Token）。
- 存储字段：`teacher_topk_log_probs [T,K]`，`teacher_topk_indices [T,K]`。
- 损失：在 `K+1` 个桶上的逆向 KL（Top-K + 尾部质量）：

$$D_{KL}\left(\pi_\theta^{K+1}\|\pi_{teacher}^{K+1}\right)=\sum_{k=1}^{K+1}\pi_\theta^{(k)}\left(\log\pi_\theta^{(k)}-\log\pi_{teacher}^{(k)}\right)$$

尾部桶使用：

$$\log p_{tail}=\log\left(1-\exp(\mathrm{logsumexp}(\log p_1,\dots,\log p_K))\right)$$

### 严格兼容性设计

Top-K 实现为可加性扩展：

- 传统 Token 级 OPD 路径保持不变。
- `teacher_log_probs [T]` 对传统路径保持原有语义。
- Top-K 使用独立字段（`teacher_topk_log_probs`、`teacher_topk_indices`）。
- Top-K 损失是外部自定义损失（非内置核心损失开关）。
- Top-K 教师查询默认关闭（`--distill-topk 0`）。

### 如何运行 Top-K

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk.sh
```

等效关键参数：

```bash
--loss-type custom_loss \
--custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function \
--distill-topk 50 \
--disable-compute-advantages-and-returns \
--entropy-coef 0.00
```

## 文件结构

```text
openclaw-opd/
├── README.md
├── README_CN.md                          # 本文档
├── run_qwen3_4b_openclaw_opd.sh          # Token 级 OPD（默认）
├── run_qwen3_4b_openclaw_opd_topk.sh     # Top-K 自定义损失路径
├── topk_distillation_loss.py             # 逆向 KL Top-K 损失（外部自定义损失）
├── openclaw_opd_api_server.py            # 异步判断 + 教师查询 + 样本提交
├── openclaw_opd_rollout.py               # 连接 SLIME 训练器的 Rollout 桥接
└── results/                              # 运行时记录（自动创建）
```
