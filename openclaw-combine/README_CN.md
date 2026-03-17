# Binary RL 与 On-Policy Distillation 组合方法

*取长补短，相得益彰。*

该方法同时运行 Binary RL (GRPO) 和 On-Policy Distillation (OPD)，将评估性信号和方向性信号融合到统一的训练目标中。在我们的实验中，这种组合方法相比单独使用任一方法取得了显著的性能提升。

## 为什么要组合？

| 维度 | Binary RL | OPD | 组合方法 |
|---|---|---|---|
| 信号类型 | 评估性（好/坏） | 方向性 | 评估性 + 方向性 |
| 优势估计 | 序列级标量 | Token 级方向性 | 混合序列级和 Token 级 |
| 密度 | 所有评分轮次 | 仅接受 Hint 的轮次 | 所有评分轮次 |
| 反馈类型 | 用户 / 环境 | 显式修正 | 隐式和显式反馈均可 |
| 信号丰富度 | 每样本 1 个标量 | 每个 Token 1 个值 | 每个 Token 1 个值 |

Binary RL 接受每一个被评分的轮次，无需提取 Hint，可以处理任何下一状态信号——包括简短、隐式的反应（用户只是重新提问）或结构化的环境输出（退出码、测试结果）。OPD 则应在交互流可能包含丰富指导性内容时额外启用：用户给出明确修正（"不要用那个库"、"先检查文件"），或环境产生详细错误信息（SWE diff、编译器诊断）。

实践中，Binary RL 为所有轮次提供广泛的梯度覆盖，而 OPD 为有指导性信号的轮次子集提供高分辨率的逐 Token 修正。

## 组合优势估计

两个分支共享相同的 PPO 裁剪代理损失——只有优势计算不同。组合优势为：

$$A_t = w_{\text{binary}} \, r_{\text{final}} + w_{\text{opd}} \left( \log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t) \right)$$

默认 $w_{\text{binary}} = w_{\text{opd}} = 1$。OPD 样本带有 `reward=0`，因此其 GRPO 优势为零；RL 样本带有 `teacher_logp ≈ rollout_logp`，因此其教师优势约等于零。每个分支自然地主导其对应的样本类型，组合优势就是两者的简单求和。

## 逐轮处理流程

对于每个主线路轮次，当下一状态到达后：

1. 并行运行 `m` 次 Hint 判断投票和 `m` 次评估投票。
2. 如果 Hint 被接受（最长的非平凡正向 Hint），生成一个 **OPD** 样本并附带教师对数概率。
3. 如果评估得分为 `+1` 或 `−1`，生成一个 **RL** 样本并附带标量奖励。
4. 单个轮次可以同时贡献两种样本类型。
5. 当收集的样本数达到 `rollout_batch_size` 时触发训练批次。

## 如何运行

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh
```

### 关键环境变量

| 变量 | 默认值 | 描述 |
|---|---|---|
| `OPENCLAW_COMBINE_W_RL` | `1.0` | GRPO 优势的权重 $w_{\text{binary}}$ |
| `OPENCLAW_COMBINE_W_OPD` | `1.0` | 教师优势的权重 $w_{\text{opd}}$ |
| `PRM_M` | `1` | 每轮独立判断/评估投票的次数 |

所有其他变量（`NUM_GPUS`、`ACTOR_GPUS`、`HF_CKPT` 等）与 Binary RL 和 OPD 脚本共享——完整列表请参阅[主 README](../README.md)。

## 文件结构

```text
openclaw-combine/
├── README.md
├── README_CN.md                          # 本文档
├── run_qwen3_4b_openclaw_combine.sh      # 启动脚本
├── openclaw_combine_api_server.py        # 异步代理：Hint 判断 + PRM 评估 + 样本提交
├── openclaw_combine_rollout.py           # 连接 SLIME 训练器的 Rollout 桥接
├── combine_loss.py                       # 加权优势：w_rl * GRPO + w_opd * teacher
└── results/                              # 运行时记录（自动创建）
```
