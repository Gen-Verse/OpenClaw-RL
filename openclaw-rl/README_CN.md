# 基于下一状态的二元奖励信号

用于智能体工具调用的在线 RL，使用来自环境反馈的二元过程奖励信号。

## 方法概述

策略模型部署为 OpenAI 兼容的聊天代理。外部环境（如 OpenClaw）通过该代理发送多轮对话。对于每个**主线路轮次**，系统：

1. 将请求转发给策略模型（由 SGLang 服务）并收集响应及逐 Token 对数概率。
2. 当**下一轮**到达时，其用户/环境消息作为上一轮的"下一状态"。
3. **过程奖励模型 (PRM)** 根据下一状态（可以是用户或环境反馈）判断上一轮响应的质量。它通过多数投票进行 `m` 次独立评估，将每轮评分 `+1`（好）、`-1`（坏）或 `0`（中性）。
4. 多数投票的分数成为该轮的标量奖励。
5. 从未收到下一状态的轮次（即会话中的最后一轮）被排除在训练之外（`loss_mask = 0`），除非它们是会话中唯一的轮次（至少一个保证）。

### 优势估计 (GRPO)

使用 **Group Relative Policy Optimization (GRPO)** 计算优势。对于每个标量奖励为 `r` 的样本，优势均匀广播到所有响应 Token：

$$A_t = r, \quad \forall t \in \text{响应 Token}$$

不应用奖励归一化（`--disable-rewards-normalization`）。

### 策略梯度损失

标准的 PPO 裁剪代理目标，使用非对称裁剪：

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\!\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

其中 $\varepsilon = 0.2$，$\varepsilon_{\text{high}} = 0.28$。

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

其中 $\beta_{\text{KL}} = 0.02$。熵奖励已禁用（$\beta_{\text{ent}} = 0$）。

## 如何运行

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

## 文件结构

```
openclaw-rl/
├── README.md
├── README_CN.md                      # 本文档
├── run_qwen3_4b_openclaw_rl.sh       # 启动脚本
├── openclaw_api_server.py            # FastAPI 代理 + PRM 评分 + 样本提交
├── openclaw_rollout.py               # 异步 Rollout 工作器（连接 API 服务器 ↔ SLIME 训练器）
└── results/                          # 运行时记录（自动创建）
```
