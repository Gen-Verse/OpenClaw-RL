# OpenClaw-RL -> 可自我进化 Coding Agent 方案（基于 slime）

## 1. 目标定义：从“会对话”到“会持续变强”

基于当前仓库，建议把目标拆成三层：

1. **可用层（Agent Runtime）**：能稳定完成代码修改、测试、提交、解释。
2. **可学习层（RL Loop）**：把真实开发过程中的反馈（测试结果、lint、review、回归）转为训练信号。
3. **可进化层（Auto Curriculum）**：自动发现薄弱能力、构造新任务、持续刷新训练分布。

OpenClaw-RL + slime 已经具备第 2 层的关键组件，重点是把 coding domain 的“环境状态、奖励、动作空间”标准化。

---

## 2. 从现有项目可提取的关键 insight

## 2.1 异步四段式天然适合 coding workflow

OpenClaw-RL 在 README 中明确了“服务、rollout、judge、训练”解耦的异步设计。这个结构和 coding agent 的真实流程高度同构：

- 服务：响应用户 coding 请求；
- rollout：在沙盒中执行改代码/跑命令；
- judge：由测试、静态分析、审查模型组成；
- 训练：把轨迹写回策略更新。

**创新点**：把“开发者日常行为日志”直接映射为 RL 轨迹，而不是人工构造离线数据集。

## 2.2 “下一状态反馈”比静态标签更贴近工程真实价值

仓库强调用 next-state（下一轮用户/工具/环境反馈）作为训练信号。这对 coding 特别关键：

- patch 是否正确，通常要在后续测试/运行中才揭晓；
- 一次成功修改的长期价值要看后续回归是否稳定。

**创新点**：把“短期通过 + 长期不回归”做成分层奖励，避免 agent 追求一次性过测的投机策略。

## 2.3 slime 的 data buffer + rollout 自定义接口是“任务工厂”

slime README 提到其 data buffer 与自定义生成接口可支持任意数据生成流程。用于 coding 时可做：

- 将 issue / TODO / failing test 自动变为 prompt；
- 将多仓库、多语言任务统一为同一轨迹 schema；
- 支持 server-based evaluator（编译器、测试器、安全扫描器）并行评分。

**创新点**：把 coding benchmark 从“固定题库”升级为“在线任务流”，实现 continuously-on policy training。

---

## 3. 可自我进化 coding agent 的参考架构

## 3.1 五层架构

1. **Interaction Layer**：IDE 插件、CLI、PR Bot、Issue Bot。
2. **Execution Layer**：隔离沙盒（容器/微 VM）+ 工具调用网关（git、pytest、lint、build）。
3. **Evaluation Layer**：
   - 硬指标：测试通过率、构建成功率、时延、token 成本；
   - 软指标：可读性、最小改动原则、架构一致性。
4. **Training Layer（slime）**：异步 rollout、buffer 聚合、策略更新。
5. **Evolution Layer**：自动课程生成 + 失败模式聚类 + 策略路由。

## 3.2 状态/动作/奖励定义（建议）

- **状态 S**：仓库快照、报错日志、历史对话、工具输出摘要、文件级依赖图。
- **动作 A**：编辑、执行命令、检索文档、请求澄清、回滚、提交。
- **奖励 R**：
  - `R_pass`：测试/构建通过；
  - `R_quality`：lint、复杂度、duplication 改善；
  - `R_safety`：无密钥泄露、无高危依赖；
  - `R_human`：人工偏好反馈（accept/reject/edit distance）。

建议使用组合奖励：

`R_total = w1*R_pass + w2*R_quality + w3*R_safety + w4*R_human - w5*R_cost`

---

## 4. 基于 slime 的工程优化建议（可落地）

## 4.1 统一 rollout 事件协议（高优先级）

在 rollout 数据结构中新增 coding 专用字段：

- `repo_id`, `commit_base`, `changed_files`, `commands_run`, `tests_run`, `artifact_uri`, `sandbox_profile`。

收益：后续 reward 复算、失败复盘、策略对比更容易。

## 4.2 引入“多评审器并行投票”

沿用 OpenClaw-RL 的 judge voting 思路，扩展为：

- 单元测试评审器；
- 静态检查评审器；
- LLM 代码审查评审器；
- 安全审查评审器。

对冲单一评估器噪声，降低 reward hacking。

## 4.3 在 data buffer 增加“失败片段重放池”

将失败轨迹按错误类型聚类（编译错误、依赖错误、语义错误、回归错误），提高训练采样概率。

收益：快速补齐短板能力，形成 self-healing learning loop。

## 4.4 策略路由：Planner / Editor / Verifier 三头专家

不用一个模型做所有动作，可在 slime 里做角色化路由：

- Planner：拆任务与生成子目标；
- Editor：产出 patch；
- Verifier：决定下一步是测试、回滚还是提交。

收益：降低长链路任务上的漂移与幻觉。

## 4.5 成本感知训练

把 token 与执行时间纳入负奖励，并统计每类任务的“成功成本曲线”。

收益：不仅会做，还会“便宜地做对”。

---

## 5. 三阶段实施路线图

## Phase 1（1~2 周）— MVP 闭环

- 接入最小 coding 环境（编辑 + pytest + git diff）；
- 建立轨迹 schema 与 reward 计算；
- 用 Binary RL 跑通“失败测试 -> 修复 -> 通过”。

验收指标：

- 任务通过率相对基线提升；
- 平均迭代步数下降；
- 无明显 reward hacking。

## Phase 2（2~4 周）— 混合信号增强

- 接入 OPD（文本 hint）；
- 多评审器投票；
- 失败重放池。

验收指标：

- 长链路任务（>5 steps）成功率提升；
- 回归率下降；
- 人工改写量下降（edit distance）。

## Phase 3（持续）— 自进化

- 自动课程生成（从真实失败样本反推训练任务）；
- 难度自适应采样；
- 多仓库迁移评估。

验收指标：

- 新仓库冷启动成功率持续上升；
- 每周 reward/成本比改善；
- 人工干预频次下降。

---

## 6. 可对外叙述的创新点（论文/项目介绍可用）

1. **Live Coding RL**：在线真实开发流中的持续学习，而非静态 benchmark 训练。
2. **Next-State Rewarding for SWE**：通过后续环境反馈定义过程奖励，解决代码任务延迟奖励难题。
3. **Asynchronous Multi-Judge Alignment**：并行评审器投票，兼顾正确性、可维护性与安全性。
4. **Failure Replay Evolution**：以失败模式为课程引擎，驱动 agent 自我进化。
5. **Cost-Aware Agent Optimization**：把成功率与工程成本共同纳入优化目标。

---

## 7. 建议优先改造的仓库模块

- `swe-rl/`：最接近 coding agent 场景，可作为首个在线训练闭环。
- `slime/slime/rollout/` 与 `slime/slime_plugins/rollout_buffer/`：承接轨迹协议与失败重放池。
- `openclaw-combine/`：适合承载 Binary RL + OPD 的混合训练配方。
- `extensions/rl-training-headers/`：可作为 IDE / 用户反馈注入入口。

---

## 8. 一句话结论

这个项目已经具备“自我进化 coding agent”的核心土壤：异步训练架构、可插拔 rollout、混合奖励路径。下一步不是重写框架，而是把 coding 场景的状态/奖励协议产品化，并在 slime 中工程化落地为可持续运行的学习闭环。
