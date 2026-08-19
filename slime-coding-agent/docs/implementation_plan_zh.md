# 项目搭建思路（分阶段）

## Phase 1（MVP，1-2 周）

目标：建立最小闭环。

- 定义统一 rollout 事件协议；
- 打通 edit -> test -> reward -> sample 入库；
- 支持最基础二值奖励。

交付物：

- `configs/rollout_event_schema.yaml`
- 单仓库任务脚本
- 每日成功率/成本报表

## Phase 2（增强，2-4 周）

目标：提升稳定性与泛化。

- 多评审器投票（test/lint/LLM review/security）；
- 失败重放池 + 错误类型采样；
- 三角色策略路由（planner/editor/verifier）。

交付物：

- 失败聚类仪表盘
- 任务难度分层评估
- 人工 review 接受率统计

## Phase 3（自进化，持续）

目标：自动课程与跨仓库迁移。

- 从真实失败样本反推训练任务模板；
- 难度自适应采样；
- 新仓冷启动自动评估。

交付物：

- 自动课程生成器
- 周度 reward/cost 趋势看板
- 跨仓库能力基线对比
