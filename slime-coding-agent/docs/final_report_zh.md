# 最终报告：基于 slime + sglang 的可自我进化 Coding Agent

## 一、项目重构结论

针对“需要部署、使用、SWE-bench 跑测、消融验证”的要求，本次将原先偏文档化方案升级为**可执行项目框架**：

- 提供部署模板（slime + sglang）；
- 提供 agent 运行入口（runner）；
- 提供 SWE-bench 评测入口（eval）；
- 提供消融实验入口（ablation）。

## 二、当前可执行能力

1. **部署层**：`deploy/compose_slime_sglang.yaml` 定义训练服务、rollout 服务与 runner 依赖。
2. **运行层**：`runner/coding_agent_runner.py` 输出标准轨迹事件。
3. **评测层**：`eval/swebench_runner.py` 产出指标 json。
4. **实验层**：`eval/ablation_runner.py` 读取实验矩阵并输出对比结果。

## 三、实验设计（建议）

- 主指标：Resolve Rate、Pass@1、平均 token 成本、平均步骤数。
- 消融维度：
  - 是否启用多评审器；
  - 是否启用失败重放；
  - 是否启用成本惩罚。

## 四、下一步接入真实系统

1. 将 runner 的 mock executor 替换为真实 sandbox + git + pytest 工具链。
2. 将 swebench_runner 接到 `swe-rl` 的真实执行管线。
3. 将 ablation runner 映射到 slime 真实训练参数（ray job submit / train.py 参数组合）。
4. 增加结果追踪看板（W&B 或本地 dashboard）。

## 五、验收标准

- 能完成单任务端到端闭环并生成事件日志；
- 能完成 SWE-bench 批量评测并导出指标；
- 能完成至少 4 组消融并输出可对比结果。
