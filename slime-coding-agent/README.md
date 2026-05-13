# Slime Coding Agent

基于 **slime + sglang** 的可自我进化 Coding Agent 项目框架，覆盖：

- 模型部署（slime 训练 + sglang rollout serving）
- Agent 使用（任务执行、工具调用、验证循环）
- SWE-bench 跑测（批量评测与指标汇总）
- 消融实验（奖励、评审器、失败重放等）

## 1. 项目目标

本子项目面向“工程可落地”，重点不是写概念文档，而是建立可扩展框架：

1. **Deploy**：可配置地拉起 slime/sglang 训练-推理闭环。
2. **Run**：以 planner/editor/verifier 三角色跑 coding 任务。
3. **Eval**：对接 SWE-bench 批测与成功率、成本统计。
4. **Ablation**：用统一配置切换组件验证增益。

## 2. 目录结构

```text
slime-coding-agent/
  deploy/
    compose_slime_sglang.yaml      # 部署参数模板
  runner/
    coding_agent_runner.py         # agent 执行入口
  eval/
    swebench_runner.py             # SWE-bench 批测入口
    ablation_runner.py             # 消融实验入口
  configs/
    base.yaml                      # 基础配置
    rollout_event_schema.yaml      # 轨迹协议
    swebench_eval.yaml             # SWE-bench 配置
    ablation.yaml                  # 消融配置
  scripts/
    run_deploy_demo.sh             # 部署演示
    run_agent_task_demo.sh         # 单任务演示
    run_swebench_eval.sh           # 跑测演示
    run_ablation.sh                # 消融演示
  docs/
    final_report_zh.md             # 最终设计与实验方案报告
```

## 3. 快速开始

> 当前为框架搭建阶段：先保证配置与执行入口完整，再逐步填充真实训练逻辑。

### 3.1 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyyaml
```

### 3.2 部署配置检查（slime + sglang）

```bash
bash slime-coding-agent/scripts/run_deploy_demo.sh
```

### 3.3 单任务 Agent 运行演示

```bash
bash slime-coding-agent/scripts/run_agent_task_demo.sh
```

### 3.4 SWE-bench 评测演示

```bash
bash slime-coding-agent/scripts/run_swebench_eval.sh
```

### 3.5 消融实验演示

```bash
bash slime-coding-agent/scripts/run_ablation.sh
```

## 4. 如何接入真实 slime / sglang

1. 在 `deploy/compose_slime_sglang.yaml` 中填入实际模型路径、并行参数与端口。
2. 在 `runner/coding_agent_runner.py` 中替换 `mock_execute_step` 为真实工具执行器。
3. 在 `eval/swebench_runner.py` 中替换 `mock_swebench_score` 为真实 SWE-bench harness 调用。
4. 在 `eval/ablation_runner.py` 中将实验矩阵映射到真实训练脚本参数。

## 5. 输出物

- 可复现实验日志（jsonl）
- SWE-bench 指标汇总（json）
- 消融对比结果（json）

详见 `docs/final_report_zh.md`。


## 6. 自主测试

```bash
export PYTHONPATH=slime-coding-agent:$PYTHONPATH
python -m unittest discover -s slime-coding-agent/tests -v
```
