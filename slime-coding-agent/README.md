# Slime Coding Agent

基于 **slime + sglang** 的可自我进化 Coding Agent 项目框架，覆盖：

- 模型部署（slime 训练 + sglang rollout serving）
- Agent 使用（任务执行、工具调用、验证循环）
- SWE-bench 跑测（批量评测与指标汇总）
- 消融实验（奖励、评审器、失败重放等）
- 自进化 Agent Server（资源感知训练 / 无卡 Skill 积累）

## 1. 项目目标

本子项目面向“工程可落地”，重点不是写概念文档，而是建立可扩展框架：

1. **Deploy**：可配置地拉起 slime/sglang 训练-推理闭环。
2. **Run**：以 planner/editor/verifier 三角色跑 coding 任务。
3. **Eval**：对接 SWE-bench 批测与成功率、成本统计。
4. **Ablation**：用统一配置切换组件验证增益。
5. **Evolve**：夜间自动判断 GPU 容量；资源充足时启动 Binary RL + OPD 联合训练，资源不足时将失败轨迹沉淀为可复用 Skill。

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
  server/
    agent_server.py                # 自进化 Agent Server
  configs/
    base.yaml                      # 基础配置
    rollout_event_schema.yaml      # 轨迹协议
    swebench_eval.yaml             # SWE-bench 配置
    ablation.yaml                  # 消融配置
    evolution.yaml                 # 调度、显卡门槛、训练与 Skill 配置
  scripts/
    run_deploy_demo.sh             # 部署演示
    run_agent_task_demo.sh         # 单任务演示
    run_swebench_eval.sh           # 跑测演示
    run_ablation.sh                # 消融演示
  docs/
    final_report_zh.md             # 最终设计与实验方案报告
```

## 3. 快速开始

### 3.1 准备环境

```bash
uv venv slime-coding-agent/.venv
uv pip install --python slime-coding-agent/.venv/Scripts/python.exe fastapi uvicorn pyyaml
```

### 3.2 单卡训练部署（slime + sglang）

```bash
export HF_CKPT=/path/to/Qwen3-4B
export PRM_EXTERNAL_API_BASE=https://your-prm-api/v1
export PRM_EXTERNAL_MODEL=your-prm-model
export PRM_EXTERNAL_API_KEY=your-api-key
docker compose -f slime-coding-agent/deploy/compose_slime_sglang.yaml up
```

该模板调用仓库内的单卡脚本，启用 INT4 QLoRA、FSDP、rollout offload、长序列 logits 分块与外部 OpenAI 兼容 PRM；训练与 rollout 共用一张 24GB GPU。

### 3.3 单任务 Agent 运行演示

```bash
bash slime-coding-agent/scripts/run_agent_task_demo.sh
```

每次运行默认覆盖当前事件日志，以避免不同实验互相污染；需要积累多轮轨迹时传入 `--append`。

### 3.4 SWE-bench 评测演示

```bash
bash slime-coding-agent/scripts/run_swebench_eval.sh
```

### 3.5 消融实验演示

```bash
bash slime-coding-agent/scripts/run_ablation.sh
```

消融不会合成分数。请将每组实验的轨迹写到 `outputs/ablations/<experiment>.jsonl`；脚本会对已有轨迹计算指标，并将尚未完成的组标记为 `pending`。

### 3.6 自进化 Agent Server

```bash
slime-coding-agent/.venv/Scripts/python.exe -m uvicorn server.agent_server:app --app-dir slime-coding-agent --host 0.0.0.0 --port 8010
```

服务通过 `POST /v1/trajectories` 接收任务轨迹。夜间窗口由 `configs/evolution.yaml` 控制：

- 当至少有 8 张、每张可用显存至少 24GB 的 GPU 时，生成失败轨迹批次并准备执行 `openclaw-combine` 的 Binary RL + OPD 联合训练。
- 当资源不足时，对失败轨迹去重、总结并写入 `skills/generated/`，供后续 Agent 运行和提示词检索。
- `training.execute` 默认为 `false`。完成模型、PRM、W&B 和 Linux/Ray 运行环境配置后，再显式改为 `true` 以允许夜间启动真实训练。

对于外部 LLM 总结，可将 `skill_fallback.summarizer.mode` 改为 `openai-compatible`，并设置 `SKILL_LLM_API_BASE`、`SKILL_LLM_MODEL` 和可选的 `SKILL_LLM_API_KEY`。

主要 API：

- `POST /v1/trajectories`：写入包含 `repo_id`、`task_id`、`final_status` 的执行轨迹。
- `POST /v1/evolution/run?force=true`：立即执行一次资源判断和进化循环。
- `GET /v1/evolution/status`：读取最近一次调度结果。
- `GET /v1/skills`：列出从失败轨迹积累的 Skill。

## 4. 如何接入真实 slime / sglang

1. 在 `deploy/compose_slime_sglang.yaml` 中填入实际模型路径、并行参数与端口。
2. 在 `runner/coding_agent_runner.py` 中接入 planner/editor 的模型调用；当前执行器已可在受限命令白名单中真实运行测试命令。
3. 将 `eval/swebench_runner.py` 的事件输入替换为 SWE-bench harness 的任务轨迹。
4. 使用 `--experiment <name>` 运行各组实验，并将轨迹分别落盘到 `outputs/ablations/<name>.jsonl`。
5. 用 `server/agent_server.py` 承接生产轨迹；将训练配置与资源阈值写入 `configs/evolution.yaml`。

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
