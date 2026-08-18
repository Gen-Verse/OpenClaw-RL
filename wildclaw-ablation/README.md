# WildClawBench Ablation: Base / RL / Skill / RL+Skill

四组消融在 WildClawBench **纯文本子集**（37 个任务：01/03/04/06 类）上对比
“参数进化”与“经验进化”的贡献。纯文本模型不跑 02/05 多模态任务。

## 变体

| variant | checkpoint | skills injected | 检验的问题 |
|---|---|---|---|
| `base` | `BASE_CKPT` | 无 | 基座能力 |
| `skill_only` | `BASE_CKPT` | 是 | 零训练成本，经验库值多少 |
| `rl_only` | `RL_CKPT` | 无 | Binary GRPO 值多少 |
| `rl_skill` | `RL_CKPT` | 是 | 参数 + 经验是否互补 |

可选第五组 `rl_opd`（GRPO+OPD checkpoint）复用 `rl_only` 配置换 ckpt。

## 准备（在 WildClawBench 克隆目录执行一次）

```bash
git clone https://github.com/InternLM/WildClawBench.git
cd WildClawBench

# 1. Docker 镜像（只需 OpenClaw harness 一个）
pip install -U "huggingface_hub[cli]"
hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar \
  --repo-type dataset --local-dir .
docker load -i Images/wildclawbench-ubuntu_v1.3.tar

# 2. 任务数据
hf download internlm/WildClawBench workspace --repo-type dataset --local-dir .

# 3. 本仓库的消融目录保持原样使用，不用拷数据
```

跳过 `script/prepare.sh`（YouTube 视频与 SAM3 权重只服务 02/05 多模态任务）。
检索类任务需要 `BRAVE_API_KEY`，没有就先跑 01/03/06。

## 模型 endpoint

`configs/my_api.template.json` 指到本机起的 SGLang/vLLM 服务
（WildClawBench 官方支持 `--models-config` 注入 openclaw.json）：

```bash
cp configs/my_api.template.json configs/my_api.json
# 把 BASE_URL 改成你的服务地址；容器内访问宿主机用 host.docker.internal
```

每个变体只改两样东西：endpoint 后面的模型名（base ckpt 或 RL ckpt 各起
一个 served model name）、是否注入 skills。

## 模型尺寸

支持 0.6B / 4B / 8B 三档，下载（服务器上执行）：

```bash
# 0.6B（弱，只适合冒烟）
hf download Qwen/Qwen3-0.6B --local-dir ~/Desktop/OpenClaw-RL/models/qwen3-0.6B
# 4B（推荐起点）
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/Desktop/OpenClaw-RL/models/Qwen3-4B-Instruct-2507
# 8B
hf download Qwen/Qwen3-8B --local-dir ~/Desktop/OpenClaw-RL/models/Qwen3-8B
# 国内可换 modelscope：modelscope download --model Qwen/Qwen3-8B --local_dir <同上>
```

按尺寸起服务（一次只起一个；换尺寸先 down 再起）：

```bash
bash scripts/up.sh base 0p6b   # qwen3-0p6b-base
bash scripts/up.sh base 4b     # qwen3-4b-instruct-base
bash scripts/up.sh base 8b     # qwen3-8b-base
bash scripts/up.sh down        # 停止当前服务
```

评测时模型名跟着变，例如 8B：`--model local/qwen3-8b-base`
（`run_variant.sh` 里用 `BASE_MODEL=qwen3-8b-base`）。

## 判分模型（grading judge）

部分任务的判分用 LLM judge（容器内 `OPENROUTER_BASE_URL` + `JUDGE_MODEL`
的 OpenAI 兼容调用），默认是 `openai/gpt-5.4` 走 OpenRouter。两种选择：

- **外部 judge（推荐写报告）**：`.env` 里填真 `OPENROUTER_API_KEY`，
  judge 保持官方默认——分数和官方榜单口径一致，消融相对差也最干净。
- **自评（自包含、零 API 成本）**：把 judge 也指到本机服务：

  ```bash
  # ~/Desktop/WildClawBench/.env
  OPENROUTER_BASE_URL=http://10.200.0.1:8000/v1   # 容器视角的网桥地址
  OPENROUTER_API_KEY=dummy
  JUDGE_MODEL=qwen3-8b-base                           # 与 up.sh 的 served name 一致
  ```

  权衡：0.6B/4B 当 judge 噪声很大，**判分不准**；而且 RL 变体如果
  judge 和被评模型是同一个，会有自评偏差。只建议在没 OpenRouter key、
  又需要跑通含 judge 任务时临时用，正式数字尽量用强外部 judge。
  消融对比时四组必须用**同一个 judge**，否则组间不可比。

## 评测与进化的尺寸参数

`run_variant.sh` / `run_cycle.sh` 都认 `SIZE`（0p6b|4b|8b），和
`up.sh` 的命名规则一致：

```bash
# 4B skill-only 闭环
SIZE=4b MODE=skill_only bash scripts/run_cycle.sh

# 8B 单变体评测
SIZE=8b bash scripts/run_variant.sh base
```

## Skill 注入方式

用官方 `--lobster-workspace` 机制：`run_variant.sh` 会把
`slime-coding-agent/skills/generated/*.md` 组装成
`results/<variant>/lobster_workspace/skills/<name>/SKILL.md`，
harness 启动任务容器时整个工作区被拷进容器 `/root/`，
openclaw 启动后即携带这套 skill 库。

注意：skill 只能来自**训练集轨迹**，评测集（本 benchmark 的任何任务）
失败轨迹不得进 skill 库，否则 skill_only/rl_skill 的分数不可信。

## 跑

```bash
cd <WildClawBench 克隆目录>

# 一键拉起模型服务（base / rl / down）
export ABLATION_ROOT=/path/to/OpenClaw-RL/wildclaw-ablation
bash $ABLATION_ROOT/scripts/up.sh          # 起 base ckpt 的 SGLang 服务
```

之后二选一：

```bash
# A. 一键闭环（切分→收集→进化→[训练]→四组评测→汇总）
bash $ABLATION_ROOT/scripts/run_cycle.sh

# B. 手动分步（调试/只跑某段时用）
export WCB_ROOT=$PWD

# 1. 切分
python3 $ABLATION_ROOT/scripts/make_split.py --wcb-root $WCB_ROOT --output $ABLATION_ROOT/configs/split.json

# 2. base 收集 train 切分（ROLLOUTS_PER_TASK>1 给 GRPO 凑组）
RUN_NAME=collect_base ROLLOUTS_PER_TASK=4 \
  bash $ABLATION_ROOT/scripts/run_tasks.sh train local/qwen3-0p6b-base 0

# 3. skill 进化一轮
PYTHONPATH=<repo>/slime-coding-agent python3 -m skill_evolve.run_round \
  --raw-dir $ABLATION_ROOT/results/collect_base/raw \
  --skills-dir $ABLATION_ROOT/skills \
  --report $ABLATION_ROOT/results/evolve_report.json

# 4. (可选) 离线 GRPO：转数据 -> 训练 -> 换 rl 服务
python3 $ABLATION_ROOT/scripts/wcb_to_rl_dataset.py \
  --raw-dir $ABLATION_ROOT/results/collect_base/raw \
  --model <repo>/models/qwen3-0.6B \
  --output $ABLATION_ROOT/results/rl_data/{rollout_id}.pt
HF_CKPT=<repo>/models/qwen3-0.6B RL_DATA=$ABLATION_ROOT/results/rl_data/{rollout_id}.pt \
  bash $ABLATION_ROOT/scripts/train_grpo_offline.sh
bash $ABLATION_ROOT/scripts/up.sh rl

# 5. 四组评测（只跑 held-out eval 切分）
bash $ABLATION_ROOT/scripts/run_variant.sh base
bash $ABLATION_ROOT/scripts/run_variant.sh skill_only
bash $ABLATION_ROOT/scripts/run_variant.sh rl_only
bash $ABLATION_ROOT/scripts/run_variant.sh rl_skill

# 6. 汇总
python3 $ABLATION_ROOT/scripts/compare_results.py \
  --results-dir $ABLATION_ROOT/results \
  --output $ABLATION_ROOT/results/ablation_summary.json
```

# 冒烟：单任务
bash $ABLATION_ROOT/scripts/run_variant.sh base \
  tasks/01_Productivity_Flow/<某个任务>.md
```

每个变体的结果落在 `results/<variant>/`（`output/summary_all.json` 会被
移动到这里，避免互相覆盖）。

## 汇总

```bash
python $ABLATION_ROOT/scripts/compare_results.py \
  --results-dir $ABLATION_ROOT/results \
  --output $ABLATION_ROOT/results/ablation_summary.json
```

输出 per-category 与 overall（子集内）分数表，四组并排。

## 口径声明（写报告时用）

- 子集 = 01/03/04/06（37 任务，纯文本）；不含 02/05 多模态。
- 分数是 WildClawBench 官方 grading（per-metric 0-1 + overall_score），
  子集 overall 为 37 任务均值，**不与官方 60 任务榜单直接可比**。
- 每组 `--parallel 4`，judge 用官方默认模型；如需无 judge 口径另报。
