
# a3s-code-rl

`a3s-code-rl` 是一个将 `a3s-code` 真实多轮代码生成会话接入 `slime` 在线强化学习（RL）训练循环的集成模块。

本项目无需离线构造指令数据。训练样本直接源自真实的 Agent 交互会话：从捕获用户请求，到 Agent 调用工具、修改文件以及获取后续状态，RL Proxy 会实时筛选符合训练条件的交互轮次（Turn）并封装为 `Sample`，最终提交给 `slime` 进行模型更新。

首次阅读本项目，建议优先了解以下 4 个核心文件：

- [`code_rl_api_server.py`](code_rl_api_server.py)：OpenAI 兼容的代理服务（Proxy），负责 Session 跟踪、下一状态（Next State）获取、PRM 打分以及样本提交。
- [`code_rl_rollout.py`](code_rl_rollout.py)：`slime` 的 Rollout 入口组件，负责样本队列的等待与批处理返回。
- [`a3s_code_agent_traffic_driver.py`](a3s_code_agent_traffic_driver.py)：真实流量驱动器（Traffic Driver），负责持续拉起并管理 `a3s-code` 的交互会话。
- [`run_a3s_code_rl_4gpu.sh`](run_a3s_code_rl_4gpu.sh)：默认的 4 卡训练启动入口，也是核心环境变量的配置事实标准。

## 目录结构

| 路径 | 用途 |
|------|------|
| `code_rl_api_server.py` | 核心 RL Proxy。 |
| `code_rl_rollout.py` | 面向 `slime` 的 Rollout 接口实现。 |
| `a3s_code_agent_traffic_driver.py` | 持续生成并驱动真实 `a3s-code` 会话。 |
| `run_a3s_code_rl_4gpu.sh` | 默认 4 卡训练启动脚本。 |
| `run_a3s_code_agent_traffic.sh` | 默认流量驱动启动脚本。 |
| `check_simulated_user_backends.py` | 检查模拟用户（Simulated-User）后端的健康与可用性。 |
| `refresh_simulated_user_backends.sh` | 刷新 `simulated_user_backends.json` 列表。 |
| `simulated_user_backends.json` | **[运行产物]** 当前可用 simulated-user 后端列表，由刷新脚本生成。 |
| `seed_data/` | 初始化任务种子数据。 |
| `task_templates/` | 小型代码仓库模板集。 |
| `generated_workspaces/` | **[运行产物]** 隔离的 Session 独立工作区。 |
| `generated_configs/` | **[运行产物]** 动态配置。默认生成一份共享 HCL；若开启 `per_session` 模式，则生成按会话隔离的 HCL 批次。 |
| `workspace_template_cache/` | **[运行产物]** 工作区模板缓存。 |
| `results/` | **[运行产物]** Driver 运行记录与日志。 |

## 核心工作流

系统的核心执行链路如下：

```text
Seed Task
  -> Simulated User 对初始 Seed 进行拟人化改写
  -> a3s-code 在隔离的独立工作区中执行代码任务
  -> code_rl_api_server.py 记录交互 Turn / Next State / Reward
  -> code_rl_rollout.py 收集并组装 Batch Sample
  -> slime 执行一轮 Train Step
```

**详细运行步骤：**

1. **任务采样**：`a3s_code_agent_traffic_driver.py` 从 `seed_data/code_task_seeds.json` 中抽取任务。
2. **环境隔离**：系统从 `task_templates/` 复制对应模板至 `generated_workspaces/`，为当前任务建立独立仓库。
3. **配置分发**：为 `a3s-code` 准备 HCL 配置文件（默认使用共享配置，显式开启 `per_session` 后退化为独立配置）。
4. **请求拦截**：`a3s-code` 根据配置将请求发送至本地 RL Proxy，而非直连 Rollout 模型。
5. **策略转发与记录**：`code_rl_api_server.py` 将请求转发至本地 Policy 模型，并同步记录具备训练价值的交互轮次。
6. **状态回填与打分**：当环境进入下一状态时，Proxy 会为上一轮次补充 `next_state` 信息，并执行 PRM 或规则维度的奖励打分。
7. **样本提交**：`code_rl_rollout.py` 缓存样本，满足 Batch 条件后返回给 `slime`。
8. **模型更新**：`slime` 完成一次 Actor 权重更新，进入下一迭代循环。

## 前置依赖与运行说明

本目录作为集成模块，需依赖外部组件运行。确保已具备以下条件：

- 环境中已安装并可导入的 `a3s-code` Python SDK。
- 仓库根目录同级或指定路径下存在可用的 `slime/` 和 `Megatron-LM/` 项目。
- 具备本地 Policy 模型的初始权重文件。
- 若启用外部 PRM 或外部 Simulated-User，需备齐对应的 API Endpoint 与访问密钥。

*注：`generated_workspaces/`、`generated_configs/`、`workspace_template_cache/` 及 `results/` 均为运行时动态生成的产物目录，通常无需手动提交或修改。*

## 快速启动

以下为当前推荐的标准启动流程。命令中的脚本默认采用相对路径，以确保模块的跨目录可用性。

### 1. 环境准备

首先配置基础环境变量：

```bash
export CONDA_ENV=/path/to/openclaw-rl
export A3S_CODE_REPO_ROOT=/path/to/a3s-lab/Code
export HF_CKPT=/path/to/Qwen3-4B-Thinking-2507
```

由于当前 `a3s-code` 主线已支持共享 HCL 所需的 `sessionIdHeader` 功能，建议直接拉取最新 `main` 分支，并将 SDK 编译至对应的运行环境中：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
git -C "$A3S_CODE_REPO_ROOT" pull --rebase origin main
cd "$A3S_CODE_REPO_ROOT/sdk/python"
conda run -p "$CONDA_ENV" python -m maturin develop --release
```

**验证安装：**

```bash
conda run -p "$CONDA_ENV" python -c "import a3s_code, inspect; print(inspect.getfile(a3s_code))"
```
*预期输出应指向 `openclaw-rl` 环境路径下的 `site-packages/a3s_code/__init__.py`。*

**配置模式说明：**
当前脚本默认采用共享 HCL 模式。若使用的 `a3s-code` 版本过旧或需回退旧逻辑，请显式开启兼容模式：
```bash
export A3S_CODE_AGENT_CONFIG_MODE=per_session
```

### 2. 更新模拟用户 (Simulated User) 配置

在仓库根目录执行以下命令，更新后端可用性列表：

```bash
bash a3s-code-rl/refresh_simulated_user_backends.sh
```

这个脚本会优先复用当前 `CONDA_PREFIX/bin/python3`，其次才是 `CONDA_ENV/bin/python3`。输出会写回 `a3s-code-rl/simulated_user_backends.json`。

### 3. 启动训练主栈

```bash
env ENABLE_PRM=1 PRM_BACKEND=external_openai CODE_RL_REWARD_MODE=prm \
  bash a3s-code-rl/run_a3s_code_rl_4gpu.sh
```
该脚本将依次拉起 Ray 集群、本地 SGLang Policy 服务、`slime` 训练进程及 RL Proxy。

脚本启动后会打印一行：

```text
=== RUN_ROOT: runs/<run_id> ===
```

后面的记录文件和 `launch_info.json` 都在这个目录下。

### 4. 检查服务状态

验证关键端口是否处于监听状态：

```bash
ss -ltnp | grep -E ':30000|:15001|:8265'
```
- `30000`：RL Proxy 端口
- `15001`：本地 Policy 端口
- `8265`：Ray Dashboard 端口

### 5. 启动流量驱动 (Traffic Driver)

```bash
bash a3s-code-rl/run_a3s_code_agent_traffic.sh
```
该脚本将自动清理/更新后端列表，筛选健康节点，并持续注入新的 `a3s-code` 任务会话。

### 6. 监控训练进度

可通过以下日志判断训练链路是否正常推进：
- Proxy 日志中持续输出 `submitted sample`。
- Rollout 日志中的 `waiting for API-produced samples: x/16` 计数递增。
- Slime 训练日志中出现 Step 更新指标。

**常用监控命令：**
```bash
cat runs/<run_id>/launch_info.json
tail -f runs/<run_id>/code_rl_record.jsonl
tail -f runs/<run_id>/code_rl_prm_record.jsonl
```

## 默认硬件与模型拓扑

目前提供的默认启动脚本基于 4 卡 GPU 环境设计，拓扑分配如下：

- **Actor**: 2 GPU (`TP_TRAIN=2`)
- **Rollout**: 2 GPU (`TP_SGLANG=2`)
- **PRM**: 外部 OpenAI 兼容接口
- **上下文长度**: 8192

相关核心阈值定义在启动脚本中：

```bash
CODE_RL_MATCHED_CONTEXT_TOKENS=8192
CODE_RL_MAX_TRAIN_TOKENS=8192
ROLLOUT_MAX_CONTEXT_LEN=8192
ROLLOUT_MAX_RESPONSE_LEN=2048

A3S_CODE_MAX_MAIN_TURNS=4
A3S_CODE_MAX_TOOL_ROUNDS=16
A3S_CODE_MODEL_CONTEXT_TOKENS=7168
A3S_CODE_MODEL_OUTPUT_TOKENS=2048
```
*提示：建议保持 `CODE_RL_MATCHED_CONTEXT_TOKENS` 与 `ROLLOUT_MAX_CONTEXT_LEN` 处于同一量级，以防训练数据与 Rollout 实际分布产生偏移。*

## 关键环境变量参考

**训练配置层 (`run_a3s_code_rl_4gpu.sh`)：**
- `ACTOR_GPUS` / `ROLLOUT_GPUS`：节点资源分配
- `TP_TRAIN` / `TP_SGLANG`：模型并行度
- `CODE_RL_MATCHED_CONTEXT_TOKENS` / `CODE_RL_MAX_TRAIN_TOKENS`：训练截断长度
- `ROLLOUT_MAX_CONTEXT_LEN` / `ROLLOUT_MAX_RESPONSE_LEN`：推理截断长度
- `ENABLE_PRM` / `PRM_BACKEND` / `CODE_RL_REWARD_MODE`：奖励模型控制开关

**流量驱动层 (`run_a3s_code_agent_traffic.sh`)：**
- `A3S_CODE_TRAFFIC_CONCURRENCY`：并发会话数
- `A3S_CODE_MAX_MAIN_TURNS` / `A3S_CODE_MAX_TOOL_ROUNDS`：会话轮次上限
- `A3S_CODE_MODEL_CONTEXT_TOKENS` / `A3S_CODE_MODEL_OUTPUT_TOKENS`：Agent 侧视角长度限制
- `A3S_CODE_AGENT_CONFIG_MODE`：HCL 生成模式 (`shared` 或 `per_session`)

**外部服务与模拟用户配置：**
- `A3S_CODE_SIMULATED_USER_BACKENDS_FILE`
- `A3S_CODE_SIMULATED_USER_TIMEOUT_SEC`
- `CODE_RL_PRM_OPENAI_URL` / `CODE_RL_PRM_OPENAI_MODEL_NAME`

## HCL 配置与会话 (Session) 绑定机制

当前系统默认采用 **共享 HCL 配置** 模式。所有产生的会话将读取同一份底层的 HCL 配置，其核心网关参数如下：

```hcl
providers {
  name = "openai"
  base_url = "http://127.0.0.1:30000"
  sessionIdHeader = "X-Session-Id"
}
```

### 共享配置的实现原理
共享机制成立的核心在于 `a3s-code` 客户端的支持：
1. HCL / `CodeConfig` 现已原生支持声明 Provider 级别的 `sessionIdHeader`。
2. `AgentSession` 在实例化 LLM Client 时，会动态将自身的 `session_id` 注入该 HTTP Header 随请求发出。
因此，底层的 RL Proxy 配置可以被安全复用，而具体的会话隔离由请求头动态保障。

### RL Proxy 的 Session 识别策略
`code_rl_api_server.py` 按以下优先级提取当前请求的所属会话：
1. URL Path 中的 `session_id`
2. HTTP Header 中的 `X-Session-Id`（共享模式的主要匹配途径）
3. Request Body 中的 `session_id` 字段
4. Request Body 中的 `user` 字段
5. 降级通过消息历史前缀进行模糊匹配

### 服务端点配置要求 (限制直连 Rollout)
在共享配置模式下，`base_url` 必须始终指向 RL Proxy 的代理端口（如 `http://127.0.0.1:30000`）。
**严禁**将 `base_url` 直接修改为 Rollout 模型端口（如 `15001`）。若绕过 Proxy 进行直连，将导致以下核心功能失效：
- 跨轮次 Session 跟踪与管理
- Turn 与 Next State 的拼接
- 基于 PRM 的实时轨迹打分
- 训练样本向 `code_rl_rollout.py` 的回调提交

### 向下兼容：独立 HCL 模式 (per-session)
若所在环境中的 `a3s-code` 版本不支持动态 Header 注入，可通过以下变量回退至旧版的独立配置模式：

```bash
export A3S_CODE_AGENT_CONFIG_MODE=per_session
```

在此模式下，Driver 将退化为每个会话生成独立 HCL，硬编码目标会话 ID 于 URL 路径中（例如：`base_url = "http://127.0.0.1:30000/session/<session_id>"`）。
