# 远程 Docker 架构

SWE RL 将 Docker 执行层解耦到独立的 Docker 节点，GPU 节点只负责 LLM 推理和训练。

## 架构

```
┌─ Docker Node(s) ──────────────────────┐
│  server/swe_exec_server.py (:5000)    │  ← 每个 Docker 节点预装
│    /container/create                  │
│    /container/exec                    │
│    /container/diff                    │
│    /container/evaluate                │
│    /container/destroy                 │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ GPU Head Node─┼──────────────────────┐
│  server/swe_env_pool_server.py        │  ← 训练启动时由脚本运行
│    (:18090) 负载均衡 + lease 管理     │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ RolloutManager┼──────────────────────┐
│  swe_env_client.py                    │
│  generate_with_swe_remote.py          │
└───────────────────────────────────────┘
```

## 数据流

一个 SWE-Bench instance 的完整生命周期：

1. `generate()` 被 RolloutManager 调用
2. `SweEnvClient.allocate(image)` → pool_server 选最空闲节点 → `docker run` → 返回 `lease_id`
3. 多轮 Agent 循环（最多 20 步）：
   - LLM 推理（litellm → SGLang Router → SGLang 引擎）
   - 解析 bash 命令
   - `SweEnvClient.exec(command)` → pool_server → Docker Node → `docker exec` → 返回 output
   - 构建 observation → 回到 LLM 推理
4. Agent 提交 patch（或 `diff()` 获取 fallback patch）
5. 关闭 agent 容器，分配新的 eval 容器
6. `SweEnvClient.evaluate(patch, eval_script)` → Docker Node 上 `git apply` + 运行测试 → `resolved?`
7. `SweEnvClient.close()` → pool_server → Docker Node → `docker rm -f`
8. 编码 tokens + loss_mask + reward → 返回 Sample

---

## API 参考

### swe_exec_server.py（Docker Node :5000）

| 端点 | 方法 | 请求 | 响应 | 说明 |
|------|------|------|------|------|
| `/healthz` | GET | — | `{ok}` | 健康检查 |
| `/images` | GET | — | `{images, count}` | 列出本地 Docker 镜像 |
| `/status` | GET | — | `{active_containers}` | 活跃容器统计 |
| `/container/create` | POST | `{image, cwd?, timeout?}` | `{container_id, name}` | `docker run -d` |
| `/container/exec` | POST | `{container_id, command, cwd?, timeout?, env?}` | `{returncode, output}` | `docker exec` |
| `/container/diff` | POST | `{container_id, cwd?}` | `{patch, returncode}` | `git add -A && git diff --cached` |
| `/container/evaluate` | POST | `{container_id, patch, eval_script, cwd?, timeout?}` | `{resolved, output}` | `git apply` + 运行测试 |
| `/container/destroy` | POST | `{container_id}` | `{ok}` | `docker rm -f` |

容器创建时限制资源：`--pids-limit 256 --memory 4g`。

### swe_env_pool_server.py（GPU Head Node :18090）

| 端点 | 方法 | 请求 | 响应 | 说明 |
|------|------|------|------|------|
| `/healthz` | GET | — | `{ok}` | 健康检查 |
| `/status` | GET | — | `{total_leases, nodes}` | 池状态 |
| `/allocate` | POST | `{image, instance_id}` | `{lease_id, container_id}` | 选最空闲节点 + 创建容器 |
| `/heartbeat` | POST | `{lease_id}` | `{ok}` | 续租 |
| `/exec` | POST | `{lease_id, command, cwd?, timeout?, env?}` | `{returncode, output}` | 转发到 Docker Node |
| `/diff` | POST | `{lease_id, cwd?}` | `{patch}` | 转发到 Docker Node |
| `/evaluate` | POST | `{lease_id, patch, eval_script, cwd?, timeout?}` | `{resolved}` | 转发到 Docker Node |
| `/close` | POST | `{lease_id}` | `{ok}` | 销毁容器 + 释放 lease |

节点选择策略：选 `active_containers` 最小的健康节点。后台每 30 秒做一次健康检查。

### SweEnvClient（RolloutManager 内, async）

| 方法 | 调用端点 | 说明 |
|------|---------|------|
| `allocate(image, instance_id)` | `POST /allocate` | 获取 lease |
| `heartbeat(lease_id)` | `POST /heartbeat` | 续租 |
| `exec(lease_id, command, ...)` | `POST /exec` | 执行 bash |
| `diff(lease_id)` | `POST /diff` | 获取 git patch |
| `evaluate(lease_id, patch, eval_script)` | `POST /evaluate` | 评测 |
| `close(lease_id)` | `POST /close` | 关闭 |

---

## Docker 节点搭建

Docker 节点是任何能运行 Docker 并通过网络被 GPU 集群访问的 Linux 机器——可以是云厂商的 ECS/EC2 实例、裸金属服务器、或同集群内的 CPU 节点。

### 节点要求

| 要求 | 说明 |
|------|------|
| OS | Ubuntu 20.04+ / Debian 11+（需要 Docker） |
| CPU / 内存 | 每个 SWE 容器约 1-2 vCPU + 2-4 GB，按并发数规划 |
| 磁盘 | ≥ 1.5 TB（SWE Docker 镜像总量 500GB-1.5TB） |
| 网络 | GPU 集群能访问该节点的 :5000 端口 |
| GPU | 不需要 |

推荐规格参考（按并发数）：

| vCPU | 内存 | 最大并发容器 |
|------|------|-------------|
| 16 | 64 GB | ~4 |
| 32 | 128 GB | ~8 |

### 阶段 1：准备种子节点

创建或选择一台满足上述要求的机器。确保它与 GPU 训练集群网络互通（同一 VPC、VPN、或公网可达）。
防火墙 / 安全组需放行：TCP 22（SSH）、TCP 5000（exec server）。

### 阶段 2：安装软件

上传文件到节点：

```bash
NODE_IP=<节点IP>
scp server/swe_exec_server.py    root@${NODE_IP}:~/
scp server/setup_ecs_seed.sh     root@${NODE_IP}:~/
scp data/pull_swe_images.sh      root@${NODE_IP}:~/
scp ~/data/train.jsonl           root@${NODE_IP}:~/train.jsonl
```

SSH 登录后运行：

```bash
bash ~/setup_ecs_seed.sh
```

脚本自动执行：安装 Docker → 安装 Flask → 设置 swe_exec_server systemd 服务（开机自启 :5000）→ 拉取 SWE Docker 镜像。

验证：

```bash
curl http://localhost:5000/healthz
# {"ok": true, "running_containers": "0"}

curl http://localhost:5000/images | python3 -m json.tool | grep count
# "count": xxx
```

### 阶段 3：制作节点镜像（可选）

如果需要批量创建多台 Docker 节点，可以在种子节点安装完成后制作机器镜像（云厂商的自定义镜像 / AMI / snapshot），后续基于该镜像创建新节点，每台自带所有 SWE 镜像和 exec server。

### 阶段 4：启动 pool server

将所有 Docker 节点 IP 配置到 pool server（在 GPU head 节点运行）：

```bash
python3 -m swe_env_pool_server \
    --port 18090 \
    --exec-server-urls "http://10.0.0.10:5000,http://10.0.0.11:5000,http://10.0.0.12:5000"
```

---

## 配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SWE_ENV_SERVER_URL` | `http://localhost:18090` | Pool server URL |
| `SWE_ENV_SERVER_PORT` | `18090` | Pool server 端口 |
| `SWE_EXEC_SERVER_URLS` | `http://localhost:5000` | 逗号分隔的 exec server URL 列表 |
| `SWE_MAX_CONTAINERS_PER_NODE` | `16` | 每节点最大容器数 |
| `SWE_MAX_CONCURRENT` | `8` | Rollout 侧最大并发容器数 |
| `SWE_ENV_HTTP_MAX_RETRIES` | `10` | HTTP 请求最大重试次数 |
| `SWE_EVALUATE_MAX_RETRIES` | `3` | 评测请求最大重试次数 |
| `SWE_ROLLOUT_TIMEOUT` | `1800` | 单个 rollout 总超时（秒） |

### 磁盘规划

| 内容 | 大小 |
|------|------|
| OS + Docker daemon | ~10 GB |
| SWE Docker 镜像 | 500 GB - 1.5 TB |
| 运行时容器可写层 | ~50-100 GB |
| **推荐磁盘总量** | **≥ 1.5 TB** |

---

## 相关文件

| 文件 | 运行位置 | 职责 |
|------|---------|------|
| `server/swe_exec_server.py` | Docker Node | HTTP 包装本地 docker CLI |
| `server/swe_env_pool_server.py` | GPU Head Node | 多节点容器 lease 分配与转发 |
| `server/setup_ecs_seed.sh` | Docker Node（一次性） | 安装 Docker + exec server + 拉镜像 |
| `swe_env_client.py` | RolloutManager | 异步 HTTP 客户端 |
| `generate_with_swe_remote.py` | RolloutManager | Agent 循环 + 训练数据构建 |
| `data/pull_swe_images.sh` | ECS | 拉取 SWE Docker 镜像 |
