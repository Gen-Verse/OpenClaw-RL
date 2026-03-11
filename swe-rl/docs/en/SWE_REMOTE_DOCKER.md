# Remote Docker Architecture

SWE RL decouples the Docker execution layer onto separate Docker nodes, leaving GPU nodes solely responsible for LLM inference and training.

## Architecture

```
┌─ Docker Node(s) ──────────────────────┐
│  server/swe_exec_server.py (:5000)    │  ← pre-installed on each Docker node
│    /container/create                  │
│    /container/exec                    │
│    /container/diff                    │
│    /container/evaluate                │
│    /container/destroy                 │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ GPU Head Node─┼──────────────────────┐
│  server/swe_env_pool_server.py        │  ← started by training script
│    (:18090) load-balancing + leases   │
└────────────────▲──────────────────────┘
                 │ HTTP
┌─ RolloutManager┼──────────────────────┐
│  swe_env_client.py                    │
│  generate_with_swe_remote.py          │
└───────────────────────────────────────┘
```

## Data Flow

Full lifecycle of one SWE-Bench instance:

1. `generate()` is called by the RolloutManager
2. `SweEnvClient.allocate(image)` → pool_server picks the least-loaded node → `docker run` → returns `lease_id`
3. Multi-turn agent loop (up to 20 steps):
   - LLM inference (litellm → SGLang Router → SGLang engines)
   - Parse bash command
   - `SweEnvClient.exec(command)` → pool_server → Docker Node → `docker exec` → returns output
   - Build observation → back to LLM inference
4. Agent submits patch (or `diff()` for fallback patch)
5. Close agent container, allocate a fresh eval container
6. `SweEnvClient.evaluate(patch, eval_script)` → Docker Node runs `git apply` + test suite → `resolved?`
7. `SweEnvClient.close()` → pool_server → Docker Node → `docker rm -f`
8. Encode tokens + loss_mask + reward → return Sample

---

## API Reference

### swe_exec_server.py (Docker Node :5000)

| Endpoint | Method | Request | Response | Description |
|----------|--------|---------|----------|-------------|
| `/healthz` | GET | — | `{ok}` | Health check |
| `/images` | GET | — | `{images, count}` | List local Docker images |
| `/status` | GET | — | `{active_containers}` | Active container stats |
| `/container/create` | POST | `{image, cwd?, timeout?}` | `{container_id, name}` | `docker run -d` |
| `/container/exec` | POST | `{container_id, command, cwd?, timeout?, env?}` | `{returncode, output}` | `docker exec` |
| `/container/diff` | POST | `{container_id, cwd?}` | `{patch, returncode}` | `git add -A && git diff --cached` |
| `/container/evaluate` | POST | `{container_id, patch, eval_script, cwd?, timeout?}` | `{resolved, output}` | `git apply` + run tests |
| `/container/destroy` | POST | `{container_id}` | `{ok}` | `docker rm -f` |

Containers are created with resource limits: `--pids-limit 256 --memory 4g`.

### swe_env_pool_server.py (GPU Head Node :18090)

| Endpoint | Method | Request | Response | Description |
|----------|--------|---------|----------|-------------|
| `/healthz` | GET | — | `{ok}` | Health check |
| `/status` | GET | — | `{total_leases, nodes}` | Pool status |
| `/allocate` | POST | `{image, instance_id}` | `{lease_id, container_id}` | Pick least-loaded node + create container |
| `/heartbeat` | POST | `{lease_id}` | `{ok}` | Renew lease |
| `/exec` | POST | `{lease_id, command, cwd?, timeout?, env?}` | `{returncode, output}` | Forward to Docker Node |
| `/diff` | POST | `{lease_id, cwd?}` | `{patch}` | Forward to Docker Node |
| `/evaluate` | POST | `{lease_id, patch, eval_script, cwd?, timeout?}` | `{resolved}` | Forward to Docker Node |
| `/close` | POST | `{lease_id}` | `{ok}` | Destroy container + release lease |

Node selection: pick the healthy node with the lowest `active_containers`. Background health checks run every 30 seconds.

### SweEnvClient (inside RolloutManager, async)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `allocate(image, instance_id)` | `POST /allocate` | Acquire lease |
| `heartbeat(lease_id)` | `POST /heartbeat` | Renew lease |
| `exec(lease_id, command, ...)` | `POST /exec` | Execute bash |
| `diff(lease_id)` | `POST /diff` | Get git patch |
| `evaluate(lease_id, patch, eval_script)` | `POST /evaluate` | Evaluate |
| `close(lease_id)` | `POST /close` | Close |

---

## Docker Node Setup

A Docker node is any Linux machine that can run Docker and is network-reachable from the GPU training cluster — cloud instances (ECS/EC2/GCE), bare-metal servers, or CPU nodes in the same cluster all work.

### Node Requirements

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 20.04+ / Debian 11+ (Docker required) |
| CPU / Memory | ~1–2 vCPU + 2–4 GB per SWE container; size by concurrency |
| Disk | ≥ 1.5 TB (SWE Docker images total 500GB–1.5TB) |
| Network | GPU cluster must be able to reach port :5000 on this node |
| GPU | Not needed |

Sizing reference (by concurrency):

| vCPU | Memory | Max Concurrent Containers |
|------|--------|---------------------------|
| 16 | 64 GB | ~4 |
| 32 | 128 GB | ~8 |

### Stage 1: Prepare a Seed Node

Create or pick a machine meeting the requirements above. Ensure it has network connectivity to the GPU training cluster (same VPC, VPN, or publicly reachable).
Firewall / security group must allow: TCP 22 (SSH), TCP 5000 (exec server).

### Stage 2: Install Software

Upload files to the node:

```bash
NODE_IP=<node IP>
scp server/swe_exec_server.py    root@${NODE_IP}:~/
scp server/setup_ecs_seed.sh     root@${NODE_IP}:~/
scp data/pull_swe_images.sh      root@${NODE_IP}:~/
scp ~/data/train.jsonl           root@${NODE_IP}:~/train.jsonl
```

SSH in and run:

```bash
bash ~/setup_ecs_seed.sh
```

The script automatically: installs Docker → installs Flask → sets up swe_exec_server as a systemd service (auto-start on :5000) → pulls SWE Docker images.

Verify:

```bash
curl http://localhost:5000/healthz
# {"ok": true, "running_containers": "0"}

curl http://localhost:5000/images | python3 -m json.tool | grep count
# "count": xxx
```

### Stage 3: Create a Node Image (optional)

If you need multiple Docker nodes, create a machine image (cloud custom image / AMI / snapshot) from the seed node after setup is complete. All subsequent nodes created from this image will have all SWE images and the exec server pre-installed.

### Stage 4: Start Pool Server

Configure all Docker node IPs into the pool server (runs on the GPU head node):

```bash
python3 -m swe_env_pool_server \
    --port 18090 \
    --exec-server-urls "http://10.0.0.10:5000,http://10.0.0.11:5000,http://10.0.0.12:5000"
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SWE_ENV_SERVER_URL` | `http://localhost:18090` | Pool server URL |
| `SWE_ENV_SERVER_PORT` | `18090` | Pool server port |
| `SWE_EXEC_SERVER_URLS` | `http://localhost:5000` | Comma-separated exec server URLs |
| `SWE_MAX_CONTAINERS_PER_NODE` | `16` | Max containers per node |
| `SWE_MAX_CONCURRENT` | `8` | Max concurrent containers (rollout side) |
| `SWE_ENV_HTTP_MAX_RETRIES` | `10` | HTTP request max retries |
| `SWE_EVALUATE_MAX_RETRIES` | `3` | Evaluate request max retries |
| `SWE_ROLLOUT_TIMEOUT` | `1800` | Per-rollout total timeout (seconds) |

### Disk Planning

| Content | Size |
|---------|------|
| OS + Docker daemon | ~10 GB |
| SWE Docker images | 500 GB – 1.5 TB |
| Runtime container writable layers | ~50–100 GB |
| **Recommended total disk** | **≥ 1.5 TB** |

---

## Related Files

| File | Runs On | Role |
|------|---------|------|
| `server/swe_exec_server.py` | Docker Node | HTTP wrapper around local docker CLI |
| `server/swe_env_pool_server.py` | GPU Head Node | Multi-node container lease allocation and forwarding |
| `server/setup_ecs_seed.sh` | Docker Node (one-time) | Install Docker + exec server + pull images |
| `swe_env_client.py` | RolloutManager | Async HTTP client |
| `generate_with_swe_remote.py` | RolloutManager | Agent loop + training data construction |
| `data/pull_swe_images.sh` | ECS | Pull SWE Docker images |
