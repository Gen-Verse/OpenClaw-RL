
### Slime Env Setup

```bash
# cuda 12.9 (nvcc -V, nvidia-smi)
conda create --name openclaw-rl python=3.12

pip install \
  torch==2.9.1+cu129 \
  torchvision==0.24.1+cu129 \
  torchaudio==2.9.1+cu129 \
  --index-url https://download.pytorch.org/whl/cu129

pip install -r /absolute/path/to/OpenClaw-RL/requirements.txt

# DeepEP source is from: https://github.com/deepseek-ai/DeepEP
pip install -e /absolute/path/to/sgl-workspace/DeepEP --no-build-isolation

pip install -e /absolute/path/to/OpenClaw-RL/slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation

# apex
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# flash_attn
export MAX_JOBS=8
pip install --no-build-isolation -v flash-attn==2.7.4.post1
pip install "flashinfer-jit-cache==0.5.3" --index-url https://flashinfer.ai/whl/cu129

apt-get update
apt-get install -y python3-apt
```



### More Details about Configurations in .sh


| Variable | Default | Description |
|---|---|---|
| `rollout-batch-size` | `32` | after collect rollout-batch-size samples, start training |
| `rollout-max-response-len` | `8192` | max response length in each message |
| `rollout-max-context-len` | `32768` | max context length in a session |
| `rollout-temperature` | `0.6` | temperature |
| `advantage-estimator` | (see script) | `on_policy_distillation` / `grpo` |
| `kl-loss-coef` | `0.01` | kl loss weight |

















