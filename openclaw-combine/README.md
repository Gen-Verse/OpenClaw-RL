# OpenClaw Hybrid GRPO + Top-K OPD

This directory contains the OpenClaw hybrid objective used by the current top-k
hint-selection launcher:

```bash
cd /data_storage/wyj/OpenClaw-RL/slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

Note that you MUST convert the hf format to torch list format before optimization. See details in comments of sh files.

The older `openclaw_combine` launchers are kept for reference. The main path for
the current experiments is `run_qwen3_4b_openclaw_topk_select.sh`, which uses
multi-candidate hint generation, overlap-guided hint selection, and the custom
loss in `openclaw_topk_select_loss.py`.

## Method

OpenClaw uses two complementary signals from each scored interaction turn.

The evaluative signal is a scalar PRM vote, `r_t in {+1, -1, 0}`, for the pair
`(a_t, s_{t+1})`. It is dense because every scored turn can contribute a GRPO
sample, including turns whose feedback is implicit, such as a re-query, a test
result, or a short user reaction.

The directive signal is produced only when the PRM judges that `s_{t+1}`
contains a meaningful correction. In that case, the PRM extracts one or more
candidate hints, each wrapped as `[HINT_START]...[HINT_END]`. A hint-augmented
prompt `s_t^h = s_t + h` is sent through the teacher model to obtain a
token-level teacher distribution. This signal is sparse, but it gives richer
per-token guidance than a scalar reward.

The hybrid loss combines both terms per response token:

```text
L_i = w_RL * L_i^GRPO + w_OPD * L_i^OPD
```

By default, the launcher sets both weights to `1.0`. This trains with both the
dense evaluative GRPO signal and the sparse directive OPD signal.

To run OPD-only, disable the GRPO branch:

```bash
OPENCLAW_TOPK_W_RL=0 bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

To run GRPO-only, disable the OPD branch:

```bash
OPENCLAW_TOPK_W_OPD=0 bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

## Overlap-Guided Hint Selection

The top-k select path addresses teacher-student mismatch in OPD. For each
candidate hint `h` and response token position `i`, the loss compares the
student old policy's top-k vocabulary with the hint-conditioned teacher's top-k
vocabulary:

```text
S_i^q     = top-k pi_old(. | s_t, y_<i)
S_{i,h}^p = top-k pi_T(. | s_t^h, y_<i)
O[h, i]   = |S_i^q intersection S_{i,h}^p|
```

The selected hint is controlled by `OPENCLAW_TOPK_HINT_SELECTION`:

- `sequence_optimal` selects one hint per sample by maximizing summed overlap
  across response tokens. This is the default and is generally more stable for
  agentic RL.
- `token_optimal` selects the best hint independently at each token position.
- `shortest` uses the first candidate, after the API server sorts candidates by
  hint length.

After selecting the hint, OPD is applied only on a top-k vocabulary subset. The
subset is controlled by `OPENCLAW_TOPK_SUBSET_MODE`:

- `student` uses `top-k(pi_old)`. This is the default.
- `teacher` uses `top-k(pi_T)` from the selected teacher.
- `overlap` uses the intersection of the student and teacher top-k sets.

The OPD advantage for each selected vocabulary token is the clipped teacher-old
log-probability gap weighted by the old-policy probability inside the subset.
The launcher sets `OPENCLAW_TOPK_ADV_DIFF_CLIP=1.0` to cap the magnitude of the
distillation update.

## Launcher Defaults

The default `run_qwen3_4b_openclaw_topk_select.sh` setup assumes an 8-GPU node:

- `NUM_GPUS=8`
- `ACTOR_GPUS=4`
- `ROLLOUT_GPUS=2`
- `PRM_GPUS=1`
- `PRM_TEACHER_GPUS=1`

The script forces `OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron` because
top-k selection requires per-candidate teacher top-k distributions. The
inference-side teacher path only provides single-candidate teacher log-probs.

Important defaults:

- `OPENCLAW_TOPK_W_RL=1.0`: GRPO loss weight.
- `OPENCLAW_TOPK_W_OPD=1.0`: top-k OPD loss weight.
- `OPENCLAW_TOPK_K=4`: top-k width for student and teacher vocab sets.
- `OPENCLAW_TOPK_MAX_CAND=3`: maximum number of accepted hint candidates kept
  per turn.
- `PRM_M=3`: number of PRM judge/eval votes per turn.
- `OPENCLAW_TOPK_HINT_SELECTION=sequence_optimal`: hint selection rule.
- `OPENCLAW_TOPK_SUBSET_MODE=student`: OPD vocabulary subset.
- `OPENCLAW_TOPK_ADV_DIFF_CLIP=1.0`: clip on `log pi_T - log pi_old`.
- `--eps-clip 0.2` and `--eps-clip-high 0.28`: PPO-style ratio clipping.

Model paths can be overridden through the usual environment variables:

- `HF_CKPT`: Hugging Face checkpoint used by the tokenizer and SGLang.
- `REF_LOAD`: student torch-dist checkpoint loaded by Megatron.
- `SAVE_CKPT`: output checkpoint directory.
- `PRM_MODEL_PATH`: PRM model path for SGLang.
- `PRM_TEACHER_LOAD`: teacher torch-dist checkpoint loaded by Megatron.
- `PRM_TEACHER_HF`: teacher Hugging Face checkpoint metadata.

## Runtime Flow

For each interaction turn:

1. The rollout path calls
   `openclaw_combine_select_rollout.generate_rollout_openclaw_combine_select`.
2. The API server asks the PRM for evaluative votes and candidate directive
   hints.
3. Accepted candidate hints are converted into teacher top-k tensors by the
   Megatron teacher path.
4. The trainer calls
   `openclaw_topk_select_loss.openclaw_topk_select_loss_function`.
5. The loss computes `w_RL * L^GRPO + w_OPD * L^OPD`, using the selected hint
   and selected top-k vocabulary subset for the OPD branch.

Training records are written under `openclaw-combine/results/` when
`OPENCLAW_RECORD_ENABLED=1`.

## Files

```text
openclaw-combine/
├── run_qwen3_4b_openclaw_topk_select.sh      # Main Qwen3-4B top-k select launcher
├── openclaw_topk_select_loss.py              # Hybrid GRPO + top-k OPD loss
├── openclaw_combine_select_api_server.py     # PRM eval, hint candidates, teacher tensors
├── openclaw_combine_select_rollout.py        # SLIME rollout bridge for top-k select
├── prm_teacher_postprocess.py                # Teacher tensor post-processing helpers
├── run_qwen*_openclaw_topk_select*.sh        # Other model/node variants
├── run_qwen*_openclaw_combine*.sh            # Legacy combine launchers
├── combine_loss.py                           # Legacy combine loss
└── results/                                  # Runtime records
```

---

## Single-GPU INT4 QLoRA

Train and serve on a **single 24 GB GPU** using `run_qwen3_4b_openclaw_combine_single_gpu_int4_qlora.sh`.

### Required Environment Variables

| Variable | Description |
|---|---|
| `HF_CKPT` | Path to original Qwen3-4B HF weights (**not** pre-quantized) |
| `OPENAI_API_KEY` | API key for external LLM (used by PRM and OPD teacher) |
| `OPENAI_BASE_URL` | Base URL for the external LLM API |
| `EXTERNAL_MODEL` | Model name served by the external API |

### Optional Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MICRO_BATCH_SIZE` | `1` | Micro-batch size per GPU |
| `ROLLOUT_BATCH_SIZE` | `16` | Samples collected before each training step |

### Key Features Enabled

| Feature | Flag / Config | Description |
|---|---|---|
| INT4 quantization | `--fsdp-load-in-4bit` | bitsandbytes NF4 quantization. Compresses base model weights to ~4 bits, reducing VRAM from ~8 GB (bf16) to ~2 GB for Qwen3-4B. Only LoRA adapters are trained in full precision. |
| LoRA fine-tuning | `--use-lora --lora-rank 4` | Low-Rank Adaptation. Freezes base weights and inserts small trainable matrices into attention/MLP layers.
| Gradient checkpointing | `--gradient-checkpointing` | Trades compute for memory by recomputing activations during backward pass instead of storing them. |
| Colocate (train + rollout on same GPU) | `--colocate` | Runs both the FSDP training actor and the SGLang rollout engine on the same GPU(s). Essential for single-GPU setups. |
| Rollout offload | `--offload-rollout` | When training starts, the SGLang rollout engine is offloaded from GPU to free VRAM for the training forward/backward pass. When training finishes, the engine is reloaded for the next rollout. This alternating pattern allows a single GPU to handle both roles. |
| External PRM API | `--prm-use-external-api` | Sends PRM scoring requests to an external OpenAI-compatible API instead of hosting a local PRM model on GPU. Eliminates the need for a dedicated PRM GPU — critical for single-GPU setups. Configured via `PRM_EXTERNAL_API_BASE`, `PRM_EXTERNAL_MODEL`, and `PRM_EXTERNAL_API_KEY`. |
| Train sequence truncation | `SLIME_TRAIN_MAX_SEQ_LEN=4096` | Caps the token length of each training sample. Sequences longer than this are truncated from the **left** (dropping early prompt tokens, preserving the response). Prevents OOM on long rollouts and keeps per-sample memory bounded. |
| Logit chunking | `SLIME_LOGIT_CHUNK_SIZE=512` | Computes log-softmax and entropy in chunks of this size instead of materializing the full `[seq_len, vocab_size]` tensor at once. For Qwen3-4B (vocab ~152 K), this reduces peak allocation from ~7.5 GB to ~0.9 GB per sample. Set to `0` to disable chunking. |

### Quick Start

```bash
cd slime

export HF_CKPT="/path/to/Qwen3-4B"
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-api-base"
export EXTERNAL_MODEL="your-model-name"

bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_single_gpu_int4_qlora.sh
```

The model is served at `http://0.0.0.0:30000/v1` by default. For single-GPU evaluation scripts, see [`../openclaw-test/README.md`](../openclaw-test/README.md).
