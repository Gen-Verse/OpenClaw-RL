# OpenClaw Hybrid GRPO + Top-K OPD

This directory contains the OpenClaw hybrid objective used by the current top-k
hint-selection launcher:

```bash
cd /data_storage/wyj/OpenClaw-RL/slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

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
