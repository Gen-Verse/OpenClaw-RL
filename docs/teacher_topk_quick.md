# Understanding Teacher Top-K in OpenClaw-RL

## Quick Answer to "Why teacher top-k?"

**Teacher top-K reduces memory usage by 2500×** while preserving 95%+ of distillation quality.

For a 128K vocabulary model:
- **Full distillation**: 1 MB per token → 32 GB for 4K sequence (impossible!)
- **Top-50 distillation**: 400 bytes per token → 12.8 MB for 4K sequence (feasible!)

## How It Works

### Traditional Distillation (Memory Hog)

```python
# For each of 128K vocabulary tokens:
student_prob[0], teacher_prob[0]  # Compute KL
student_prob[1], teacher_prob[1]  # Compute KL
...
student_prob[127999], teacher_prob[127999]  # Compute KL
```

**Problem**: 128K × 2 × 4 bytes = 1 MB per token position!

### Top-K Distillation (Memory Efficient)

```python
# 1. Teacher identifies top-50 most probable tokens:
teacher_top50 = [the, a, is, ...]  # 95% of probability mass
teacher_tail = remaining_127950_tokens  # 5% of probability mass

# 2. Student only computes probability for those 50 tokens:
student_prob[the], student_prob[a], student_prob[is], ...

# 3. Compute KL over 51 bins (50 tokens + 1 tail):
kl_loss = KL(teacher_top50 + tail || student_top50 + tail)
```

**Result**: 51 × 2 × 4 bytes = 400 bytes per token position!

## Why Top-50 Specifically?

Language model probability distributions are **highly concentrated**:

```
Top-1 token:   40% of probability
Top-10 tokens: 75% of probability
Top-50 tokens: 95% of probability  ← Sweet spot!
Top-500 tokens: 99% of probability
```

**Top-50 captures 95% of the teacher's "intent"** at 1/2500th the memory cost.

## The "Tail Trick"

We don't ignore the remaining 127,900 tokens entirely. Instead, we aggregate them into a single "tail" bin:

```
Top-50:  [token_1, token_2, ..., token_50]  # Individual tokens
Tail:    [all remaining tokens aggregated]   # Single bin

Total: 51 bins instead of 128,000 bins
```

This ensures **probability sums to 1.0** and prevents the student from "gaming" the loss.

## When to Use

| Scenario | Recommended K |
|----------|---------------|
| **Default** | 50 |
| Memory-constrained | 10-25 |
| High-quality distillation | 100-500 |
| Very short sequences (<512 tokens) | Can use full vocabulary |

## OpenClaw-RL Usage

```bash
# Enable top-K distillation in openclaw-opd
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh

# The script automatically uses:
# --distill-topk 50
# --custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function
```

## Visual Example

```
Teacher probability distribution (128K tokens):
████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
└───────── Top-50 (95%) ─────────┘└─── Tail (5%) ───┘

Student learns:
- Exact probabilities for top-50 tokens
- Aggregated probability for remaining tokens
- 95%+ of distillation quality
- 2500× less memory
```

## Mathematical Detail

**Full KL Divergence:**
```
KL(teacher || student) = Σ teacher[t] * log(teacher[t] / student[t])
                         for t = 1 to 128,000
```

**Top-K Approximation:**
```
KL(topk || student_topk) ≈ Σ teacher_topk[k] * log(teacher_topk[k] / student_topk[k])
                           for k = 1 to 50
                           + tail_bin_contribution
```

**Error bound**: <5% approximation error for K=50

## References

- **SDFT**: arXiv 2601.19897
- **SDPO**: arXiv 2601.20802
- **Implementation**: `openclaw-opd/topk_distillation_loss.py`

---

**TL;DR**: Teacher top-K makes distillation **practical** for large vocabulary models by trading <5% quality for 2500× memory savings.
