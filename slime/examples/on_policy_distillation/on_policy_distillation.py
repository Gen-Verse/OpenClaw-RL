import aiohttp
import torch

from slime.rollout.rm_hub.math_dapo_utils import compute_score
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    K = getattr(args, "distill_topk", 50)
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": K,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

    if getattr(sample, "label", None) is not None:
        score_result = compute_score(sample.response, sample.label)
        result["acc"] = score_result["acc"]

    return result


def post_process_rewards(args, samples: list[Sample], **kwargs):
    K = getattr(args, "distill_topk", 50)
    rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    meta_key = "input_top_logprobs"
    use_topk = meta_key in rewards[0].get("meta_info", {})

    if not use_topk:
        # Fallback: legacy 1D path (no top-K available from server)
        teacher_log_probs = [
            torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            for reward in rewards
        ]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
            sample.teacher_log_probs = t_log_probs
        return teacher_log_probs, teacher_log_probs

    # Top-K path: parse [T, K] logprobs and indices
    teacher_log_probs_2d = []
    teacher_topk_indices_2d = []
    for reward in rewards:
        input_top_logprobs = reward["meta_info"][meta_key]
        # Each position returns a list of K tuples: (logprob, token_id, token_text)
        # Skip position 0 (BOS with no context)
        pos_list = input_top_logprobs[1:] if len(input_top_logprobs) > 1 else input_top_logprobs
        row_logprobs = []
        row_indices = []
        for pos_data in pos_list:
            lp_row = []
            idx_row = []
            if isinstance(pos_data, (list, tuple)):
                for entry in pos_data:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        lp_row.append(float(entry[0]) if entry[0] is not None else 0.0)
                        idx_row.append(int(entry[1]))
                    elif isinstance(entry, dict):
                        lp_row.append(float(entry.get("logprob", 0.0)))
                        idx_row.append(int(entry.get("token_id", 0)))
                    else:
                        lp_row.append(0.0)
                        idx_row.append(0)
            # Pad or truncate to K
            while len(lp_row) < K:
                lp_row.append(0.0)
                idx_row.append(0)
            row_logprobs.append(lp_row[:K])
            row_indices.append(idx_row[:K])
        teacher_log_probs_2d.append(torch.tensor(row_logprobs, dtype=torch.float32))
        teacher_topk_indices_2d.append(torch.tensor(row_indices, dtype=torch.long))

    # Trim to response length
    teacher_log_probs_2d = [
        t[-response_length:]
        for t, response_length in zip(teacher_log_probs_2d, response_lengths, strict=False)
    ]
    teacher_topk_indices_2d = [
        t[-response_length:]
        for t, response_length in zip(teacher_topk_indices_2d, response_lengths, strict=False)
    ]

    for sample, t_log_probs, t_indices in zip(
        samples, teacher_log_probs_2d, teacher_topk_indices_2d, strict=False
    ):
        sample.teacher_log_probs = t_log_probs  # [T, K]
        sample.teacher_topk_indices = t_indices  # [T, K]

    return teacher_log_probs_2d, teacher_log_probs_2d
