"""Ported from SkyRL: skyrl-train/skyrl_train/generators/utils.py

Only the functions needed for Mini-SWE-Agent message post-processing:
  - get_generation_prompt_ids
  - encode_messages_subset
  - get_response_ids_and_loss_mask_from_messages
"""

from typing import List, Optional, Tuple


def get_generation_prompt_ids(tokenizer, chat_template: Optional[str] = None) -> List[int]:
    """Get the generation prompt token ids (e.g. ``<|im_start|>assistant\\n`` for Qwen)."""
    empty_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True, chat_template=chat_template
    )
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, chat_template=chat_template
    )
    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user):]
    return generation_prompt_ids


def encode_messages_subset(messages, tokenizer, chat_template: Optional[str] = None) -> List[int]:
    """Encode a subset of messages using the fixed-base approach.

    Prepends a dummy base conversation so that the tokenizer's chat template
    does not inject an extra default system message, then strips those base
    tokens to yield only the tokens for *messages*.

    Reference: https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
    """
    assert len(messages), "messages list cannot be empty"
    base_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]
    base_conversation_token_ids = tokenizer.apply_chat_template(
        base_conversation,
        add_generation_prompt=False,
        tokenize=True,
        chat_template=chat_template,
    )
    full_conversation = base_conversation + messages
    full_conversation_token_ids = tokenizer.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
        tokenize=True,
        chat_template=chat_template,
    )
    conversation_token_ids = full_conversation_token_ids[len(base_conversation_token_ids):]
    return conversation_token_ids


def get_response_ids_and_loss_mask_from_messages(
    messages, tokenizer, assistant_logprobs=None, chat_template: Optional[str] = None
) -> Tuple[List[int], List[int], Optional[List[float]]]:
    """Build response token ids, loss mask, and (optionally) rollout logprobs
    from the multi-turn messages produced by Mini-SWE-Agent.

    For each message:
      - role == "user"      → all tokens get loss_mask = 0
      - role == "assistant"  → split into three segments:
          [generation_prompt_ids]  → loss_mask = 0  (e.g. <|im_start|>assistant\\n)
          [generated_tokens + EOS] → loss_mask = 1  (content produced by the model)
          [tokens_after_eos]       → loss_mask = 0  (e.g. Qwen's trailing \\n)
    """
    assert len(messages), "messages list cannot be empty"

    generation_prompt_ids = get_generation_prompt_ids(tokenizer, chat_template=chat_template)

    response_ids: List[int] = []
    loss_mask: List[int] = []
    rollout_logprobs: Optional[List[float]] = None if assistant_logprobs is None else []
    assistant_msg_idx = 0

    for i in range(len(messages)):
        cur_message = messages[i]
        cur_token_ids = encode_messages_subset([cur_message], tokenizer, chat_template=chat_template)
        response_ids.extend(cur_token_ids)

        if cur_message["role"] == "user":
            loss_mask.extend([0] * len(cur_token_ids))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(cur_token_ids))

        elif cur_message["role"] == "assistant":
            assert cur_token_ids[:len(generation_prompt_ids)] == generation_prompt_ids, (
                f"Assistant message tokens should start with generation prompt. "
                f"Expected {generation_prompt_ids}, got {cur_token_ids[:len(generation_prompt_ids)]}"
            )

            if tokenizer.eos_token_id in cur_token_ids:
                last_eos_token_index = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(tokenizer.eos_token_id)
                generated_token_ids = cur_token_ids[len(generation_prompt_ids):last_eos_token_index + 1]
                tokens_after_eos = cur_token_ids[last_eos_token_index + 1:]
            else:
                generated_token_ids = cur_token_ids[len(generation_prompt_ids):]
                tokens_after_eos = []

            assert len(generation_prompt_ids) + len(generated_token_ids) + len(tokens_after_eos) == len(cur_token_ids), (
                "The sum of the lengths of the generation prompt IDs, the generated tokens, and the tokens "
                "after the EOS token should equal the length of the current token IDs"
            )

            # generation prompt → mask 0
            loss_mask.extend([0] * len(generation_prompt_ids))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(generation_prompt_ids))

            # generated content → mask 1
            loss_mask.extend([1] * len(generated_token_ids))
            if assistant_logprobs:
                if assistant_msg_idx >= len(assistant_logprobs):
                    raise ValueError(
                        f"Missing logprobs for assistant message #{assistant_msg_idx + 1}. "
                        f"Provided {len(assistant_logprobs)} logprob lists."
                    )
                msg_logprobs = assistant_logprobs[assistant_msg_idx]
                if len(msg_logprobs) != len(generated_token_ids):
                    raise ValueError(
                        f"Logprobs count ({len(msg_logprobs)}) does not match token count "
                        f"({len(generated_token_ids)}) for assistant message #{assistant_msg_idx + 1}."
                    )
                rollout_logprobs.extend(msg_logprobs)

            # tokens after EOS → mask 0
            loss_mask.extend([0] * len(tokens_after_eos))
            if assistant_logprobs:
                rollout_logprobs.extend([0.0] * len(tokens_after_eos))

            assistant_msg_idx += 1
        else:
            raise ValueError(f"Expected message role to be 'user' or 'assistant', got {cur_message['role']}")

        assert len(loss_mask) == len(response_ids)
        assert len(rollout_logprobs) == len(response_ids) if rollout_logprobs is not None else True

    return response_ids, loss_mask, rollout_logprobs
