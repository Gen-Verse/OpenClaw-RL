import logging
import math

from megatron.training.arguments import parse_args, validate_args

try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
except ModuleNotFoundError:

    def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
        """Fallback for newer Megatron-Core layouts without legacy tokenizer module."""
        after = orig_vocab_size
        multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
        after = int(math.ceil(after / multiple) * multiple)
        if getattr(args, "rank", 0) == 0 and logging_enabled:
            print(
                f" > padded vocab (size: {orig_vocab_size}) with {after - orig_vocab_size} dummy tokens "
                f"(new size: {after})",
                flush=True,
            )
        return after

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # Newer Megatron-Core uses layernorm_epsilon as the argparse destination
    # for --norm-epsilon. Keep the older alias for downstream code.
    if not hasattr(args, "norm_epsilon") and hasattr(args, "layernorm_epsilon"):
        args.norm_epsilon = args.layernorm_epsilon
    # Newer Megatron-Core exposes use_gloo_process_groups while this repo still
    # reads enable_gloo_process_groups in a few places.
    if not hasattr(args, "enable_gloo_process_groups") and hasattr(args, "use_gloo_process_groups"):
        args.enable_gloo_process_groups = args.use_gloo_process_groups

    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
