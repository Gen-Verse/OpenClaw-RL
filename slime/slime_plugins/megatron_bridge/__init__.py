# rope_theta_compat: monkey-patch Qwen3Config for transformers>=5.0
import os

try:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config

    _qwen3_orig_init = _Qwen3Config.__init__

    def _qwen3_patched_init(self, *args, **kwargs):
        _rope_theta = kwargs.pop("rope_theta", None)
        _qwen3_orig_init(self, *args, **kwargs)
        if not hasattr(self, "rope_theta") or self.rope_theta is None:
            self.rope_theta = _rope_theta if _rope_theta is not None else 1000000

    _Qwen3Config.__init__ = _qwen3_patched_init
except Exception:
    pass

if os.getenv("SLIME_QWEN35_TEXT_ONLY_BRIDGE", "").lower() in {"1", "true", "yes", "on"}:
    import slime_plugins.megatron_bridge.qwen3_5_text  # noqa: F401  # register text-only Qwen3.5 bridge
else:
    import slime_plugins.megatron_bridge.qwen3_5  # noqa: F401  # register multimodal Qwen3.5 bridge
