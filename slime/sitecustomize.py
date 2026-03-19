import os


if os.environ.get("SLIME_ENABLE_QWEN35_SGLANG_PATCH") == "1":
    try:
        from slime.backends.sglang_utils.qwen3_5 import patch_sglang_qwen35

        patch_sglang_qwen35()
    except Exception:
        pass
