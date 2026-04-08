"""Qwen3.5 agent — adapts Qwen3VLAgentLocal for Qwen3.5's tool call format.

Qwen3.5 uses <function=NAME><parameter=KEY>VALUE</parameter></function> format
instead of Qwen3-VL's JSON format for tool calls. The chat template also injects
its own tool format instructions, so the system prompt should not duplicate them.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.qwen3vl_agent import Qwen3VLAgentLocal
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class Qwen35AgentLocal(Qwen3VLAgentLocal):
    """Qwen3.5 agent with adapted system prompt and tool call parsing."""

    def get_system_prompt(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> str:
        """Task instructions only; tool format is handled by the Qwen3.5 chat template."""
        tools_def = self.get_tool_spec(
            processed_width=processed_width, processed_height=processed_height
        )
        description = tools_def["function"]["description"]
        return (
            f"{description}\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block with the function call.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

    @staticmethod
    def _parse_qwen35_tool_call(block: str) -> Optional[Dict[str, Any]]:
        """Parse a Qwen3.5 <function=NAME>...</function> block into JSON dict."""
        func_match = re.search(r"<function=([^>]+)>", block)
        if not func_match:
            return None
        name = func_match.group(1).strip()
        arguments: Dict[str, Any] = {}
        for param_match in re.finditer(
            r"<parameter=([^>]+)>(.*?)</parameter>", block, re.DOTALL
        ):
            key = param_match.group(1).strip()
            raw_value = param_match.group(2).strip()
            # Try to parse as JSON (for lists, numbers, etc.)
            try:
                arguments[key] = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = raw_value
        return {"name": name, "arguments": arguments}

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Parse Qwen3.5 XML tool calls, convert to JSON, then use parent logic."""
        if response is None or not response.strip():
            return "", [], {"raw_response": response, "tool_calls":[]}

        # Strip trailing <|im_end|> if present
        response = response.replace("<|im_end|>", "").strip()

        # Convert Qwen3.5 XML tool calls to Qwen3-VL JSON format
        converted = response
        for tc_match in re.finditer(
            r"<tool_call>(.*?)</tool_call>", response, re.DOTALL
        ):
            tc_block = tc_match.group(1)
            parsed = self._parse_qwen35_tool_call(tc_block)
            if parsed:
                json_str = json.dumps(parsed, ensure_ascii=False)
                converted = converted.replace(
                    tc_match.group(0),
                    f"<tool_call>\n{json_str}\n</tool_call>",
                )

        return super().parse_response(
            converted,
            original_width,
            original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

    def build_train_data(
        self,
        *,
        args: Any,
        state: GenerateState,
        train_messages: List[Dict[str, Any]],
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """Override to force qwen3 loss mask type for Qwen3.5 template compatibility."""
        tokenizer = state.tokenizer
        processor = state.processor
        text_prompt = tokenizer.apply_chat_template(
            train_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=[tool_spec or self.get_tool_spec()],
        )
        if processor:
            multimodal_inputs = self._extract_multimodal(train_messages, processor)
            kwargs: Dict[str, Any] = {"text": [text_prompt], "return_tensors": "pt", **multimodal_inputs}
            proc_out = processor(**kwargs)
            input_ids = proc_out["input_ids"][0].tolist()
            
            ignore_keys = ["input_ids", "attention_mask", "token_type_ids"]
            mm_train = {k: v for k, v in proc_out.items() if k not in ignore_keys} or None
        else:
            input_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            mm_train = None

        # Qwen3.5 template requires user query in messages; "qwen3" mask type handles this
        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
        _, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
            train_messages, input_ids, tools=[tool_spec or self.get_tool_spec()]
        )
        return input_ids, loss_mask, mm_train
