import base64
import json
import logging
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agents.utils.qwen_vl_utils import smart_resize
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.processing_utils import process_vision_info as slime_process_vision_info

logger = None


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def process_image(image_bytes: bytes) -> str:
    """Resize + re-encode screenshot and return base64 PNG."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8")


class Qwen35VLAgentLocal:
    """
    Lightweight Qwen3.5-VL agent (local sglang variant).

    Characteristics:
    - XML tool-call output format.
    - History truncation by `history_n`.
    - Old screenshot folding by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen35-vl",
        max_steps: int = 100,
        max_image_history_length: int = 20,
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coordinate_type: str = "relative",
        example_result_dir: Optional[str] = None,
        add_thought_prefix: bool = False,
        history_n: int = 100,
        fold_size: int = 10,
        collapse_text: Optional[str] = None,
        **_unused_kwargs: Any,
    ):
        self.platform = platform
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.image_max = max(1, int(max_image_history_length))
        self.coordinate_type = coordinate_type
        self.example_result_dir = example_result_dir or os.getcwd()
        self.history_n = int(history_n)
        self.fold_size = max(1, int(fold_size))
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT

        assert action_space == "pyautogui", "qwen35vl_agent only supports pyautogui action space"
        assert observation_type == "screenshot", "qwen35vl_agent only supports screenshot observations"

        self.actions: List[str] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen35_agent_local")

        self.actions = []
        self.responses = []
        self.screenshots = []
        self.folded_prefix_k = 0

    @staticmethod
    def _py_string(text: str) -> str:
        return json.dumps("" if text is None else str(text), ensure_ascii=False)

    def _update_folding_state(self, total_screenshots: int) -> None:
        while (total_screenshots - self.folded_prefix_k) > self.image_max:
            self.folded_prefix_k += self.fold_size
        if self.folded_prefix_k > total_screenshots:
            self.folded_prefix_k = total_screenshots

    def _should_collapse_step(self, step_num_1based: int) -> bool:
        return step_num_1based <= self.folded_prefix_k

    def _wrap_tool_response(self, parts: List[Dict]) -> List[Dict]:
        return (
            [{"type": "text", "text": "<tool_response>\n"}]
            + parts
            + [{"type": "text", "text": "\n</tool_response>"}]
        )

    def get_tool_spec(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if self.coordinate_type == "absolute" and processed_width and processed_height
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """\
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

        return {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "scroll",
                                "hscroll",
                                "wait",
                                "terminate",
                                "answer",
                            ],
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`."},
                        "text": {
                            "type": "string",
                            "description": (
                                "Required by `action=type` and `action=answer`. "
                                "Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) "
                                "to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). "
                                "Optional for scroll actions (scroll, hscroll) to specify a modifier key "
                                "(e.g., 'shift', 'ctrl') to hold during scrolling."
                            ),
                        },
                        "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                        "pixels": {"type": "number", "description": "Scroll amount."},
                        "time": {"type": "number", "description": "Seconds to wait."},
                        "status": {
                            "type": "string",
                            "description": "Task status for terminate.",
                            "enum": ["success", "failure"],
                        },
                    },
                },
            },
        }

    def get_system_prompt(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> str:
        tools_def = self.get_tool_spec(processed_width=processed_width, processed_height=processed_height)
        return (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def)
            + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
            f"- Collapsed screenshots appear as text: {self.collapse_text}\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

    def build_train_system_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.get_system_prompt()}

    def build_instruction_prompt(self, instruction: str, previous_actions_text: List[str]) -> str:
        prev = "\n".join(previous_actions_text) if previous_actions_text else "None"
        return (
            f"\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n"
            f"{prev}"
        )

    @staticmethod
    def _extract_multimodal(messages: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
        if not processor:
            return {}
        return slime_process_vision_info(messages, processor) or {}

    def build_policy_messages(self, instruction: str, obs: Dict) -> Dict[str, Any]:
        """
        Build policy messages with a bounded recent-history window.

        Keep the same retention rule as the working Qwen3VL GUI path:
        only the latest `max_image_history_length - 1` completed steps plus
        the current screenshot are included in the prompt. Older steps are
        summarized only through `Previous actions`.
        """
        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        all_screenshots = list(self.screenshots) + [processed_image_b64]
        total_steps = len(all_screenshots)

        # Bound retained context to the same recent-history window used by
        # qwen3vl GUI training; otherwise response_length grows cumulatively
        # with old assistant turns and breaks training-time packing.
        start_step = max(1, total_steps - self.image_max + 1)

        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}"
            for i in range(0, min(start_step - 1, len(self.actions)))
        ]

        system_prompt = self.get_system_prompt(
            processed_width=processed_width, processed_height=processed_height
        )
        tool_spec = self.get_tool_spec(
            processed_width=processed_width, processed_height=processed_height
        )
        instruction_prompt = self.build_instruction_prompt(
            instruction=instruction, previous_actions_text=previous_actions
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        image_traj: List[str] = []

        for step_num in range(start_step, total_steps + 1):
            is_first_turn = step_num == start_step
            screenshot_idx = step_num - 1

            img_url = f"data:image/png;base64,{all_screenshots[screenshot_idx]}"
            if is_first_turn:
                user_content = [
                    {"type": "image", "image": img_url},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = self._wrap_tool_response(
                    [{"type": "image", "image": img_url}]
                )
            messages.append({"role": "user", "content": user_content})
            image_traj.append(
                os.path.join(self.example_result_dir, f"step_{screenshot_idx}.png")
            )

            if step_num <= total_steps - 1 and (step_num - 1) < len(self.responses):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": self.responses[step_num - 1]}],
                    }
                )

        return {
            "messages": messages,
            "image_traj": image_traj,
            "step_index": step_index,
            "processed_image_b64": processed_image_b64,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": processed_width,
            "processed_height": processed_height,
            "system_prompt": system_prompt,
            "tool_spec": tool_spec,
        }

    async def generate_with_sglang(
        self,
        *,
        args: Any,
        state: GenerateState,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        sampling_seed: int | None = None,
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[str, str]:
        tokenizer = state.tokenizer
        processor = state.processor
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=[tool_spec or self.get_tool_spec()],
        )
        current_sampling_params = dict(sampling_params)
        if sampling_seed is not None:
            current_sampling_params["sampling_seed"] = int(sampling_seed)

        payload: Dict[str, Any] = {"sampling_params": current_sampling_params, "return_logprob": True}
        image_data: List[str] = []
        if processor:
            multimodal_inputs = self._extract_multimodal(messages, processor)
            images = multimodal_inputs.get("images") or []
            if images:
                image_data = [encode_image_for_rollout_engine(img) for img in images]

        if image_data:
            payload["text"] = prompt_text
            payload["image_data"] = image_data
        else:
            payload["input_ids"] = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        output = await post(url, payload)
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if "output_token_logprobs" in output["meta_info"]:
            output_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            response = tokenizer.decode(output_tokens)
        else:
            response = output.get("text", "")
        return response, finish_type

    def build_train_data(
        self,
        *,
        args: Any,
        state: GenerateState,
        train_messages: List[Dict[str, Any]],
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], List[int], Dict[str, Any] | None]:
        tokenizer = state.tokenizer
        processor = state.processor

        # Qwen3.5's chat template unconditionally requires at least one
        # non-tool_response user message (enforced at template line 79).
        # In abort/fallback scenarios generate_with_gui passes system-only
        # messages. Pad with dummy turns whose step_loss_mask=0 so they
        # never contribute to training loss.
        if not any(m.get("role") == "user" for m in train_messages):
            train_messages = list(train_messages) + [
                {"role": "user", "content": "N/A"},
                {"role": "assistant", "content": "N/A", "step_loss_mask": 0},
            ]

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
            mm_train = {k: v for k, v in proc_out.items() if k not in ["input_ids", "attention_mask"]} or None
        else:
            input_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            mm_train = None

        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
        _, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
            train_messages, input_ids, tools=[tool_spec or self.get_tool_spec()]
        )
        return input_ids, loss_mask, mm_train

    def record_policy_turn(self, *, action_text: str, response: str, screenshot_bytes: bytes) -> None:
        self.actions.append(action_text)
        self.responses.append(response)
        self.screenshots.append(process_image(screenshot_bytes))

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []
        other: Dict[str, Any] = {"raw_response": response, "tool_calls": []}

        if response is None or not response.strip():
            return low_level_instruction, pyautogui_code, other

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            if not (original_width and original_height):
                return int(x), int(y)
            if self.coordinate_type == "absolute":
                if processed_width and processed_height:
                    x_scale = original_width / processed_width
                    y_scale = original_height / processed_height
                    return int(x * x_scale), int(y * y_scale)
                return int(x), int(y)
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
            params: Dict = {}
            func_match = re.search(r"<function=([^>]+)>", xml_content)
            if not func_match or func_match.group(1) != "computer_use":
                return None
            for match in re.finditer(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL
            ):
                name = match.group(1)
                value = match.group(2).strip()
                if value.startswith("[") or value.startswith("{"):
                    try:
                        params[name] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                params[name] = value
            return params

        def parse_keys(raw_keys):
            if isinstance(raw_keys, str):
                try:
                    raw_keys = json.loads(raw_keys)
                except Exception:
                    raw_keys = [raw_keys]
            if isinstance(raw_keys, list):
                return [str(key).strip() for key in raw_keys]
            return [str(raw_keys).strip()]

        def parse_coordinate(raw_coord):
            if isinstance(raw_coord, str):
                try:
                    raw_coord = json.loads(raw_coord)
                except Exception:
                    return None
            if isinstance(raw_coord, list) and len(raw_coord) >= 2:
                return raw_coord[0], raw_coord[1]
            return None

        def process_tool_call_params(params: Dict) -> None:
            action = params.get("action")
            if not action:
                return

            other["tool_calls"].append(params)
            coordinate = parse_coordinate(params.get("coordinate"))
            text = params.get("text")

            def press_modifier_keys() -> None:
                if text:
                    for key in str(text).split("+"):
                        key = key.strip().lower()
                        if key:
                            pyautogui_code.append(f"pyautogui.keyDown({self._py_string(key)})")

            def release_modifier_keys() -> None:
                if text:
                    keys = [key.strip().lower() for key in str(text).split("+") if key.strip()]
                    for key in reversed(keys):
                        pyautogui_code.append(f"pyautogui.keyUp({self._py_string(key)})")

            if action == "left_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.click({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.click()")
                release_modifier_keys()

            elif action == "right_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.rightClick()")
                release_modifier_keys()

            elif action == "middle_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.middleClick()")
                release_modifier_keys()

            elif action == "double_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()

            elif action == "triple_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()

            elif action == "type":
                text = params.get("text", "")
                pyautogui_code.append(f"pyautogui.typewrite({self._py_string(text)})")

            elif action == "key":
                keys = parse_keys(params.get("keys", []))
                keys_str = ", ".join(self._py_string(key) for key in keys)
                if len(keys) > 1:
                    pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                else:
                    pyautogui_code.append(f"pyautogui.press({keys_str})")

            elif action in {"scroll", "hscroll"}:
                press_modifier_keys()
                pixels = params.get("pixels", 0)
                try:
                    pixels = int(float(pixels))
                except Exception:
                    pixels = 0
                pyautogui_code.append(f"pyautogui.scroll({pixels})")
                release_modifier_keys()

            elif action == "wait":
                pyautogui_code.append("WAIT")

            elif action in {"terminate", "answer"}:
                pyautogui_code.append("DONE")

            elif action == "mouse_move":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.moveTo(0, 0)")

            elif action == "left_click_drag":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    duration = 0.5
                    if "duration" in params:
                        try:
                            duration = float(params["duration"])
                        except Exception:
                            duration = 0.5
                    pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
                else:
                    pyautogui_code.append("pyautogui.dragTo(0, 0)")

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                low_level_instruction = stripped.split("Action:", 1)[-1].strip()
                break

        for tool_call_match in re.finditer(
            r"<tool_call>(.*?)</tool_call>", response, re.DOTALL
        ):
            params = parse_xml_tool_call(tool_call_match.group(1))
            if params:
                process_tool_call_params(params)

        if not low_level_instruction and pyautogui_code:
            first_code = pyautogui_code[0]
            if first_code == "DONE":
                low_level_instruction = "Task completed"
            elif first_code == "WAIT":
                low_level_instruction = "Waiting"
            elif "." in first_code:
                low_level_instruction = f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
            else:
                low_level_instruction = "Performing action"

        other["action"] = low_level_instruction
        other["code"] = pyautogui_code
        return low_level_instruction, pyautogui_code, other
