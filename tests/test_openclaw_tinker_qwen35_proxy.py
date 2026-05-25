from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

from fastapi.testclient import TestClient

OPENCLAW_TINKER_DIR = Path(__file__).resolve().parents[1] / "openclaw-tinker"
sys.path.insert(0, str(OPENCLAW_TINKER_DIR))

from api_server import _BaseServer, _extract_tool_calls, _normalize_messages  # noqa: E402
from config import TinkerConfig  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    }
]


class _FakeSequence:
    def __init__(self, tokens, logprobs, stop_reason="stop"):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason


class _FakeSampleResponse:
    def __init__(self, sequence):
        self.sequences = [sequence]


class _FakeSamplingClient:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSampleResponse(self.sequence)


class _CapturingTokenizer:
    eos_token_id = 0

    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.last_template_messages = None
        self.last_template_tools = None
        self.last_template_enable_thinking = None
        self.last_template_tool_choice = None

    def encode(self, text: str, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) % 128 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        del tokens, skip_special_tokens
        return self.decoded_text

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
        tool_choice=None,
    ):
        self.last_template_messages = messages
        self.last_template_tools = tools
        self.last_template_enable_thinking = enable_thinking
        self.last_template_tool_choice = tool_choice
        del tokenize, add_generation_prompt
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)


class _HarnessServer(_BaseServer):
    def __init__(self, *, tokenizer, sampling_client, record_dir: str):
        self._test_tokenizer = tokenizer
        submission_enabled = threading.Event()
        submission_enabled.set()
        super().__init__(
            TinkerConfig(
                model_name="Qwen/Qwen3.5-4B",
                proxy_host="127.0.0.1",
                proxy_port=30000,
                served_model_name="qwen3.5-local",
                record_dir=record_dir,
            ),
            queue.Queue(),
            submission_enabled,
            sampling_client=sampling_client,
        )

    def _load_tokenizer(self):
        return self._test_tokenizer

    async def _handle_request(self, body: dict, session_id: str, turn_type: str, session_done: bool) -> dict:
        del session_id, turn_type, session_done
        return {"response": await self._forward_to_tinker(body)}


def _make_client(tmp_path, decoded_text: str):
    tokenizer = _CapturingTokenizer(decoded_text)
    sampling_client = _FakeSamplingClient(_FakeSequence(tokens=[1, 2, 3], logprobs=[-0.3, -0.4, -0.5]))
    server = _HarnessServer(
        tokenizer=tokenizer,
        sampling_client=sampling_client,
        record_dir=str(tmp_path / "records"),
    )
    return server, TestClient(server.app), tokenizer


def _sse_events(response) -> list[str]:
    return [line[6:] for line in response.iter_lines() if line.startswith("data: ")]


def test_extract_tool_calls_supports_qwen35_xml():
    content, tool_calls = _extract_tool_calls(
        '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>'
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_0"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_normalize_messages_converts_replayed_tool_arguments_to_dict():
    normalized = _normalize_messages(
        [
            {
                "role": "developer",
                "content": "be helpful",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_prev",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
                    }
                ],
            },
            {"role": "tool", "content": '{"temp_c":20}', "tool_call_id": "call_prev"},
        ]
    )

    assert normalized[0]["role"] == "system"
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"location": "Tokyo"}
    assert normalized[2]["tool_call_id"] == "call_prev"


def test_qwen35_tool_calls_are_parsed_when_tools_are_present(tmp_path):
    _server, client, _tokenizer = _make_client(
        tmp_path,
        '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>',
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] is None
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Tokyo"
    }


def test_tool_choice_none_keeps_qwen35_tool_markup_as_plain_text(tmp_path):
    _server, client, tokenizer = _make_client(
        tmp_path,
        '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>',
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["message"]["content"] == (
        '<tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>'
    )
    assert tokenizer.last_template_tools is None


def test_streaming_qwen35_tool_call_only_emits_tool_call_delta(tmp_path):
    _server, client, _tokenizer = _make_client(
        tmp_path,
        '<think>scratchpad</think><tool_call><function=get_weather><parameter=location>"Tokyo"</parameter></function></tool_call>',
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "stream": True,
            "max_tokens": 16,
        },
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    payloads = [json.loads(event) for event in events[:-1]]
    first_delta = payloads[0]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert "content" not in first_delta
    assert first_delta["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "[DONE]"


def test_reasoning_is_returned_separately_from_visible_content(tmp_path):
    _server, client, tokenizer = _make_client(
        tmp_path,
        "<think>private chain of thought</think>\nack",
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "Reply with exactly ack."}],
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert tokenizer.last_template_enable_thinking is True
    assert message["content"] == "ack"
    assert message["reasoning_content"] == "private chain of thought"


def test_enable_thinking_false_is_forwarded_and_suppresses_reasoning_field(tmp_path):
    _server, client, tokenizer = _make_client(
        tmp_path,
        "ack",
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "user", "content": "Reply with exactly ack."}],
            "max_tokens": 16,
            "extra_body": {"enable_thinking": False},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert tokenizer.last_template_enable_thinking is False
    assert message["content"] == "ack"
    assert "reasoning_content" not in message
