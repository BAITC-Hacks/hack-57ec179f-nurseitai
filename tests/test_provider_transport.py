"""Exercise the actual SDK over an in-memory HTTP transport, without network."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from assistant import AssistantConfig, create_client, run_turn
from recommender import load_contractors


class ProviderTransportTests(unittest.TestCase):
    def test_actual_sdk_serializes_both_protocols(self):
        contractors = load_contractors(Path(__file__).parents[1] / "data/contractors.csv")
        for provider in ("openai", "nvidia"):
            with self.subTest(provider=provider):
                requests = []

                def handle(request):
                    payload = json.loads(request.content)
                    requests.append((request.url.path, payload))
                    if provider == "openai":
                        output = ([{"type": "function_call", "id": "fc_1", "call_id": "call_1",
                                    "name": "get_search_options", "arguments": "{}", "status": "completed"}]
                                  if len(requests) == 1 else
                                  [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                                    "content": [{"type": "output_text", "text": "Готово", "annotations": []}]}])
                        data = {"id": f"resp_{len(requests)}", "object": "response", "created_at": 1,
                                "status": "completed", "model": "test", "output": output}
                    else:
                        message = ({"role": "assistant", "content": None, "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {
                                "name": "get_search_options", "arguments": "{}"}}]}
                            if len(requests) == 1 else {"role": "assistant", "content": "Готово"})
                        data = {"id": "chat_1", "object": "chat.completion", "created": 1, "model": "test",
                                "choices": [{"index": 0, "message": message,
                                             "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]}
                    return httpx.Response(200, json=data)

                with OpenAI(api_key="fake-test-key", base_url="https://test.invalid/v1",
                            http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
                    turn = run_turn(AssistantConfig(provider, "test", "fake"), contractors,
                                    "Какие города?", client=client)
                self.assertEqual(turn.text, "Готово")
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0][0], "/v1/responses" if provider == "openai" else "/v1/chat/completions")
                second = requests[1][1]
                messages = second["input" if provider == "openai" else "messages"]
                result = messages[-1]
                output = result["output" if provider == "openai" else "content"]
                self.assertIn("Алматы", json.loads(output)["cities"])

    def test_provider_keys_have_explicit_separate_endpoints(self):
        with patch("openai.OpenAI") as constructor:
            for provider, url in (("openai", "https://api.openai.com/v1"),
                                  ("nvidia", "https://integrate.api.nvidia.com/v1")):
                create_client(AssistantConfig(provider, "test", "key-" + provider))
                self.assertEqual(constructor.call_args.kwargs["base_url"], url)
                self.assertEqual(constructor.call_args.kwargs["api_key"], "key-" + provider)
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
