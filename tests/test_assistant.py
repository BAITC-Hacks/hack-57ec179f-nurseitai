import json
import unittest
from datetime import date
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from assistant import (AssistantConfig, AssistantError, SearchTools, run_turn,
                       request_dict, agent_instructions, MAX_CALLS, MAX_ROUNDS)
from recommender import load_contractors


class Item(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self).copy()


def openai_reply(calls=(), text="Готово"):
    output = [Item(type="function_call", call_id=f"call_{i}", name=name, arguments=args)
              for i, (name, args) in enumerate(calls)]
    if not calls:
        output.append(Item(type="message", role="assistant", content=[{"type": "output_text", "text": text}]))
    return SimpleNamespace(output=output, output_text="" if calls else text, status="completed")


def nvidia_reply(calls=(), text="Готово"):
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        item = Mock(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=args))
        item.model_dump.return_value = {"id": item.id, "type": "function",
                                        "function": {"name": name, "arguments": args}}
        tool_calls.append(item)
    message = SimpleNamespace(content="" if calls else text, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")])


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contractors = load_contractors(Path(__file__).parents[1] / "data/contractors.csv")

    def setUp(self):
        self.tools = SearchTools(self.contractors)
        self.args = dict(city="Алматы", event_date="2026-10-03", event_format="свадьба",
                         category="Ведущий", budget=1_500_000, duration_hours=6,
                         language="русский", preferences="квантовый реактор на Марсе")

    def test_search_returns_verified_alternative_and_keeps_original_date(self):
        result = self.tools.dispatch("search_contractors", json.dumps(self.args))
        self.assertEqual(result["status"], "conditions_not_met")
        alternative = result["suggestions"][0]
        self.assertEqual(alternative["request"]["event_date"], "2026-10-04")
        self.assertEqual(alternative["count"], 2)
        self.assertEqual(result["request"]["event_date"], "2026-10-03")
        found = self.tools.dispatch("search_contractors", json.dumps(alternative["request"]))
        self.assertEqual(len(found["recommendations"]), 2)
        self.assertTrue(all("пожелание не подтверждено" in r["explanation"] for r in found["recommendations"]))

    def test_unspecified_budget_returns_all_available_wedding_hosts(self):
        args = dict(city="Алматы", event_date="2026-09-23", event_format="свадьба", category="Ведущий")
        result = self.tools.dispatch("search_contractors", json.dumps(args))
        self.assertEqual(result["status"], "matched")
        self.assertIsNone(result["request"]["budget"])
        self.assertIsNone(result["request"]["language"])
        self.assertIsNone(result["request"]["duration_hours"])
        self.assertEqual(result["matched_count"], 4)
        self.assertEqual({r["name"] for r in result["recommendations"]},
                         {"Мицури Канроджи", "Эмилия", "Сон Гоку", "Софи Хаттер"})
        self.assertNotIn("дороже бюджета", result["rejection_counts"])
        self.assertFalse(result["suggestions"])
        self.assertTrue(all("бюджет не ограничен" in r["explanation"] for r in result["recommendations"]))
        explicit_null = self.tools.dispatch("search_contractors", json.dumps({**args, "budget": None,
                      "language": None, "duration_hours": None, "preferences": ""}))
        self.assertEqual(result, explicit_null)

    def test_explicit_budget_still_filters_and_empty_message_distinguishes_catalog(self):
        args = {**self.args, "event_date": "2026-09-23", "budget": 1_000_000,
                "language": None, "duration_hours": None, "preferences": ""}
        result = self.tools.dispatch("search_contractors", json.dumps(args))
        self.assertEqual(result["matched_count"], 3)
        self.assertTrue(all(r["price_from_kzt"] <= 1_000_000 for r in result["recommendations"]))
        empty = self.tools.dispatch("search_contractors", json.dumps({**args, "budget": 100_000}))
        self.assertEqual(empty["matched_count"], 0)
        self.assertIn("В каталоге", empty["message"])
        self.assertIn("по заданным условиям подходящих нет", empty["message"])
        self.assertNotIn("Кандидаты есть", empty["message"])

    def test_today_is_explicit_for_both_providers(self):
        fixed_today = date(2026, 10, 14)
        self.assertIn("Сегодня = 2026-10-14, завтра = 2026-10-15", agent_instructions(fixed_today))
        for provider, reply in (("openai", openai_reply), ("nvidia", nvidia_reply)):
            client = Mock()
            method = client.responses.create if provider == "openai" else client.chat.completions.create
            method.return_value = reply(text="Уточните город")
            with patch("assistant.current_event_date", return_value=fixed_today):
                run_turn(AssistantConfig(provider, "test", "fake"), self.contractors, "На сегодня", client=client)
            instructions = (method.call_args.kwargs["instructions"] if provider == "openai"
                            else method.call_args.kwargs["messages"][0]["content"])
            self.assertIn("Сегодня = 2026-10-14", instructions)
            self.assertIn("budget=null", instructions)

    def test_unlimited_budget_empty_result_only_suggests_date(self):
        args = {**self.args, "budget": None}
        pool = [replace(c, busy_dates=c.busy_dates | {args["event_date"]}) for c in self.contractors]
        result = SearchTools(pool).dispatch("search_contractors", json.dumps(args))
        self.assertEqual(result["status"], "conditions_not_met")
        self.assertTrue(result["suggestions"])
        self.assertTrue(all(s["request"]["budget"] is None for s in result["suggestions"]))
        self.assertTrue(all(s["request"]["event_date"] != args["event_date"] for s in result["suggestions"]))

    def test_validation_rejects_malformed_missing_extra_and_out_of_range_arguments(self):
        for raw in ("{", "[]", "null", "{}", json.dumps({**self.args, "code": "print(1)"})):
            with self.subTest(raw=raw):
                self.assertIn("error", self.tools.dispatch("search_contractors", raw))
        for field, value in (("budget", 0), ("budget", True), ("budget", "1500000"),
                             ("duration_hours", -1), ("duration_hours", 1.5),
                             ("city", "Выдуманный город"), ("event_date", "2027-01-01"),
                             ("event_date", "2026-02-30"), ("preferences", [])):
            with self.subTest(field=field, value=value):
                self.assertIn("error", self.tools.dispatch("search_contractors", json.dumps({**self.args, field: value})))

    def test_allowlist_and_profile_lookup(self):
        self.assertIn("error", self.tools.dispatch("exec", '{"code":"print(1)"}'))
        self.assertIn("error", self.tools.dispatch("get_contractor", '{"contractor_id":"missing"}'))
        item = self.contractors[0]
        result = self.tools.dispatch("get_contractor", json.dumps({"contractor_id": item.id}))
        self.assertEqual(result["description"], item.description)
        self.assertNotIn("busy_dates", result)
        self.assertEqual(self.tools.dispatch("get_search_options", "{}")["calendar_end"], "2026-12-31")

    def test_both_provider_protocols_return_results_to_matching_tool_call(self):
        for provider, reply in (("openai", openai_reply), ("nvidia", nvidia_reply)):
            with self.subTest(provider=provider):
                client = Mock()
                method = client.responses.create if provider == "openai" else client.chat.completions.create
                method.side_effect = [reply([("search_contractors", json.dumps(self.args))]), reply(text="Можно перенести дату.")]
                original = [{"role": "user", "content": "Нужен ведущий"}]
                turn = run_turn(AssistantConfig(provider, "test-model", "fake-key"), self.contractors,
                                "3 октября", original, client=client)
                self.assertEqual(turn.text, "Можно перенести дату.")
                self.assertEqual(len(original), 1)
                self.assertEqual(turn.tool_results[0]["result"]["status"], "conditions_not_met")
                payload = method.call_args.kwargs["input" if provider == "openai" else "messages"]
                results = [m for m in payload if m.get("type") == "function_call_output" or m.get("role") == "tool"]
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].get("call_id", results[0].get("tool_call_id")), "call_0")
                self.assertNotIn("fake-key", json.dumps(turn.history))

    def test_reasoning_items_are_preserved_and_storage_is_disabled(self):
        client = Mock()
        response = openai_reply([("get_search_options", "{}")])
        response.output.insert(0, Item(type="reasoning", encrypted_content="opaque", summary=[]))
        client.responses.create.side_effect = [response, openai_reply()]
        run_turn(AssistantConfig("openai", "test", "fake"), self.contractors, "Какие города?", client=client)
        self.assertFalse(client.responses.create.call_args.kwargs["store"])
        self.assertEqual(client.responses.create.call_args.kwargs["input"][1]["type"], "reasoning")

    def test_incomplete_request_can_be_clarified_without_calling_tools(self):
        client = Mock()
        client.responses.create.return_value = openai_reply(text="В каком городе и на какую дату?")
        turn = run_turn(AssistantConfig("openai", "test", "fake"), self.contractors, "Нужен ведущий", client=client)
        self.assertFalse(turn.tool_results)
        self.assertIn("дату", turn.text)

    def test_invalid_arguments_are_returned_to_model_for_correction(self):
        client = Mock()
        client.responses.create.side_effect = [openai_reply([("search_contractors", "{}")] ),
                                               openai_reply(text="Уточните условия")]
        turn = run_turn(AssistantConfig("openai", "test", "fake"), self.contractors, "Найди", client=client)
        self.assertIn("error", turn.tool_results[0]["result"])

    def test_loop_limit_and_call_limit(self):
        for count in (1, MAX_CALLS + 1):
            client = Mock()
            client.responses.create.return_value = openai_reply([("get_search_options", "{}")] * count)
            with self.assertRaises(AssistantError):
                run_turn(AssistantConfig("openai", "test", "fake"), self.contractors, "Найди", client=client)
            self.assertLessEqual(client.responses.create.call_count, MAX_ROUNDS)

    def test_upstream_errors_do_not_leak_secrets_or_corrupt_history(self):
        client = Mock()
        error = RuntimeError("SECRET_KEY provider internals")
        error.status_code = 401
        client.responses.create.side_effect = error
        history = [{"role": "user", "content": "Привет"}]
        with self.assertRaises(AssistantError) as raised:
            run_turn(AssistantConfig("openai", "test", "fake"), self.contractors, "Найди", history, client=client)
        self.assertNotIn("SECRET", str(raised.exception))
        self.assertEqual(len(history), 1)

    def test_config_separates_credentials_and_reports_missing_key(self):
        settings = {"OPENAI_API_KEY": "openai-test", "NVIDIA_API_KEY": "nvidia-test"}
        self.assertEqual(AssistantConfig.from_settings("nvidia", settings).api_key, "nvidia-test")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(AssistantError):
                AssistantConfig.from_settings("openai")

    def test_round_trip_request(self):
        self.assertEqual(request_dict(self.tools.parse_request(self.args)), self.args)
