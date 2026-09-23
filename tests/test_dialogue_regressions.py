"""Isolated catalog + API stubs: no credentials, network or production data edits."""
import json
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import date
from unittest.mock import Mock

from assistant import AssistantConfig, SearchTools, run_turn
from dialogue import empty_state, prepare_turn, safe_text, state_from_request, question_fields, extract_date_window
from recommender import Contractor, SearchRequest, recommend
from test_assistant import openai_reply, nvidia_reply


class DialogueRegressions(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 10, 14)
        base = Contractor("photo", "Фото", ("Фотограф",), "Алматы", False, 100_000,
                          ("той",), ("русский",), 8, frozenset(), "Фотограф мероприятия.")
        self.catalog = [base, replace(base, id="video", name="Видео", categories=("Видеограф",)),
                        replace(base, id="both", name="Обе услуги", categories=("Фотограф", "Видеограф")),
                        replace(base, id="booth", name="Будка", categories=("Фото и видеобудки",), event_formats=("корпоратив",))]
        self.service = SearchTools(self.catalog, today=self.today)
        self.request = dict(city="Алматы", event_date="2026-10-14", event_format="той", category="Фотограф",
                            budget=200_000, language="русский", duration_hours=6, preferences="")
        self.config = AssistantConfig("openai", "test", "fake")

    def turn(self, text, state=None, response=None, service=None):
        client = Mock()
        client.responses.create.return_value = response or openai_reply(text="Уточните условия.")
        turn = run_turn(self.config, (service or self.service).contractors, text, client=client,
                        search_tools=service or self.service, state=state)
        return turn, client

    def test_unknown_need_is_resolved_before_event_questions(self):
        for need in ("Нужен тестировщик", "Нужен программист", "Другое"):
            turn, client = self.turn(need)
            self.assertEqual(turn.text.count("?"), 1)
            self.assertIn("задачу", turn.text)
            self.assertNotIn("бюджет", turn.text.lower())
            self.assertFalse(turn.state["required_services"])
            client.responses.create.assert_not_called()

    def test_photo_video_typo_then_both_keeps_and_not_booth(self):
        state = state_from_request(self.request)
        first, _ = self.turn("Нужен человек для фото и видеосъекмки", state)
        self.assertIn("одного", first.text)
        second, client = self.turn("обе", first.state, openai_reply(text="Могу приступить?"))
        self.assertEqual(second.state["required_services"], ["Фотограф", "Видеограф"])
        results = second.state["last_results"]
        self.assertEqual([r["id"] for r in results[0]["recommendations"]], ["both"])
        self.assertNotIn("приступить", second.text)
        self.assertEqual(second.state["conditions"], state["conditions"])
        self.assertTrue(client.responses.create.called)

    def test_each_requested_service_is_mandatory_and_order_is_stable(self):
        req = self.service.parse_request({**self.request, "required_categories": ["Фотограф", "Видеограф"]})
        self.assertEqual([r.contractor.id for r in recommend(self.catalog, req).recommendations], ["both"])
        self.assertEqual(recommend(self.catalog, req), recommend(list(reversed(self.catalog)), req))

    def test_no_combined_provider_offers_two_without_silently_switching(self):
        service = SearchTools([c for c in self.catalog if c.id != "both"], today=self.today)
        first, _ = self.turn("Фото и видеосъёмка", service=service)
        second, client = self.turn("обе", first.state, service=service)
        self.assertIn("отдельно фотографа и видеографа", second.text)
        self.assertEqual(second.state["service_mode"], "combined")
        self.assertFalse(second.tool_results)
        client.responses.create.assert_not_called()
        conditions = state_from_request(self.request)["conditions"]
        second.state["conditions"] = conditions
        third, _ = self.turn("Да, отдельно", second.state, openai_reply(text="Готово"), service)
        self.assertEqual(third.state["service_mode"], "separate")
        self.assertEqual(len(third.state["last_results"]), 2)
        self.assertEqual({r["request"]["category"] for r in third.state["last_results"]}, {"Фотограф", "Видеограф"})

    def test_budget_change_preserves_date_city_format_and_source_state(self):
        state = state_from_request(self.request)
        original = deepcopy(state)
        turn, _ = self.turn("Бюджет теперь 1.5 млн", state)
        self.assertEqual(turn.state["conditions"]["budget"], 1_500_000)
        for field in ("event_date", "city", "event_format", "language", "duration_hours"):
            self.assertEqual(turn.state["conditions"][field], state["conditions"][field])
        self.assertEqual(state, original)

    def test_today_fixed_time_and_any_are_not_defaults(self):
        state = state_from_request(self.request)
        turn, _ = self.turn("На сегодня, бюджет без разницы, язык любой, длительность неважна", state)
        self.assertEqual(turn.state["conditions"]["event_date"], "2026-10-14")
        for field in ("budget", "language", "duration_hours"):
            self.assertIsNone(turn.state["conditions"][field])
        any_date, _ = self.turn("Дата любая", turn.state)
        result = any_date.state["last_results"][0]
        self.assertIsNone(result["request"]["event_date"])
        self.assertNotIn("Свободен", result["recommendations"][0]["checks"])
        self.assertIn("не проверялась", result["recommendations"][0]["explanation"])

    def test_format_and_occupied_date_exclude_profiles(self):
        req = self.service.parse_request(self.request)
        busy = replace(self.catalog[0], busy_dates=frozenset({"2026-10-14"}))
        wrong = replace(self.catalog[0], id="wrong", event_formats=("корпоратив",))
        result = recommend([busy, wrong], req)
        self.assertFalse(result.recommendations)
        self.assertIn("формат не заявлен в каталоге", dict(result.rejection_counts))
        self.assertNotIn("не берут", result.message)

    def test_model_cannot_replace_services_or_change_locked_conditions(self):
        state = state_from_request({**self.request, "required_categories": ["Фотограф", "Видеограф"]})
        for changes in ({"category": "Фото и видеобудки", "required_categories": []},
                        {"budget": 999999}, {"event_format": "корпоратив"}):
            malicious = {**self.request, "required_categories": ["Фотограф", "Видеограф"], **changes}
            client = Mock()
            client.responses.create.side_effect = [openai_reply([("search_contractors", json.dumps(malicious))]),
                                                  openai_reply(text="Готово")]
            turn = run_turn(self.config, self.catalog, "Найди", state=state, client=client, search_tools=self.service)
            self.assertIn("error", turn.tool_results[-1]["result"])
            self.assertEqual(turn.state["required_services"], ["Фотограф", "Видеограф"])
            self.assertEqual([r["id"] for r in turn.state["last_results"][0]["recommendations"]], ["both"])

    def test_profile_inspection_does_not_mutate_search_or_recommend_booth(self):
        state = state_from_request(self.request)
        state["last_results"] = [self.service.search(self.service.parse_request(self.request))]
        original = deepcopy(state)
        client = Mock()
        client.responses.create.side_effect = [openai_reply([("get_contractor", '{"contractor_id":"booth"}')]),
                                              openai_reply(text="Рекомендую эту будку")]
        turn = run_turn(self.config, self.catalog, "Открой профиль Будка", state=state, client=client, search_tools=self.service)
        self.assertIn("формат не заявлен в каталоге", turn.text)
        self.assertIn("не заявлены все требуемые услуги", turn.text)
        self.assertNotIn("Рекомендую", turn.text)
        for key in ("original_need", "required_services", "conditions", "last_results", "selected_contractor"):
            self.assertEqual(turn.state[key], original[key])
        self.assertEqual(turn.state["viewed_profile"], "booth")

    def test_final_error_empty_json_and_embedded_json_have_text_fallback_both_protocols(self):
        for provider, reply in (("openai", openai_reply), ("nvidia", nvidia_reply)):
            for answer in (None, "", '{"budget":200000}', 'Результат: ```json\n{"city":"Алматы"}\n```'):
                with self.subTest(provider=provider, answer=answer):
                    client = Mock()
                    method = client.responses.create if provider == "openai" else client.chat.completions.create
                    if answer is None:
                        method.side_effect = RuntimeError("SECRET upstream error")
                    else:
                        method.return_value = reply(text=answer)
                    turn = run_turn(AssistantConfig(provider, "test", "fake"), self.catalog, "Найди",
                        state=state_from_request(self.request), client=client, search_tools=self.service)
                    self.assertIn("Подобрано", turn.text)
                    self.assertNotIn("{", turn.text)
                    self.assertNotIn("SECRET", turn.text)
                    self.assertTrue(turn.state["last_results"])
                    payload = method.call_args.kwargs["input" if provider == "openai" else "messages"]
                    self.assertTrue(any(m.get("type") == "function_call_output" or m.get("role") == "tool" for m in payload))

    def test_no_data_empty_reply_still_no_json(self):
        turn, _ = self.turn("Привет", response=openai_reply(text=""))
        self.assertIn("задачу", turn.text)
        self.assertEqual(safe_text("{'city': 'Алматы'}", "Уточните запрос"), "Уточните запрос")

    def test_unknown_fields_block_model_search_without_invented_defaults(self):
        client = Mock()
        client.responses.create.side_effect = [openai_reply([("search_contractors", json.dumps(self.request))]),
                                              openai_reply(text="Уточните город и дату")]
        turn = run_turn(self.config, self.catalog, "Нужен фотограф", client=client, search_tools=self.service)
        self.assertIn("error", turn.tool_results[0]["result"])
        self.assertFalse(turn.state["last_results"])
        self.assertNotIn("budget", turn.state["conditions"])

    def test_selection_is_separate_and_model_cannot_recommend_known_wrong_profile(self):
        state = state_from_request(self.request)
        first, _ = self.turn("Найди", state, openai_reply(text="Рекомендую Будка"))
        self.assertNotIn("Рекомендую", first.text)
        self.assertIn("Подобрано", first.text)
        second, client = self.turn("Выбираю photo", first.state)
        self.assertEqual(second.state["selected_contractor"], "photo")
        self.assertEqual(second.state["conditions"], first.state["conditions"])
        client.responses.create.assert_not_called()

    def test_any_applies_only_to_the_question_actually_asked(self):
        first, _ = self.turn("Нужен фотограф", response=openai_reply(text="Какой бюджет вам подходит?"))
        self.assertEqual(first.state["pending_fields"], ["budget"])
        second, _ = self.turn("без разницы", first.state, openai_reply(text="В каком городе?"))
        self.assertIsNone(second.state["conditions"]["budget"])
        self.assertNotIn("event_date", second.state["conditions"])
        self.assertNotIn("language", second.state["conditions"])
        self.assertEqual(second.state["pending_fields"], ["city"])

    def test_single_service_correction_clears_combined_question(self):
        first, _ = self.turn("Фото и видео", state_from_request(self.request))
        second, _ = self.turn("Только фотограф", first.state)
        self.assertFalse(second.state["awaiting_service_mode"])
        self.assertEqual(second.state["required_services"], ["Фотограф"])
        self.assertTrue(second.state["last_results"])

    def test_original_need_survives_clarification_and_unknown_service_with_city(self):
        first, _ = self.turn("Нужен тестировщик в Алматы")
        second, _ = self.turn("Нужен фотограф", first.state)
        self.assertEqual(second.state["original_need"], "Нужен тестировщик в Алматы")
        self.assertEqual(second.state["current_need"], "Нужен фотограф")
        self.assertEqual(second.state["conditions"]["city"], "Алматы")
        third, _ = self.turn("Нужен программист в Алматы", second.state)
        self.assertFalse(third.state["required_services"])
        self.assertIn("задачу", third.text)

    def test_negative_booth_mention_is_not_a_requested_service(self):
        first, _ = self.turn("Нужен фотограф, не фотобудка", state_from_request(self.request))
        self.assertEqual(first.state["required_services"], ["Фотограф"])

    def test_empty_results_ask_to_refine_without_changing_conditions(self):
        state = state_from_request({**self.request, "budget": 1})
        turn, _ = self.turn("Найди", state, openai_reply(text="Вот отличные варианты"))
        self.assertIn("Какое условие готовы изменить?", turn.text)
        self.assertEqual(turn.state["conditions"], state["conditions"])
        self.assertFalse(turn.state["last_results"][0]["recommendations"])

    def test_profile_facts_are_individual_and_tied_to_catalog_fields(self):
        first = replace(self.catalog[0], description="Опыт ведения свадеб 13 лет.", languages=("русский", "казахский"))
        second = replace(first, id="second", name="Другой", description="Тонкий юмор и сдержанные манеры.", languages=("русский",))
        service = SearchTools([first, second], today=self.today)
        result = service.search(service.parse_request({**self.request, "preferences": "квантовый реактор на Марсе"}))
        cards = result["recommendations"]
        self.assertNotEqual(cards[0]["profile_facts"], cards[1]["profile_facts"])
        for card in cards:
            original = first if card["id"] == first.id else second
            facts = {f["source"]: f["value"] for f in card["profile_facts"]}
            self.assertIn(facts["description"], original.description)
            self.assertEqual(facts["languages"], ", ".join(original.languages))
            self.assertFalse(card["preference_evidence"])
            self.assertIn("пожелание не подтверждено описанием", card["explanation"])

    def test_photographer_month_city_typo_and_freeform_any_preserve_need(self):
        busy = replace(self.catalog[0], busy_dates=frozenset({"2026-12-01", "2026-12-02"}))
        service = SearchTools([busy, self.catalog[1]], today=self.today)
        answer = ("Понял, вам нужен фотограф в Алматы в декабре на ближайшую свободную дату.\n"
                  "Какой бюджет вам подходит? На каком языке нужен ведущий или общение с подрядчиком? "
                  "На сколько часов требуется фотограф? Формат мероприятия укажите, пожалуйста, если он есть.")
        first, _ = self.turn("нужен фотограф в декабре в алмате на ближайщую свободную дату",
                            response=openai_reply(text=answer), service=service)
        self.assertEqual(first.state["conditions"]["city"], "Алматы")
        self.assertEqual(first.state["required_services"], ["Фотограф"])
        self.assertNotIn("ведущ", first.text.lower())
        self.assertEqual(set(first.state["pending_fields"]), {"budget", "language", "duration_hours", "event_format"})
        second, _ = self.turn("не важно", first.state, openai_reply(text="В каком городе?"), service)
        result = second.state["last_results"][0]
        self.assertEqual(result["request"]["city"], "Алматы")
        self.assertEqual(result["request"]["event_date"], "2026-12-03")
        self.assertEqual([r["id"] for r in result["recommendations"]], ["photo"])
        self.assertIn("03.12.2026", second.text)
        self.assertNotIn("В каком городе", second.text)
        self.assertEqual(second.state["date_window"], {"start": "2026-12-01", "end": "2026-12-31", "mode": "earliest"})
        for field in ("budget", "language", "duration_hours", "event_format"):
            self.assertIsNone(result["request"][field])
        third, _ = self.turn("Найди", second.state, openai_reply(text="Нашёл для вас ведущего."), service)
        self.assertNotIn("ведущ", third.text.lower())
        self.assertIn("Подобрано", third.text)

    def test_recap_and_candidate_word_do_not_turn_into_date_questions(self):
        question = ("Понял: город Алматы, ближайшая дата в декабре. "
                    "Какой бюджет кандидата? Какой язык общения нужен? На сколько часов?")
        self.assertEqual(question_fields(question), ["budget", "language", "duration_hours"])
        state = state_from_request(self.request)
        for field in ("budget", "language", "duration_hours"):
            state["conditions"].pop(field)
        first, _ = self.turn("Нужен фотограф", state, openai_reply(text=question))
        for reply in ("не важно", "Да, мне всё без разницы", "всё остальное не важно", "любой подойдет"):
            with self.subTest(reply=reply):
                second, _ = self.turn(reply, first.state)
                for field in ("city", "event_date", "event_format"):
                    self.assertEqual(second.state["conditions"][field], state["conditions"][field])
                for field in ("budget", "language", "duration_hours"):
                    self.assertIsNone(second.state["conditions"][field])

    def test_month_budget_is_not_mistaken_for_year(self):
        window = extract_date_window("Нужен фотограф в декабре, бюджет 2000 тенге", self.today)
        self.assertEqual(window["start"], "2026-12-01")
        explicit = extract_date_window("в декабре 2027 года, бюджет 2000 тенге", self.today)
        self.assertEqual(explicit["start"], "2027-12-01")
        before = extract_date_window("в 2027 году в декабре", self.today)
        self.assertEqual(before["start"], "2027-12-01")

    def test_any_day_in_month_keeps_month_and_searches_nearest_available(self):
        state = state_from_request(self.request)
        first, _ = self.turn("В декабре", state, openai_reply(text="На какой день в указанном месяце?"))
        self.assertEqual(first.state["pending_fields"], ["event_date"])
        second, _ = self.turn("без разницы", first.state)
        self.assertEqual(second.state["conditions"]["event_date"], "2026-12-01")
        self.assertEqual(second.state["date_window"]["end"], "2026-12-31")

    def test_month_window_rechecks_budget_without_losing_other_conditions(self):
        cheap = replace(self.catalog[0], busy_dates=frozenset({"2026-12-01", "2026-12-02"}))
        dear = replace(self.catalog[0], id="dear", price=300_000)
        service = SearchTools([cheap, dear], today=self.today)
        first, _ = self.turn("В декабре на ближайшую свободную дату", state_from_request(self.request), service=service)
        self.assertEqual(first.state["conditions"]["event_date"], "2026-12-03")
        second, _ = self.turn("Бюджет до 400 тысяч", first.state, service=service)
        self.assertEqual(second.state["conditions"]["event_date"], "2026-12-01")
        self.assertEqual(second.state["date_window"], first.state["date_window"])
        self.assertEqual([r["id"] for r in second.state["last_results"][0]["recommendations"]], ["dear"])
        for field in ("city", "event_format", "language", "duration_hours"):
            self.assertEqual(second.state["conditions"][field], first.state["conditions"][field])
        third, _ = self.turn("На 10 декабря", second.state, service=service)
        self.assertIsNone(third.state["date_window"])
        self.assertEqual(third.state["conditions"]["event_date"], "2026-12-10")

    def test_earliest_date_applies_all_constraints_and_service_and(self):
        combined = replace(self.catalog[2], busy_dates=frozenset({"2026-12-01"}))
        wrong_format = replace(combined, id="wrong-format", event_formats=("корпоратив",), busy_dates=frozenset())
        wrong_city = replace(combined, id="wrong-city", city="Астана", busy_dates=frozenset())
        wrong_language = replace(combined, id="wrong-language", languages=("казахский",), busy_dates=frozenset())
        short = replace(combined, id="short", max_hours=2, busy_dates=frozenset())
        service = SearchTools([self.catalog[0], combined, wrong_format, wrong_city, wrong_language, short], today=self.today)
        state = state_from_request({**self.request, "required_categories": ["Фотограф", "Видеограф"]})
        turn, _ = self.turn("В декабре на ближайшую свободную дату", state, service=service)
        result = turn.state["last_results"][0]
        self.assertEqual(result["request"]["event_date"], "2026-12-02")
        self.assertEqual([r["id"] for r in result["recommendations"]], ["both"])

    def test_no_free_date_in_month_does_not_search_outside_window(self):
        busy = replace(self.catalog[0], busy_dates=frozenset(f"2026-12-{day:02d}" for day in range(1, 32)))
        service = SearchTools([busy], today=self.today)
        turn, client = self.turn("В декабре на ближайшую свободную дату", state_from_request(self.request), service=service)
        self.assertFalse(turn.state["last_results"])
        self.assertNotIn("event_date", turn.state["conditions"])
        self.assertEqual(turn.state["date_window"]["end"], "2026-12-31")
        self.assertIn("подходящих подрядчиков нет", turn.text)
        client.responses.create.assert_not_called()

    def test_outside_catalog_month_is_reported_before_remaining_questions(self):
        for text in ("Фотограф в Алматы в декабре 2027 на ближайшую свободную дату",
                     "Фотограф в Алматы в сентябре на ближайшую свободную дату"):
            turn, client = self.turn(text)
            self.assertIn("31.12.2026", turn.text)
            self.assertEqual(turn.text.count("?"), 1)
            self.assertNotIn("бюджет", turn.text)
            self.assertFalse(turn.state["last_results"])
            client.responses.create.assert_not_called()
        self.assertEqual(extract_date_window("в декабре", date(2026, 12, 23))["start"], "2026-12-01")
        self.assertEqual(extract_date_window("в декабре", date(2027, 1, 2))["start"], "2027-12-01")

    def test_current_month_earliest_date_never_uses_past_days(self):
        service = SearchTools([self.catalog[0]], today=date(2026, 12, 23))
        turn, _ = self.turn("В декабре на ближайшую свободную дату", state_from_request(self.request), service=service)
        self.assertEqual(turn.state["last_results"][0]["request"]["event_date"], "2026-12-23")

    def test_changing_only_month_preserves_earliest_strategy(self):
        first, _ = self.turn("В декабре на ближайшую свободную дату", state_from_request(self.request))
        second, _ = self.turn("В ноябре", first.state)
        self.assertEqual(second.state["date_window"], {"start": "2026-11-01", "end": "2026-11-30", "mode": "earliest"})
        self.assertEqual(second.state["last_results"][0]["request"]["event_date"], "2026-11-01")
        for field in ("city", "event_format", "budget", "language", "duration_hours"):
            self.assertEqual(second.state["conditions"][field], first.state["conditions"][field])

    def test_legacy_history_recovers_only_questions_actually_asked(self):
        client = Mock()
        client.responses.create.return_value = openai_reply(text="На какую дату?")
        history = [{"role": "user", "content": "Нужен фотограф в Алмате"},
                   {"role": "assistant", "content": "Какой бюджет вам подходит?"}]
        turn = run_turn(self.config, self.catalog, "мне не важно", history, client=client, search_tools=self.service)
        self.assertEqual(turn.state["conditions"], {"city": "Алматы", "budget": None})

    def test_bare_number_answers_only_the_pending_numeric_question(self):
        for field, reply, expected in (("budget", "250 000", 250_000), ("duration_hours", "4", 4)):
            state = state_from_request(self.request)
            state["conditions"].pop(field)
            state["pending_fields"] = [field]
            turn, _ = self.turn(reply, state)
            self.assertEqual(turn.state["conditions"][field], expected)
            for other in ("city", "event_date", "event_format"):
                self.assertEqual(turn.state["conditions"][other], state["conditions"][other])
