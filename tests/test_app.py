import json
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from assistant import AssistantError, AssistantTurn, SearchTools
from product import search_id
from recommender import load_contractors


ROOT = Path(__file__).parents[1]


class AppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
        app.secrets["OPENAI_API_KEY"] = "test-openai"
        app.secrets["NVIDIA_API_KEY"] = "test-nvidia"
        return app

    def test_single_result_panel_syncs_chat_and_manual_controls(self):
        result = SearchTools(load_contractors(ROOT / "data/contractors.csv")).dispatch(
            "search_contractors", json.dumps(dict(city="Алматы", event_date="2026-09-23",
                                                  event_format="свадьба", category="Ведущий")))
        turn = AssistantTurn("Найдено 4 ведущих.", [], [{"name": "search_contractors", "result": result}])
        app = self.app()
        with patch("assistant_ui.run_turn", return_value=turn) as run:
            app.run()
            app.chat_input(key="ai_message").set_value("Нужен ведущий на свадьбу сегодня в Алматы").run()
            self.assertFalse(app.exception)
            names = [item.value for item in app.subheader]
            for name in ("Мицури Канроджи", "Эмилия", "Сон Гоку", "Софи Хаттер"):
                self.assertEqual(names.count(name), 1)
            self.assertEqual([h.value for h in app.header].count("Ваше мероприятие"), 1)
            self.assertTrue(app.checkbox(key="budget_unlimited").value)
            self.assertIsNone(app.session_state["search_request"].budget)
            self.assertTrue(any(e.label == "Все подходящие — ещё 1" for e in app.expander))
            app.button(key="manual_submit").click().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["search_request"].budget)
            self.assertEqual(run.call_count, 1)

    def test_chat_without_credentials_preserves_manual_search(self):
        with patch("assistant_ui.AssistantConfig.from_settings", side_effect=AssistantError("Добавьте API-ключ")):
            app = self.app().run()
            self.assertTrue(app.chat_input[0].disabled)
            app.button(key="manual_submit").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)

    def test_date_suggestion_requires_click_and_preserves_conditions(self):
        app = self.app().run()
        app.date_input(key="event_date").set_value(date(2026, 10, 3))
        app.text_area(key="preferences").set_value("квантовый реактор на Марсе")
        app.button(key="manual_submit").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 3))
        self.assertIn("04.10.2026", app.button(key="suggestion_0").label)
        app.button(key="suggestion_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 4))
        self.assertEqual(app.number_input(key="budget").value, 1_500_000)
        self.assertEqual(app.text_area(key="preferences").value, "квантовый реактор на Марсе")
        self.assertEqual(app.session_state["active_result"]["matched_count"], 2)
        self.assertTrue(any("пожелание не подтверждено" in info.value for info in app.info))

    def test_comparison_shortlist_and_history(self):
        app = self.app().run()
        app.button(key="manual_submit").click().run()
        payload = app.session_state["active_result"]
        key = search_id(payload["request"])
        first = payload["recommendations"][0]
        app.button(key=f"compare_button_{key}").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.dataframe[0].value), 3)
        app.button(key=f"save_{key}_{first['id']}").click().run()
        self.assertIn(first["id"], app.session_state["shortlist"])
        self.assertEqual(len(app.session_state["search_history"]), 1)
        app.button(key="history_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["active_result"]["request"], payload["request"])
        self.assertIn(first["id"], app.session_state["shortlist"])

    def test_multiple_languages_apply_all(self):
        app = self.app().run()
        app.multiselect(key="languages_selected").set_value(["русский", "казахский"])
        app.checkbox(key="budget_unlimited").check()
        app.button(key="manual_submit").click().run()
        self.assertFalse(app.exception)
        for card in app.session_state["active_result"]["recommendations"]:
            self.assertTrue({"русский", "казахский"} <= set(card["languages"]))

    def test_chat_api_failure_does_not_commit_history(self):
        app = self.app()
        with patch("assistant_ui.run_turn", side_effect=AssistantError("API недоступен")):
            app.run()
            app.chat_input(key="ai_message").set_value("Найди").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(all(not s["history"] for s in app.session_state["ai_sessions"].values()))

    def test_combined_manual_services_and_unlimited_date_round_trip(self):
        app = self.app().run()
        app.selectbox(key="category").select("Фотограф")
        app.multiselect(key="additional_categories").set_value(["Видеограф"])
        app.checkbox(key="date_unlimited").check()
        app.button(key="manual_submit").click().run()
        self.assertFalse(app.exception)
        request = app.session_state["active_result"]["request"]
        self.assertEqual(request["required_categories"], ["Фотограф", "Видеограф"])
        self.assertIsNone(request["event_date"])
        self.assertFalse(any("Как сужался выбор" in c.value for c in app.caption))
        for session in app.session_state["ai_sessions"].values():
            self.assertEqual(session["state"]["required_services"], ["Фотограф", "Видеограф"])
        app.button(key="history_0").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key="date_unlimited").value)
        self.assertEqual(app.multiselect(key="additional_categories").value, ["Видеограф"])

    def test_unknown_service_has_no_stale_result_or_json_in_chat(self):
        app = self.app().run()
        app.button(key="manual_submit").click().run()
        app.chat_input(key="ai_message").set_value("Нужен тестировщик").run()
        self.assertFalse(app.exception)
        self.assertFalse(any(h.value == "Ваше мероприятие" for h in app.header))
        self.assertTrue(any("Какую задачу" in m.value for m in app.markdown))
