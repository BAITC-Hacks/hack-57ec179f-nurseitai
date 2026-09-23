import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from assistant import AssistantError, AssistantTurn, SearchTools
from recommender import load_contractors


class AppTests(unittest.TestCase):
    def test_chat_without_credentials_preserves_regular_search(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "NVIDIA_API_KEY": ""}):
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)

    def test_chat_results_require_click_to_change_form_and_provider_isolation(self):
        path = Path(__file__).parents[1]
        args = dict(city="Алматы", event_date="2026-10-03", event_format="свадьба",
                    category="Ведущий", budget=1_500_000, duration_hours=6,
                    language="русский", preferences="")
        import json
        result = SearchTools(load_contractors(path / "data/contractors.csv")).dispatch("search_contractors", json.dumps(args))
        turn = AssistantTurn("Можно перенести дату.", [{"role": "user", "content": "Найди"}],
                             [{"name": "search_contractors", "result": result}])
        app = AppTest.from_file(str(path / "app.py"))
        app.secrets["OPENAI_API_KEY"] = "test-openai"
        app.secrets["NVIDIA_API_KEY"] = "test-nvidia"
        with patch("assistant_ui.run_turn", return_value=turn) as run:
            app.run()
            app.chat_input(key="ai_message").set_value("Найди ведущего").run()
            self.assertFalse(app.exception)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 14))
            button = next(b for b in app.button if b.label == "Применить дату 2026-10-04")
            button.click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 4))
            self.assertEqual(run.call_count, 1)
            app.selectbox(key="ai_provider").set_value("nvidia").run()
            self.assertEqual(len(app.chat_message), 0)

    def test_chat_api_failure_does_not_commit_history(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
        app.secrets["OPENAI_API_KEY"] = "test-openai"
        with patch("assistant_ui.run_turn", side_effect=AssistantError("API недоступен")):
            app.run()
            app.chat_input(key="ai_message").set_value("Найди").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            sessions = app.session_state["ai_sessions"]
            self.assertTrue(all(not session["history"] for session in sessions.values()))

    def test_date_suggestion_requires_click_and_preserves_conditions(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.selectbox(key="city").set_value("Алматы")
        app.date_input(key="event_date").set_value(date(2026, 10, 3))
        app.text_area(key="preferences").set_value("квантовый реактор на Марсе")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 3))
        self.assertTrue(any("04.10.2026" in info.value for info in app.info))
        app.button(key="suggestion_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 4))
        self.assertEqual(app.number_input(key="budget").value, 1_500_000)
        self.assertEqual(app.text_area(key="preferences").value, "квантовый реактор на Марсе")
        self.assertEqual(len(app.subheader), 2)
        self.assertTrue(all("пожелание не подтверждено" in info.value for info in app.info))

    def test_budget_suggestion_applies_exact_price(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.selectbox(key="city").set_value("Алматы")
        app.number_input(key="budget").set_value(100_000)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        button = next(b for b in app.button if b.label == "Применить новый бюджет")
        button.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key="event_date").value, date(2026, 10, 14))
        self.assertGreater(app.number_input(key="budget").value, 100_000)
        self.assertTrue(app.success)
