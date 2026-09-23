import unittest
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
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
