import unittest
from datetime import date
from pathlib import Path

from recommender import SearchRequest, load_contractors, recommend


DATA_PATH = Path(__file__).parents[1] / "data" / "contractors.csv"


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contractors = load_contractors(DATA_PATH)

    def test_dataset_has_expected_size_and_unique_ids(self):
        self.assertEqual(len(self.contractors), 66)
        self.assertEqual(len({item.id for item in self.contractors}), 66)

    def test_dense_category_returns_at_most_three_deterministically(self):
        request = SearchRequest(
            city="Алматы",
            event_date=date(2026, 10, 14),
            event_format="свадьба",
            category="Ведущий",
            budget=1_500_000,
            duration_hours=6,
            language="русский",
        )
        first = recommend(self.contractors, request)
        second = recommend(self.contractors, request)
        self.assertEqual(first.status, "matched")
        self.assertLessEqual(len(first.recommendations), 3)
        self.assertEqual(first.recommendations, second.recommendations)

    def test_different_dates_can_change_results(self):
        base = dict(
            city="Алматы",
            event_format="свадьба",
            category="Ведущий",
            budget=1_500_000,
            duration_hours=6,
            language="русский",
        )
        available = recommend(self.contractors, SearchRequest(event_date=date(2026, 10, 14), **base))
        busy = recommend(self.contractors, SearchRequest(event_date=date(2026, 10, 3), **base))
        self.assertEqual(available.status, "matched")
        self.assertEqual(busy.status, "conditions_not_met")

    def test_category_absent_in_city_is_explicit(self):
        request = SearchRequest(
            city="Зарубежье",
            event_date=date(2026, 10, 14),
            event_format="свадьба",
            category="Флорист",
            budget=1_000_000,
        )
        result = recommend(self.contractors, request)
        self.assertEqual(result.status, "category_absent")

    def test_preferences_are_scored_and_explained(self):
        request = SearchRequest(
            city="Алматы",
            event_date=date(2026, 10, 14),
            event_format="свадьба",
            category="Ведущий",
            budget=1_500_000,
            language="русский",
            preferences="интеллигентный ведущий для деловой аудитории",
        )
        result = recommend(self.contractors, request)
        self.assertEqual(result.status, "matched")
        self.assertTrue(result.recommendations)
        self.assertTrue(
            all(
                any(name == "Текстовая релевантность" for name, _ in item.factors)
                for item in result.recommendations
            )
        )
        self.assertIn("пожелани", result.recommendations[0].explanation)


if __name__ == "__main__":
    unittest.main()
