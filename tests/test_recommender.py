import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from recommender import SearchRequest, load_contractors, recommend


DATA_PATH = Path(__file__).parents[1] / "data" / "contractors.csv"


class RecommenderTests(unittest.TestCase):
    def base_request(self, **changes):
        return replace(SearchRequest("Алматы", date(2026, 10, 14), "свадьба",
                                     "Ведущий", 1_500_000, 6, "русский"), **changes)

    def test_impossible_preference_has_no_score_or_false_claim(self):
        result = recommend(self.contractors, self.base_request(preferences="квантовый реактор на Марсе"))
        for item in result.recommendations:
            self.assertEqual(dict(item.factors)["Текстовая релевантность"], 0)
            self.assertIn("Обязательные условия подходят, но пожелание не подтверждено описанием", item.explanation)

    def test_profile_facts_distinguish_equal_conditions_without_names(self):
        pool = [item for item in self.contractors if item.name in ("Эмилия", "Кики")]
        self.assertEqual(len(pool), 2)
        result = recommend(pool, self.base_request())
        explanations = []
        for rec in result.recommendations:
            text = rec.explanation
            for item in pool:
                text = text.replace(item.name, "")
            explanations.append(text)
            self.assertIn("В профиле указано:", text)
        self.assertEqual(len(set(explanations)), 2)

    def test_evidence_is_exact_quote_and_partial_wish_is_not_confirmed(self):
        candidate = replace(self.contractors[0], categories=("Ведущий",),
                            price=100_000, busy_dates=frozenset(),
                            description="Тонкий юмор и сдержанные манеры.")
        positive = recommend([candidate], self.base_request(preferences="тонкий юмор"))
        self.assertIn("фрагмент по пожеланию: «Тонкий юмор и сдержанные манеры.»", positive.recommendations[0].explanation)
        negative = recommend([candidate], self.base_request(preferences="тонкий юмор и квантовый реактор"))
        self.assertIn("пожелание не подтверждено", negative.recommendations[0].explanation)

    def test_next_date_is_verified_without_changing_original_request(self):
        request = self.base_request(event_date=date(2026, 10, 3))
        result = recommend(self.contractors, request)
        suggestion = next(s for s in result.suggestions if s.request.event_date != request.event_date)
        self.assertEqual(suggestion.request, replace(request, event_date=date(2026, 10, 4)))
        self.assertEqual(suggestion.count, 2)
        self.assertEqual(len(recommend(self.contractors, suggestion.request).recommendations), 2)
        self.assertEqual(request.event_date, date(2026, 10, 3))

    def test_minimum_budget_ignores_cheaper_ineligible_contractors(self):
        base = replace(self.contractors[0], categories=("Ведущий",), busy_dates=frozenset())
        pool = [replace(base, id="a", price=200_000, languages=("английский",)),
                replace(base, id="b", price=350_000), replace(base, id="c", price=450_000)]
        request = self.base_request(budget=100_000)
        result = recommend(pool, request)
        self.assertEqual(len(result.suggestions), 1)
        suggestion = result.suggestions[0]
        self.assertEqual(suggestion.request, replace(request, budget=350_000))
        self.assertEqual(suggestion.count, 1)

    def test_no_alternative_beyond_calendar_or_for_wrong_language(self):
        base = replace(self.contractors[0], categories=("Ведущий",), price=100_000,
                       busy_dates=frozenset({"2026-12-31"}))
        self.assertFalse(recommend([base], self.base_request(event_date=date(2026, 12, 31))).suggestions)
        alternatives = recommend([replace(base, languages=("английский",))], self.base_request()).suggestions
        self.assertEqual([s.kind for s in alternatives], ["language"])
        self.assertIsNone(alternatives[0].request.language)

    def test_sparse_category_summary_is_visible_in_message(self):
        result = recommend(self.contractors, self.base_request(category="Флорист"))
        self.assertEqual(len(result.recommendations), 1)
        self.assertIn("всего 2", result.message)
        self.assertIn("заняты: 1", result.message)

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
