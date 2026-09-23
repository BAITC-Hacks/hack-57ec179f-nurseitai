import json
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from assistant import SearchTools
from preferences import evidence
from product import comparison_rows, create_demo_request, record_search, toggle_shortlist
from recommender import Contractor, SearchRequest, recommend, rejection_reasons
from semantic import EmbeddingRanker


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.request = SearchRequest("Алматы", date(2026, 10, 14), "свадьба", "Ведущий", 1_500_000, 6, "русский")
        self.base = Contractor("a", "Первый", ("Ведущий",), "Алматы", False, 900_000,
                               ("свадьба",), ("русский",), 8, frozenset(),
                               "Современная подача. Интеллигентный юмор.")

    def test_pipeline_counts_sequentially_and_near_reasons_overlap(self):
        pool = [self.base, replace(self.base, id="b", city="Астана"),
                replace(self.base, id="c", categories=("Флорист",)),
                replace(self.base, id="d", event_formats=("той",)),
                replace(self.base, id="e", busy_dates=frozenset({"2026-10-14"}), price=2_000_000),
                replace(self.base, id="f", price=2_000_000)]
        result = recommend(pool, self.request)
        self.assertEqual([count for _, count in result.pipeline], [6, 5, 4, 3, 2, 1, 1, 1, 1])
        self.assertEqual(dict(result.rejection_counts)["дороже бюджета"], 2)
        near = {n.contractor.id: n.reasons for n in result.near_matches}
        self.assertEqual(set(near["e"]), {"заняты", "дороже бюджета"})
        self.assertNotIn("a", near)

    def test_morphology_across_fragments_and_negation(self):
        quote = evidence("современный ведущий с интеллигентным юмором", self.base.description)
        self.assertEqual(quote, "Современная подача. … Интеллигентный юмор.")
        self.assertFalse(evidence("без конкурсов", "Проводит конкурсы."))
        self.assertFalse(evidence("тонкий юмор", "Не использует тонкий юмор."))
        self.assertFalse(evidence("квантовый реактор на Марсе", self.base.description))

    def test_semantic_cannot_bypass_hard_filters_or_prove_preferences(self):
        pool = [self.base, replace(self.base, id="b", price=2_000_000),
                replace(self.base, id="c", description="Другой стиль.")]
        ranker = Mock(return_value=[0.1, 0.9])
        request = replace(self.request, preferences="квантовый реактор на Марсе")
        result = recommend(pool, request, semantic_ranker=ranker)
        self.assertEqual(ranker.call_args.args[1], [self.base.description, "Другой стиль."])
        self.assertEqual([r.contractor.id for r in result.recommendations], ["c", "a"])
        self.assertTrue(all(r.match_percent < 100 and not r.preference_evidence for r in result.recommendations))
        self.assertEqual(result.ranking_mode, "embeddings")
        self.assertEqual(recommend([self.base], self.request).recommendations[0].match_percent, 100)

    def test_embedding_failure_falls_back_and_no_preferences_make_no_call(self):
        ranker = Mock(side_effect=RuntimeError("secret must not leak"))
        result = recommend([self.base], replace(self.request, preferences="юмор"), semantic_ranker=ranker)
        self.assertEqual(result.ranking_mode, "fallback")
        self.assertNotIn("secret", result.ranking_notice)
        ranker.reset_mock()
        recommend([self.base], self.request, semantic_ranker=ranker)
        ranker.assert_not_called()

    def test_alternative_changes_one_condition_and_adds_real_matches(self):
        pool = [self.base, replace(self.base, id="b", languages=("казахский",)),
                replace(self.base, id="c", max_hours=4)]
        result = recommend(pool, self.request)
        self.assertEqual({s.kind for s in result.suggestions}, {"language", "duration"})
        for suggestion in result.suggestions:
            changes = [k for k in vars(self.request) if getattr(suggestion.request, k) != getattr(self.request, k)]
            self.assertEqual(len(changes), 1)
            rerun = recommend(pool, suggestion.request, limit=None, suggest_alternatives=False)
            self.assertEqual(rerun.matched_count, suggestion.count)
            self.assertEqual(suggestion.added_count, 1)
        hours = next(s.request.duration_hours for s in result.suggestions if s.kind == "duration")
        self.assertEqual(hours, 4)

    def test_category_alternative_only_from_curated_mapping(self):
        pool = [replace(self.base, categories=("Артист",))]
        self.assertFalse(recommend(pool, self.request).suggestions)
        result = recommend(pool, self.request, related_categories={"Ведущий": ["Артист"]})
        self.assertEqual([s.kind for s in result.suggestions], ["category"])
        self.assertEqual(result.suggestions[0].count, 1)

    def test_two_languages_are_and_not_or(self):
        request = replace(self.request, language="русский/казахский")
        self.assertIn("не подходит язык", rejection_reasons(self.base, request))
        self.assertFalse(rejection_reasons(replace(self.base, languages=("русский", "казахский")), request))

    def test_explain_specific_candidate_and_state_operations(self):
        service = SearchTools([self.base, replace(self.base, id="b", price=2_000_000)])
        args = {**vars(self.request), "event_date": "2026-10-14"}
        rejected = service.dispatch("explain_contractor_match", json.dumps({"contractor_id": "b", "request": args}))
        self.assertFalse(rejected["eligible"])
        self.assertEqual(rejected["reasons"], ["дороже бюджета"])
        payload = service.search(self.request)
        state = {}
        record_search(state, payload)
        record_search(state, payload)
        self.assertEqual(len(state["search_history"]), 1)
        rec = payload["recommendations"][0]
        toggle_shortlist(state, rec, args)
        self.assertEqual(len(state["shortlist"]), 1)
        first = create_demo_request(state, rec, args)
        self.assertEqual(first, create_demo_request(state, rec, args))
        self.assertEqual(len(state["demo_requests"]), 1)
        self.assertIn("Не отправлен", first["status"])
        rows = comparison_rows([rec], args)
        self.assertEqual(rows[0]["ID"], "a")
        self.assertEqual(rows[0]["Свободен по календарю"], "2026-10-14")
        toggle_shortlist(state, rec, args)
        self.assertFalse(state["shortlist"])

    def test_embedding_cache_invalidates_model_or_changed_text(self):
        client = Mock()
        def response(**kwargs):
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, float(i % 2)])
                                         for i, _ in enumerate(kwargs["input"])])
        client.embeddings.create.side_effect = response
        cache = {}
        ranker = EmbeddingRanker(client, cache=cache)
        ranker("юмор", ["Описание А", "Описание Б"])
        ranker("юмор", ["Описание А", "Описание Б"])
        self.assertEqual(client.embeddings.create.call_count, 1)
        ranker("юмор", ["Описание А изменено"])
        self.assertEqual(client.embeddings.create.call_args.kwargs["input"], ["Описание А изменено"])
        EmbeddingRanker(client, model="another-model", cache=cache)("юмор", ["Описание А"])
        self.assertEqual(client.embeddings.create.call_args.kwargs["input"], ["юмор", "Описание А"])
