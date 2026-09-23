from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from preferences import evidence, fragments, terms


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


@dataclass(frozen=True)
class Contractor:
    id: str
    name: str
    categories: tuple[str, ...]
    city: str
    synthetic: bool
    price: int
    event_formats: tuple[str, ...]
    languages: tuple[str, ...]
    max_hours: int | None
    busy_dates: frozenset[str]
    description: str


@dataclass(frozen=True)
class SearchRequest:
    city: str
    event_date: date
    event_format: str
    category: str
    budget: int | None = None
    duration_hours: int | None = None
    language: str | None = None
    preferences: str = ""


@dataclass(frozen=True)
class Recommendation:
    contractor: Contractor
    score: float
    explanation: str
    factors: tuple[tuple[str, float], ...]
    match_percent: int = 100
    checks: tuple[str, ...] = ()
    preference_evidence: str = ""
    profile_quote: str = ""
    ranking_reason: str = ""


@dataclass(frozen=True)
class Suggestion:
    request: SearchRequest
    count: int
    message: str
    kind: str = "date"
    added_count: int = 0


@dataclass(frozen=True)
class NearMatch:
    contractor: Contractor
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    status: str
    message: str
    recommendations: tuple[Recommendation, ...] = ()
    rejection_counts: tuple[tuple[str, int], ...] = ()
    suggestions: tuple[Suggestion, ...] = ()
    pipeline: tuple[tuple[str, int], ...] = ()
    near_matches: tuple[NearMatch, ...] = ()
    matched_count: int = 0
    ranking_mode: str = "local"
    ranking_notice: str = ""


CALENDAR_END = date(2026, 12, 31)


def _fragments(description: str) -> list[str]:
    return fragments(description)


def _preference_evidence(preferences: str, description: str) -> str:
    return evidence(preferences, description)


def load_contractors(path: str | Path) -> list[Contractor]:
    contractors: list[Contractor] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            contractors.append(
                Contractor(
                    id=row["id"],
                    name=row["anon_name"],
                    categories=_split(row["categories"]),
                    city=row["city"].strip(),
                    synthetic=row["synthetic"].casefold() == "true",
                    price=int(row["price_from_kzt"]),
                    event_formats=_split(row["event_formats"]),
                    languages=_split(row["languages"]),
                    max_hours=int(row["max_hours"]) if row["max_hours"].strip() else None,
                    busy_dates=frozenset(_split(row["busy_dates"])),
                    description=row["description"].strip(),
                )
            )
    return contractors


def _word_ngrams(value: str) -> list[str]:
    return sorted(terms(value))


def _text_relevance(query: str, documents: list[str]) -> list[float]:
    """Return deterministic TF-IDF cosine similarity for each document."""
    if not query.strip() or not documents:
        return [0.0] * len(documents)

    tokenized = [_word_ngrams(query), *(_word_ngrams(document) for document in documents)]
    document_count = len(tokenized)
    frequencies = Counter(term for terms in tokenized for term in set(terms))

    def vector(terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        total = max(1, sum(counts.values()))
        return {
            term: count / total * (math.log((1 + document_count) / (1 + frequencies[term])) + 1)
            for term, count in counts.items()
        }

    vectors = [vector(terms) for terms in tokenized]
    query_vector = vectors[0]
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    if query_norm == 0:
        return [0.0] * len(documents)

    scores: list[float] = []
    for candidate in vectors[1:]:
        candidate_norm = math.sqrt(sum(value * value for value in candidate.values()))
        dot = sum(value * candidate.get(term, 0.0) for term, value in query_vector.items())
        scores.append(dot / (query_norm * candidate_norm) if candidate_norm else 0.0)
    return scores


def requested_languages(request: SearchRequest) -> tuple[str, ...]:
    # Slash, comma and «и» mean ALL selected languages, never silently OR.
    return tuple(p.strip() for p in re.split(r"[/,|]|\s+и\s+", request.language or "") if p.strip())


def constraint_checks(item: Contractor, request: SearchRequest):
    checks = [("Город", item.city == request.city),
              ("Категория", request.category in item.categories),
              ("Формат", request.event_format in item.event_formats),
              ("Свободен", request.event_date.isoformat() not in item.busy_dates)]
    if request.budget is not None:
        checks.append(("Бюджет подходит", item.price <= request.budget))
    if request.language:
        checks.append(("Язык: " + request.language, all(v in item.languages for v in requested_languages(request))))
    if request.duration_hours is not None:
        checks.append((f"Длительность: {request.duration_hours} ч", item.max_hours is None or item.max_hours >= request.duration_hours))
    return checks


def rejection_reasons(item: Contractor, request: SearchRequest) -> tuple[str, ...]:
    reasons = []
    if request.event_date.isoformat() in item.busy_dates:
        reasons.append("заняты")
    if request.budget is not None and item.price > request.budget:
        reasons.append("дороже бюджета")
    if request.event_format not in item.event_formats:
        reasons.append("не берут формат")
    if request.language and not all(v in item.languages for v in requested_languages(request)):
        reasons.append("не подходит язык")
    if request.duration_hours is not None and item.max_hours is not None and item.max_hours < request.duration_hours:
        reasons.append("не подходит длительность")
    return tuple(reasons)


def profile_quote(item: Contractor) -> str:
    parts = _fragments(item.description)
    part = next((p for p in parts if re.search(r"опыт|лет|юмор|манер", p, re.I)), parts[0] if parts else "")
    experience = re.search(r"\b[Оо]пыт[^.!?]{0,70}?\b\d+\s+(?:лет|года?)\b", part)
    if experience:
        return experience.group(0)
    return part if len(part) <= 240 else part[:240].rsplit(" ", 1)[0] + "…"


def _explain(contractor: Contractor, request: SearchRequest) -> str:
    fragments = _fragments(contractor.description)
    evidence = _preference_evidence(request.preferences, contractor.description)
    profile_fact = evidence or profile_quote(contractor)
    facts = [
        f"В профиле указано: «{profile_fact}»" if profile_fact else "Описание профиля отсутствует",
        f"свободен по календарю {request.event_date.strftime('%d.%m.%Y')}",
        f"берёт формат «{request.event_format}»",
        (f"цена от {contractor.price:,} ₸" +
         (f" при бюджете {request.budget:,} ₸" if request.budget is not None else "; бюджет не ограничен")).replace(",", " "),
    ]
    if request.language:
        facts.append(f"работает на языке «{request.language}»")
    if request.duration_hours is not None:
        if contractor.max_hours is None:
            facts.append("услуга не привязана к длительности присутствия")
        else:
            facts.append(f"доступен до {contractor.max_hours} ч")
    if request.preferences.strip():
        if evidence:
            facts.append(f"фрагмент по пожеланию: «{evidence}» (лексическое совпадение, со слов подрядчика)")
        else:
            facts.append("Обязательные условия подходят, но пожелание не подтверждено описанием")
    return "; ".join(facts) + "."


def recommend(contractors: Iterable[Contractor], request: SearchRequest, limit: int | None = 3,
              *, suggest_alternatives: bool = True, semantic_ranker=None,
              related_categories: dict[str, list[str]] | None = None) -> SearchResult:
    catalog = list(contractors)
    city_pool = [c for c in catalog if c.city == request.city]
    pool = [c for c in city_pool if request.category in c.categories]
    pipeline = [("Профилей", len(catalog)), (request.city, len(city_pool)), (request.category, len(pool))]
    stages = [("Формат", lambda c: request.event_format in c.event_formats),
              ("Свободны", lambda c: request.event_date.isoformat() not in c.busy_dates)]
    if request.budget is not None:
        stages.append(("В бюджете", lambda c: c.price <= request.budget))
    if request.language:
        stages.append(("Язык", lambda c: all(v in c.languages for v in requested_languages(request))))
    if request.duration_hours is not None:
        stages.append(("Длительность", lambda c: c.max_hours is None or c.max_hours >= request.duration_hours))
    eligible = pool
    for label, predicate in stages:
        eligible = [c for c in eligible if predicate(c)]
        pipeline.append((label, len(eligible)))
    rejected = [(c, rejection_reasons(c, request)) for c in pool]
    counts = Counter(reason for _, reasons in rejected for reason in reasons)
    near = tuple(NearMatch(c, reasons) for c, reasons in sorted(
        ((c, r) for c, r in rejected if r), key=lambda pair: (len(pair[1]), pair[0].price, pair[0].id)))
    mode, notice = "local", "Локальное сопоставление словоформ."
    similarities = [0.0] * len(eligible)
    if request.preferences.strip() and eligible:
        if semantic_ranker is not None:
            try:
                similarities = list(semantic_ranker(request.preferences, [c.description for c in eligible]))
                if len(similarities) != len(eligible) or not all(math.isfinite(v) and 0 <= v <= 1 for v in similarities):
                    raise ValueError("Invalid similarity scores")
                mode, notice = "embeddings", "Семантическая близость влияет на порядок, но не доказывает исполнение пожелания."
            except Exception:
                mode, notice = "fallback", "Семантический сервис недоступен. Использовано локальное сопоставление словоформ."
        if mode != "embeddings":
            similarities = _text_relevance(request.preferences, [c.description for c in eligible])
    cards = []
    for c, similarity in zip(eligible, similarities):
        proof = _preference_evidence(request.preferences, c.description)
        active = constraint_checks(c, request)
        confirmed = len(active) + bool(proof)
        total = len(active) + bool(request.preferences.strip())
        match_percent = round(100 * confirmed / total)
        budget_bonus = (max(0.0, 1 - c.price / request.budget) if request.budget else 0.0)
        # Ranking is separate from the visible percentage of confirmed conditions.
        preference_score = 80 * similarity if request.preferences.strip() else 0.0
        factors = (("Текстовая релевантность", round(preference_score, 2)), ("Запас бюджета", round(20 * budget_bonus, 2)))
        score = round(sum(value for _, value in factors), 2)
        reason = ("Порядок: 80% близость пожеланий + 20% запас бюджета; при равенстве — цена и ID."
                  if request.preferences.strip() else "Пожелания не заданы: порядок по цене, затем ID.")
        cards.append(Recommendation(c, score, _explain(c, request), factors, match_percent,
                                    tuple(label for label, passed in active if passed), proof, profile_quote(c), reason))
    cards.sort(key=lambda r: (-r.score, r.contractor.price, r.contractor.id))
    selected = tuple(cards[:limit])
    pipeline.append((f"TOP-{len(selected)}" if limit is not None else "Подходят", len(selected)))
    suggestions = _alternatives(catalog, request, related_categories or {}) if suggest_alternatives else ()
    if not pool:
        status, message = "category_absent", f"В городе {request.city} нет подрядчиков категории «{request.category}»."
    elif not eligible:
        status = "conditions_not_met"
        message = (f"В каталоге города {request.city} есть профили категории «{request.category}»: {len(pool)}. "
                   f"На {request.event_date:%d.%m.%Y} по заданным условиям подходящих нет.")
    else:
        status = "matched"
        message = f"Подобрано {len(selected)} из {len(eligible)} подходящих подрядчиков. В категории всего {len(pool)}."
    if counts:
        message += " Причины отсева: " + "; ".join(f"{name}: {count}" for name, count in counts.items()) + ". Причины могут пересекаться."
    return SearchResult(status=status, message=message, recommendations=selected,
                        rejection_counts=tuple(counts.items()), suggestions=suggestions,
                        pipeline=tuple(pipeline), near_matches=near, matched_count=len(eligible),
                        ranking_mode=mode, ranking_notice=notice)


def _alternatives(catalog: list[Contractor], request: SearchRequest,
                  related_categories: dict[str, list[str]]) -> tuple[Suggestion, ...]:
    def matches(changed):
        return {c.id for c in catalog if c.city == changed.city and changed.category in c.categories
                and not rejection_reasons(c, changed)}

    baseline = matches(request)
    suggestions = []

    def offer(changed, kind, label):
        found = matches(changed)
        added = len(found - baseline)
        if added:
            suggestions.append(Suggestion(changed, len(found),
                f"{label} → {len(found)} вариантов (+{added}). Остальные условия сохранены.", kind, added))
        return added

    if not baseline:
        for offset in range(1, (CALENDAR_END - request.event_date).days + 1):
            changed = replace(request, event_date=request.event_date + timedelta(days=offset))
            if offer(changed, "date", f"Дата {changed.event_date:%d.%m.%Y}"):
                break
        prices = sorted({c.price for c in catalog if c.city == request.city and request.category in c.categories
                         and request.budget is not None and c.price > request.budget})
        for price in prices:
            if offer(replace(request, budget=price), "budget",
                     f"Бюджет +{price - request.budget:,} ₸ (до {price:,} ₸)".replace(",", " ")):
                break
    if request.language:
        offer(replace(request, language=None), "language", "Любой язык")
    if request.duration_hours:
        for hours in range(request.duration_hours - 1, 0, -1):
            if offer(replace(request, duration_hours=hours), "duration", f"Сократить до {hours} ч"):
                break
    for category in related_categories.get(request.category, []):
        if category != request.category:
            offer(replace(request, category=category), "category", f"Сменить категорию на «{category}»")
    return tuple(suggestions)
