from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


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
    budget: int
    duration_hours: int | None = None
    language: str | None = None
    preferences: str = ""


@dataclass(frozen=True)
class Recommendation:
    contractor: Contractor
    score: float
    explanation: str
    factors: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SearchResult:
    status: str
    message: str
    recommendations: tuple[Recommendation, ...] = ()
    rejection_counts: tuple[tuple[str, int], ...] = ()


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
    words = re.findall(r"[а-яёa-z0-9]+", value.casefold())
    terms = [word for word in words if len(word) > 2]
    terms.extend(f"{left}_{right}" for left, right in zip(words, words[1:]))
    return terms


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


def _score_factors(
    contractor: Contractor, request: SearchRequest, relevance: float
) -> tuple[tuple[str, float], ...]:
    budget_cushion = max(0.0, 1.0 - contractor.price / request.budget)
    budget_score = 10 + 10 * budget_cushion

    if request.duration_hours is None:
        duration_score = 5.0
    elif contractor.max_hours is None:
        duration_score = 10.0
    else:
        duration_score = 5 + 5 * min(
            1.0, max(0.0, (contractor.max_hours - request.duration_hours) / 4)
        )

    language_score = 10.0 if request.language else 5.0
    relevance_score = 20 * min(1.0, relevance * 4)
    return (
        ("Обязательные условия", 40.0),
        ("Бюджет", round(budget_score, 2)),
        ("Язык", language_score),
        ("Длительность", round(duration_score, 2)),
        ("Текстовая релевантность", round(relevance_score, 2)),
    )


def _explain(contractor: Contractor, request: SearchRequest, relevance: float) -> str:
    facts = [
        f"свободен {request.event_date.strftime('%d.%m.%Y')}",
        f"берёт формат «{request.event_format}»",
        f"цена от {contractor.price:,} ₸ укладывается в бюджет".replace(",", " "),
    ]
    if request.language:
        facts.append(f"работает на языке «{request.language}»")
    if request.duration_hours is not None:
        if contractor.max_hours is None:
            facts.append("услуга не привязана к длительности присутствия")
        else:
            facts.append(f"доступен до {contractor.max_hours} ч")
    if request.preferences.strip():
        if relevance > 0:
            facts.append("описание профиля связано с указанными пожеланиями")
        else:
            facts.append("прямого совпадения с дополнительными пожеланиями не найдено")
    return "; ".join(facts).capitalize() + "."


def recommend(
    contractors: Iterable[Contractor], request: SearchRequest, limit: int = 3
) -> SearchResult:
    pool = [
        item
        for item in contractors
        if item.city == request.city and request.category in item.categories
    ]
    if not pool:
        return SearchResult(
            status="category_absent",
            message=f"В городе {request.city} нет подрядчиков категории «{request.category}».",
        )

    rejected = {"заняты": 0, "дороже бюджета": 0, "не берут формат": 0, "не подходит язык": 0, "не подходит длительность": 0}
    eligible: list[Contractor] = []
    requested_date = request.event_date.isoformat()

    for item in pool:
        reasons: list[str] = []
        if requested_date in item.busy_dates:
            reasons.append("заняты")
        if item.price > request.budget:
            reasons.append("дороже бюджета")
        if request.event_format not in item.event_formats:
            reasons.append("не берут формат")
        if request.language and request.language not in item.languages:
            reasons.append("не подходит язык")
        if (
            request.duration_hours is not None
            and item.max_hours is not None
            and item.max_hours < request.duration_hours
        ):
            reasons.append("не подходит длительность")

        if reasons:
            for reason in reasons:
                rejected[reason] += 1
        else:
            eligible.append(item)

    rejection_counts = tuple((key, value) for key, value in rejected.items() if value)
    if not eligible:
        detail = ", ".join(f"{name}: {count}" for name, count in rejection_counts)
        return SearchResult(
            status="conditions_not_met",
            message=f"Кандидаты есть, но никто не прошёл условия. {detail}.",
            rejection_counts=rejection_counts,
        )

    query_text = " ".join(
        part
        for part in (
            request.category,
            request.event_format,
            request.language or "",
            request.preferences,
        )
        if part
    )
    documents = [
        " ".join(
            [
                *item.categories,
                *item.event_formats,
                *item.languages,
                item.description,
            ]
        )
        for item in eligible
    ]
    relevance_scores = _text_relevance(query_text, documents)
    factors_by_id = {
        item.id: _score_factors(item, request, relevance)
        for item, relevance in zip(eligible, relevance_scores)
    }
    relevance_by_id = {
        item.id: relevance for item, relevance in zip(eligible, relevance_scores)
    }

    def total_score(item: Contractor) -> float:
        return round(sum(value for _, value in factors_by_id[item.id]), 2)

    ranked = sorted(
        eligible,
        key=lambda item: (-total_score(item), item.price, item.id),
    )
    recommendations = tuple(
        Recommendation(
            item,
            total_score(item),
            _explain(item, request, relevance_by_id[item.id]),
            factors_by_id[item.id],
        )
        for item in ranked[:limit]
    )
    suffix = "" if len(eligible) >= limit else f" Подходящих найдено только {len(eligible)}."
    return SearchResult(
        status="matched",
        message=f"Подобрано {len(recommendations)} из {len(eligible)} подходящих подрядчиков.{suffix}",
        recommendations=recommendations,
        rejection_counts=rejection_counts,
    )
