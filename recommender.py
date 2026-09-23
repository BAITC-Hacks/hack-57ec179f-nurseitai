from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[а-яёa-z0-9]+", value.casefold()))


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


@dataclass(frozen=True)
class Recommendation:
    contractor: Contractor
    score: float
    explanation: str


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


def _score(contractor: Contractor, request: SearchRequest) -> float:
    # Every ranked contractor already passed the hard constraints. The score
    # differentiates candidates transparently without letting an LLM reorder them.
    budget_cushion = max(0.0, 1.0 - contractor.price / request.budget)
    description_overlap = len(
        _tokens(contractor.description)
        & _tokens(f"{request.event_format} {request.category} {request.language or ''}")
    )
    duration_buffer = 0.0
    if request.duration_hours is not None and contractor.max_hours is not None:
        duration_buffer = min(1.0, (contractor.max_hours - request.duration_hours) / 4)

    return round(
        50
        + 25 * budget_cushion
        + 15 * min(1.0, description_overlap / 2)
        + 10 * max(0.0, duration_buffer),
        2,
    )


def _explain(contractor: Contractor, request: SearchRequest) -> str:
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

    ranked = sorted(
        eligible,
        key=lambda item: (-_score(item, request), item.price, item.id),
    )
    recommendations = tuple(
        Recommendation(item, _score(item, request), _explain(item, request))
        for item in ranked[:limit]
    )
    suffix = "" if len(eligible) >= limit else f" Подходящих найдено только {len(eligible)}."
    return SearchResult(
        status="matched",
        message=f"Подобрано {len(recommendations)} из {len(eligible)} подходящих подрядчиков.{suffix}",
        recommendations=recommendations,
        rejection_counts=rejection_counts,
    )

