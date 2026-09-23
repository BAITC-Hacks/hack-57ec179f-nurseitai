"""Local evidence is conservative; semantic similarity never proves a promise."""
import re
from functools import lru_cache

import snowballstemmer

STEMMER = snowballstemmer.stemmer("russian")
STOP = set("для на и с в по хочу нужен нужна чтобы мне ведущий ведущего ведущая который которая это мероприятие свадьба свадьбу".split())
NEGATIONS = {"без", "не", "нет", "никаких", "никакого"}


def fragments(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|[\n•]+", text) if p.strip()]


@lru_cache(maxsize=2048)
def terms(text):
    words = re.findall(r"[а-яёa-z0-9]+", text.casefold().replace("ё", "е"))
    return frozenset(STEMMER.stemWords([w for w in words if w not in STOP and w not in NEGATIONS]))


def evidence(preferences, description):
    wanted = terms(preferences)
    if not wanted:
        return ""
    # Uninterpreted negation is not evidence. Require an exact normalized phrase
    # for negative requests rather than claiming that competitions == no competitions.
    words = set(re.findall(r"[а-яёa-z]+", preferences.casefold()))
    normalize = lambda value: " ".join(re.findall(r"[а-яёa-z0-9]+", value.casefold()))
    if words & NEGATIONS:
        phrase = normalize(preferences)
        return next((p for p in fragments(description) if phrase in normalize(p)
                     and ("не " + phrase) not in normalize(p)), "")
    selected, covered = [], set()
    for part in fragments(description):
        if set(re.findall(r"[а-яёa-z]+", part.casefold())) & NEGATIONS:
            continue
        common = wanted & terms(part)
        if common - covered:
            selected.append(part)
            covered.update(common)
        if wanted <= covered:
            # Each quoted sentence is verbatim. Ellipsis marks separate fragments.
            return " … ".join(selected)
    return ""
