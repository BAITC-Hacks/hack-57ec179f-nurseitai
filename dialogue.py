"""User-owned event state. Catalog data and model tool arguments cannot rewrite it."""
import re
from copy import deepcopy
from datetime import date, timedelta

from preferences import terms


FIELDS = ("city", "event_date", "event_format", "budget", "language", "duration_hours")
QUESTIONS = {"city": "В каком городе?", "event_date": "На какую дату?",
             "event_format": "Какой формат мероприятия?", "budget": "Какой бюджет вам подходит?",
             "language": "Какой язык нужен?", "duration_hours": "На сколько часов?"}
ANY = r"без разницы|неважн\w*|не важн\w*|люб\w*|без ограничени\w*"
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
          "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}


def empty_state():
    return dict(original_need="", required_services=[], conditions={}, last_results=[],
                selected_contractor=None, viewed_profile=None, pending_fields=[], service_mode="combined",
                awaiting_service_mode=False, awaiting_separate=False)


def normalize(text):
    return text.casefold().replace("ё", "е").replace("видеосъекм", "видеосъем")


def resolve_services(text, categories):
    value = normalize(text)
    if re.search(r"фото\s*буд|фото\s*зеркал|видео\s*буд", value):
        requested = ["Фото и видеобудки"]
    else:
        requested = []
        if re.search(r"фото|фотограф", value):
            requested.append("Фотограф")
        if re.search(r"видео|видеограф", value):
            requested.append("Видеограф")
        if not requested:
            # Whole normalized word stems, not substring/fuzzy category similarity.
            words = terms(value)
            for category in sorted(categories, key=len, reverse=True):
                if category in ("Фото и видеобудки", "Фотограф", "Видеограф", "Другое"):
                    continue
                category_words = terms(category)
                # "ведущий" is a preference stopword, so match that category explicitly.
                match = (bool(category_words) and category_words <= words)
                if category == "Ведущий":
                    match = bool(re.search(r"\bведущ\w*", value))
                if match:
                    requested = [category]
                    break
    return requested if all(c in categories for c in requested) else []


def extract_conditions(text, options, today, pending=()):
    """Conservative common Russian forms. Unrecognized values remain unanswered."""
    value = normalize(text)
    patch = {}
    for field, values in (("city", options["cities"]), ("event_format", options["event_formats"])):
        for option in values:
            if (normalize(option) in value if field == "city" else terms(option) and terms(option) <= terms(value)):
                patch[field] = option
                break
    if "свадьба" in options["event_formats"] and re.search(r"свадьб\w*", value):
        patch["event_format"] = "свадьба"
    if "послезавтра" in value:
        patch["event_date"] = (today + timedelta(days=2)).isoformat()
    elif "завтра" in value:
        patch["event_date"] = (today + timedelta(days=1)).isoformat()
    elif "сегодня" in value:
        patch["event_date"] = today.isoformat()
    iso = re.search(r"\b(20\d\d-\d{2}-\d{2})\b", value)
    numeric = re.search(r"\b(\d{1,2})[./](\d{2})(?:[./](20\d\d))?\b(?!\s*(?:млн|миллион|тыс))", value)
    named = re.search(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")(?:\s+(20\d\d))?\b", value)
    try:
        if iso:
            patch["event_date"] = date.fromisoformat(iso[1]).isoformat()
        elif numeric:
            patch["event_date"] = date(int(numeric[3] or today.year), int(numeric[2]), int(numeric[1])).isoformat()
        elif named:
            patch["event_date"] = date(int(named[3] or today.year), MONTHS[named[2]], int(named[1])).isoformat()
    except ValueError:
        pass  # Never replace an invalid date with today.
    amount = re.search(r"(?:бюджет\w*\s*(?:до|на|теперь)?\s*|до\s+)(\d[\d ]*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\w*|[кk]\b)?", value)
    if not amount:
        amount = re.search(r"(\d[\d ]*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\w*|[кk]\b|тенге|₸)", value)
    if amount:
        number = float(amount[1].replace(" ", "").replace(",", "."))
        suffix = amount[2] or ""
        patch["budget"] = int(number * (1_000_000 if suffix.startswith(("млн", "миллион")) else
                                        1000 if suffix.startswith(("тыс", "к", "k")) else 1))
        if not suffix and re.match(r"\s*(?:час|ч\b)", value[amount.end():]):
            patch.pop("budget", None)
    hours = re.search(r"\b(\d{1,2})\s*(?:час\w*|ч\b)", value)
    if hours:
        patch["duration_hours"] = int(hours[1])
    languages = [v for v in options["languages"] if terms(v) and terms(v) <= terms(value)]
    if languages:
        patch["language"] = "/".join(languages)
    hints = {"city": r"город\w*", "event_date": r"дат\w*|день", "event_format": r"формат\w*",
             "budget": r"бюджет\w*|цен\w*", "language": r"язык\w*", "duration_hours": r"длительност\w*|час\w*"}
    for field, hint in hints.items():
        if re.search(rf"(?:{hint})\s*(?:—|-|:)?\s*(?:{ANY})|(?:{ANY})\s*(?:{hint})", value):
            patch[field] = None
    if re.fullmatch(r"\s*(?:все\s+)?(?:без разницы|неважно|не важно|любые|любой)\s*[.!]?", value):
        patch.update({field: None for field in pending})
    pref = re.search(r"(?:пожелани\w*\s*[:—-]\s*)(.+)", text, re.I)
    if pref:
        patch["preferences"] = pref[1].strip()
    else:
        pref = re.search(r"\bбез\s+(?!разницы|ограничени)([^.!?]+)", text, re.I)
        if pref:
            patch["preferences"] = pref[0].strip()
    return patch


def prepare_turn(previous, text, options, today, contractors):
    state = deepcopy(previous or empty_state())
    value = normalize(text)
    if re.search(r"выбираю|выбрал|остановимся", value):
        candidates = [c for r in state["last_results"] for c in r["recommendations"]
                      if normalize(c["id"]) in value or normalize(c["name"]) in value]
        if len(candidates) == 1:
            state["selected_contractor"] = candidates[0]["id"]
            return state, f"Выбран подрядчик «{candidates[0]['name']}». Это сохранённый выбор, не бронирование.", False
    viewing = bool(re.search(r"откр\w*|профиль|почему не|почему .*не подход", value))
    if viewing:
        return state, "", True
    before = (state["required_services"][:], deepcopy(state["conditions"]))
    state["conditions"].update(extract_conditions(text, options, today, state["pending_fields"]))
    resolved = resolve_services(text, options["categories"])
    if state["awaiting_service_mode"] and re.search(r"\bобе\b|обо\w*|одного|один|вместе|совмещ", value):
        resolved = state["required_services"]
        state["awaiting_service_mode"] = False
    separate = bool(re.search(r"отдельно|два специалиста|двух исполнителей|двух специалистов", value))
    if (state["awaiting_separate"] and re.fullmatch(r"\s*(да|давай|давайте|согласен)[.!]?\s*", value)) or separate:
        state.update(service_mode="separate", awaiting_separate=False, awaiting_service_mode=False)
    if resolved:
        state["original_need"] = text if not state["original_need"] or state["required_services"] != resolved else state["original_need"]
        if state["required_services"] != resolved:
            state.update(service_mode="separate" if separate else "combined", selected_contractor=None,
                         awaiting_separate=False, awaiting_service_mode=False)
        state["required_services"] = resolved
        if len(resolved) > 1 and not re.search(r"\bобе\b|обо\w*|одного|один|вместе|совмещ|отдельно|двух|два", value):
            state["awaiting_service_mode"] = True
    elif re.search(r"\bтестиров\w*|^другое[.!]?$", value) or (
            re.search(r"нуж\w*|ищу|человек|специалист", value) and
            (not state["required_services"] or not extract_conditions(text, options, today))):
        state.update(original_need=text, required_services=[], last_results=[], selected_contractor=None,
                     awaiting_service_mode=False, awaiting_separate=False, service_mode="combined")
        return state, "Такую услугу пока не удалось сопоставить с каталогом. Какую задачу должен выполнять специалист?", False
    if before != (state["required_services"], state["conditions"]):
        state["last_results"] = []
        state["selected_contractor"] = None
    if state["awaiting_service_mode"]:
        return state, "Нужны обе услуги у одного подрядчика или отдельно фотограф и видеограф?", False
    if len(state["required_services"]) > 1 and state["service_mode"] == "combined" and not any(
            set(state["required_services"]) <= set(c.categories) for c in contractors):
        state["awaiting_separate"] = True
        return state, "В каталоге нет подрядчика, у которого заявлены обе услуги. Рассмотреть отдельно фотографа и видеографа?", False
    state["pending_fields"] = [f for f in FIELDS if f not in state["conditions"]]
    return state, "", False


def missing_question(state):
    if not state["required_services"]:
        return "Какую задачу должен выполнять специалист?"
    missing = [f for f in FIELDS if f not in state["conditions"]]
    return " ".join(QUESTIONS[f] for f in missing) + (" Можно указать: без разницы." if missing else "")


def question_fields(text):
    if not re.search(r"\?|уточните|укажите", normalize(text)):
        return []
    patterns = {"city": r"город", "event_date": r"дат|какой день", "event_format": r"формат",
                "budget": r"бюджет|сумм", "language": r"язык", "duration_hours": r"час|длительност"}
    return [field for field, pattern in patterns.items() if re.search(pattern, normalize(text))]


def search_arguments(state, category=None):
    services = [category] if category else state["required_services"]
    if not services or any(f not in state["conditions"] for f in FIELDS):
        return None
    return {"preferences": "", **state["conditions"], "category": services[0],
            **({"required_categories": services} if len(services) > 1 else {})}


def state_from_request(request, previous=None):
    state = deepcopy(previous or empty_state())
    state["required_services"] = list(request.get("required_categories") or [request["category"]])
    state["conditions"] = {k: request[k] for k in (*FIELDS, "preferences")}
    state.update(pending_fields=[], awaiting_service_mode=False, awaiting_separate=False,
                 service_mode="combined", selected_contractor=None)
    return state


def safe_text(answer, fallback):
    # Discard the whole technical reply, including JSON inside prose/code fences.
    if not isinstance(answer, str) or not answer.strip():
        return fallback
    if re.search(r"[\{\}]|```|function_call|tool_call|\b(?:city|event_date|event_format|contractor_id|budget)\s*['\"]?\s*:", answer):
        return fallback
    if answer.lstrip().startswith("[") or re.search(r"могу (?:приступить|начать)|подтвердите|можно (?:начать|приступить)|разреш\w* .*поиск", normalize(answer)):
        return fallback
    return answer.strip()


def fallback_text(outputs, state):
    results = [o["result"] for o in outputs if o["name"] == "search_contractors" and "error" not in o["result"]]
    if results:
        return "\n\n".join(" + ".join(r["request"].get("required_categories") or [r["request"]["category"]]) +
                            ": " + r["message"] for r in results)
    profiles = [o["result"] for o in outputs if o["name"] in ("get_contractor", "explain_contractor_match") and "error" not in o["result"]]
    if profiles:
        p = profiles[-1]
        description = safe_text(p.get("description", ""), "Описание не удалось отобразить.")
        return (f"Профиль «{p['name']}». Цена от {p['price_from_kzt']:,} ₸. " +
                f"В профиле указано: {description}\n\n" +
                ("Несовпадения: " + "; ".join(p["mismatches"]) + ". " if p.get("mismatches") else "") +
                "Это просмотр профиля, не рекомендация. Исходные условия сохранены.")
    return missing_question(state) or "Не удалось сформировать ответ. Условия сохранены; можно повторить поиск."
