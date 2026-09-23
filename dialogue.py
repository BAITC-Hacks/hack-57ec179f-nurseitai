"""User-owned event state. Catalog data and model tool arguments cannot rewrite it."""
import re
from calendar import monthrange
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
MONTH_FORMS = ("январь января январе", "февраль февраля феврале", "март марта марте",
               "апрель апреля апреле", "май мая мае", "июнь июня июне", "июль июля июле",
               "август августа августе", "сентябрь сентября сентябре", "октябрь октября октябре",
               "ноябрь ноября ноябре", "декабрь декабря декабре")
MONTH_NUMBERS = {form: number for number, forms in enumerate(MONTH_FORMS, 1) for form in forms.split()}
EARLIEST = r"ближай[шщ]\w*|перв\w*\s+свободн\w*"
CITY_ALIASES = {"Алматы": r"алматы|алмате|алмату|алма[- ]?ата|алма[- ]?ате"}


def empty_state():
    return dict(original_need="", current_need="", required_services=[], conditions={}, last_results=[],
                selected_contractor=None, viewed_profile=None, pending_fields=[], service_mode="combined",
                awaiting_service_mode=False, awaiting_separate=False, date_window=None)


def normalize(text):
    return text.casefold().replace("ё", "е").replace("видеосъекм", "видеосъем")


def is_any_reply(text):
    return bool(re.fullmatch(r"\s*(?:да[, ]+)?(?:(?:мне|это|все|остальное)\s+)*(?:без разницы|неважно|не важно|любые|любой|любая)"
                             r"(?:\s+(?:подойдет|подойдут))?\s*[.!]?", normalize(text)))


def resolve_services(text, categories):
    value = normalize(text)
    # Reviews describe proof of work; they are not a request for a videographer.
    value = re.sub(r"\b(?:фото|видео)[- ]?отзыв\w*", "", value)
    # Explicit exclusions cannot become requested services through keyword matching.
    value = re.sub(r"\bне\s+(?:фото\s*буд\w*|фото\s*зеркал\w*|видео\s*буд\w*|фотограф\w*|видеограф\w*)", "", value)
    if re.search(r"фото\s*буд|фото\s*зеркал|видео\s*буд", value):
        requested = ["Фото и видеобудки"]
    else:
        requested = []
        if re.search(r"\bфотограф\w*|\bфото\s*с[ъь]?ем\w*|\bфото\b", value):
            requested.append("Фотограф")
        if re.search(r"\bвидеограф\w*|\bвидео\s*с[ъь]?ем\w*|\bвидео\b", value):
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
    format_value = re.sub(r"\bне\s+(?:на\s+)?(?:свадьб\w*|той\b|корпоратив\w*|юбиле\w*|день\s+рождени\w*)", "", value)
    patch = {}
    for field, values in (("city", options["cities"]), ("event_format", options["event_formats"])):
        for option in values:
            city_match = bool(re.search(r"\b(?:" + CITY_ALIASES.get(option, re.escape(normalize(option))) + r")\b", value))
            # Catalog spelling is canonical; inflected city names are accepted as input.
            if (city_match or (terms(option) and terms(option) <= terms(value)) if field == "city" else terms(option) and terms(option) <= terms(format_value)):
                patch[field] = option
                break
    if "свадьба" in options["event_formats"] and re.search(r"свадьб\w*", format_value):
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
    amount = re.search(r"(?:бюджет\w*\s*(?:(?:до|на|теперь|максимум|не более)\s*)*|(?:до|максимум|не более)\s+)(\d[\d ]*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\w*|[кk]\b)?", value)
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
    if is_any_reply(text):
        patch.update({field: None for field in pending})
    if re.fullmatch(r"\s*\d[\d ]*\s*", value) and len(pending) == 1:
        number = int(value.replace(" ", ""))
        if pending[0] == "budget":
            patch["budget"] = number
        elif pending[0] == "duration_hours" and 1 <= number <= 12:
            patch["duration_hours"] = number
    pref = re.search(r"(?:пожелани\w*\s*[:—-]\s*)(.+)", text, re.I)
    if pref:
        patch["preferences"] = pref[1].strip()
    else:
        pref = re.search(r"\bбез\s+(?!разницы|ограничени)([^.!?]+)", text, re.I)
        if pref:
            patch["preferences"] = pref[0].strip()
        else:
            # Preserve the user's own wording; these are wishes, not catalog facts.
            phrases = [match[0].strip() for match in re.finditer(
                r"\b(?:тонк\w*|интеллигентн\w*|добрым|современн\w*)\s+юмор\w*"
                r"|\b(?:интеллигентн\w*|тактичн\w*|сдержанн\w*|харизматичн\w*)"
                r"|\b(?:с\s+)?(?:видео|фото)[- ]?отзыв\w*"
                r"|\bсовременн\w*\s+подач\w*", text, re.I)]
            if phrases:
                patch["preferences"] = "; ".join(dict.fromkeys(phrases))
    return patch


def extract_date_window(text, today):
    """A month is a range, never an invented day. Keep explicit years intact."""
    value = normalize(text)
    month = re.search(r"\b(" + "|".join(MONTH_NUMBERS) + r")\b", value)
    if not month:
        return None
    number = MONTH_NUMBERS[month[1]]
    explicit_year = (re.match(r"\s*,?\s+(20\d\d)\b(?!\s*(?:тенге|₸|тыс|млн))", value[month.end():]) or
                     re.search(r"\b(20\d\d)\s+год\w*\b", value))
    year = int(explicit_year[1]) if explicit_year else today.year
    if not explicit_year and date(year, number, monthrange(year, number)[1]) < today:
        year += 1
    return {"start": date(year, number, 1).isoformat(),
            "end": date(year, number, monthrange(year, number)[1]).isoformat(),
            "mode": "earliest" if re.search(EARLIEST, value) else "choose"}


def missing_fields(state):
    window = state.get("date_window")
    return [f for f in FIELDS if f not in state["conditions"] and not (
        f == "event_date" and window and window["mode"] == "earliest")]


def prepare_turn(previous, text, options, today, contractors):
    state = deepcopy(previous or empty_state())
    state.setdefault("current_need", state["original_need"])
    state.setdefault("date_window", None)
    value = normalize(text)
    selected = None
    if re.search(r"выбираю|выбрал|остановимся", value):
        candidates = [c for r in state["last_results"] for c in r["recommendations"]
                      if normalize(c["id"]) in value or normalize(c["name"]) in value]
        if len(candidates) == 1:
            selected = candidates[0]["id"]
    viewing = bool(re.search(r"откр\w*|профиль|почему не|почему .*не подход", value))
    before = (state["required_services"][:], deepcopy(state["conditions"]), deepcopy(state["date_window"]))
    pending = [f for f in state["pending_fields"] if f in missing_fields(state)]
    answering_preference = bool(state.get("pending_preference"))
    patch = extract_conditions(text, options, today, pending)
    if "preferences" in patch:
        state["pending_preference"] = False
    if state.get("pending_preference") and not patch and not resolve_services(text, options["categories"]):
        patch["preferences"] = "" if is_any_reply(text) else text.strip()
        state["pending_preference"] = False
    if re.search(r"\bостальн\w*\s+(?:" + ANY + r")", value):
        remaining = set(missing_fields(state)) | {"budget", "language", "duration_hours"}
        patch.update({field: None for field in remaining if field not in patch})
    if (state["date_window"] and state["date_window"]["mode"] == "choose" and
            "event_date" in patch and is_any_reply(text)):
        # "Any day" answers the day-within-this-month question, not a new month.
        patch.pop("event_date")
        state["date_window"]["mode"] = "earliest"
    state["conditions"].update(patch)
    if "event_date" in patch:
        state["date_window"] = None
    else:
        window = extract_date_window(text, today)
        if window:
            if state["date_window"] and state["date_window"]["mode"] == "earliest":
                window["mode"] = "earliest"
            state["date_window"] = window
            state["conditions"].pop("event_date", None)
        elif state["date_window"] and re.search(EARLIEST, value):
            state["date_window"]["mode"] = "earliest"
    selection_check = selected is not None or (state.get("selected_contractor") and
        bool(re.search(r"(?:он|она|этот|эта)\b.*подход|все еще подход|по-прежнему подход", value)))
    if selection_check or viewing:
        if patch or before[2] != state["date_window"]:
            state["last_results"] = []
        if selected is not None:
            state["selected_contractor"] = selected
        if selection_check:
            state["selection_check"] = True
        state["pending_fields"] = missing_fields(state)
        return state, "", True
    resolved = resolve_services(text, options["categories"])
    if state["awaiting_service_mode"] and re.search(r"\bобе\b|обо\w*|одного|один|вместе|совмещ", value):
        resolved = state["required_services"]
        state["awaiting_service_mode"] = False
    separate = bool(re.search(r"отдельно|два специалиста|двух исполнителей|двух специалистов", value))
    if (state["awaiting_separate"] and re.fullmatch(r"\s*(да|давай|давайте|согласен)[.!]?\s*", value)) or separate:
        state.update(service_mode="separate", awaiting_separate=False, awaiting_service_mode=False)
    if resolved:
        if not state["original_need"]:
            state["original_need"] = text
        if state["required_services"] != resolved:
            state["current_need"] = text
        if state["required_services"] != resolved:
            state.update(service_mode="separate" if separate else "combined", selected_contractor=None,
                         awaiting_separate=False, awaiting_service_mode=False)
        state["required_services"] = resolved
        if len(resolved) > 1 and not re.search(r"\bобе\b|обо\w*|одного|один|вместе|совмещ|отдельно|двух|два", value):
            state["awaiting_service_mode"] = True
    elif not answering_preference and (re.search(r"\bтестиров\w*|^другое[.!]?$", value) or (
            re.search(r"нуж\w*|ищу|человек|специалист", value) and
            (not state["required_services"] or not extract_conditions(text, options, today) or
             (re.search(r"нужен|нужна|ищу|специалист|человек", value) and not re.search(r"бюджет|язык|дат|город|час|длительност", value))))):
        state.update(original_need=state["original_need"] or text, current_need=text,
                     required_services=[], last_results=[], selected_contractor=None,
                     awaiting_service_mode=False, awaiting_separate=False, service_mode="combined")
        return state, "Такую услугу пока не удалось сопоставить с каталогом. Какую задачу должен выполнять специалист?", False
    if before != (state["required_services"], state["conditions"], state["date_window"]):
        state["last_results"] = []
        state["selected_contractor"] = None
    soft_tail = re.search(r"\b(?:ведущ\w*|фотограф\w*|видеограф\w*)\s+((?:с|со)\s+[^,.!?]+)", text, re.I)
    if (soft_tail and "preferences" not in patch and len(state["required_services"]) == 1 and
            not re.match(r"(?:с|со)\s+(?:бюджет\w*|лимит\w*|цен\w*|язык\w*|длительност\w*|русск\w*|казахск\w*|английск\w*|\d+\s*час\w*)\b", soft_tail[1], re.I) and
            not resolve_services(soft_tail[1], options["categories"])):
        state["pending_preference"] = True
        state["pending_fields"] = []
        return state, f"Уточните пожелание «{soft_tail[1].strip()}»: что именно должно быть указано в профиле?", False
    if state["awaiting_service_mode"]:
        return state, "Нужны обе услуги у одного подрядчика или отдельно фотограф и видеограф?", False
    if len(state["required_services"]) > 1 and state["service_mode"] == "combined" and not any(
            set(state["required_services"]) <= set(c.categories) for c in contractors):
        state["awaiting_separate"] = True
        services = ", ".join(state["required_services"])
        label = "отдельно фотографа и видеографа" if set(state["required_services"]) == {"Фотограф", "Видеограф"} else f"отдельных исполнителей для услуг: {services}"
        return state, f"В каталоге нет подрядчика, у которого заявлены все услуги: {services}. Рассмотреть {label}?", False
    if state["date_window"]:
        start, end = (date.fromisoformat(state["date_window"][key]) for key in ("start", "end"))
        available_start = max(today, date.fromisoformat(options["calendar_start"]))
        available_end = date.fromisoformat(options["calendar_end"])
        if end < available_start or start > available_end:
            state["pending_fields"] = ["event_date"]
            return state, (f"Для указанного периода нет доступных дат в календаре каталога. "
                           f"Календарь: {date.fromisoformat(options['calendar_start']):%d.%m.%Y} — "
                           f"{available_end:%d.%m.%Y}. "
                           "Какую другую дату или месяц рассмотреть?"), False
    state["pending_fields"] = missing_fields(state)
    return state, "", False


def missing_question(state):
    if not state["required_services"]:
        return "Какую задачу должен выполнять специалист?"
    missing = missing_fields(state)
    questions = dict(QUESTIONS)
    if state.get("date_window") and state["date_window"]["mode"] == "choose":
        questions["event_date"] = "На какой день в указанном месяце или найти ближайшую свободную дату?"
    return " ".join(questions[f] for f in missing) + (" Можно указать: без разницы." if missing else "")


def question_fragments(text):
    # A recap such as "Понял: Алматы, ближайшая дата в декабре" is not a question.
    return [part for part in re.findall(r"[^.!?\n]+[.!?]?", normalize(text))
            if "?" in part or re.search(r"\b(?:уточните|укажите|скажите|напишите|назовите)\b", part)]


def question_fields(text):
    patterns = {"city": r"\bгород\w*", "event_date": r"\bдат\w*|\bдень\b|\bмесяц\w*", "event_format": r"\bформат\w*",
                "budget": r"\bбюджет\w*|\bсумм\w*", "language": r"\bязык\w*", "duration_hours": r"\bчас\w*|\bдлительност\w*"}
    prompts = question_fragments(text)
    return [field for field, pattern in patterns.items() if any(re.search(pattern, part) for part in prompts)]


def search_arguments(state, category=None):
    services = [category] if category else state["required_services"]
    if not services or any(f not in state["conditions"] for f in FIELDS):
        return None
    return {"preferences": "", **state["conditions"], "category": services[0],
            **({"required_categories": services} if len(services) > 1 else {})}


def state_from_request(request, previous=None):
    state = deepcopy(previous or empty_state())
    state["required_services"] = list(request.get("required_categories") or [request["category"]])
    state["current_need"] = ", ".join(state["required_services"])
    state["original_need"] = state["original_need"] or state["current_need"]
    state["conditions"] = {k: request[k] for k in (*FIELDS, "preferences")}
    state.update(pending_fields=[], awaiting_service_mode=False, awaiting_separate=False,
                 service_mode="combined", selected_contractor=None, date_window=None, pending_preference=False)
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
                            ": " + r["message"] + ("\n" + r["clarification"] if r.get("clarification") else "") for r in results)
    profiles = [o["result"] for o in outputs if o["name"] in ("get_contractor", "explain_contractor_match") and "error" not in o["result"]]
    if profiles:
        p = profiles[-1]
        description = safe_text(p.get("description", ""), "Описание не удалось отобразить.")
        return (f"Профиль «{p['name']}». Цена от {p['price_from_kzt']:,} ₸. " +
                f"В профиле указано: {description}\n\n" +
                ("Несовпадения: " + "; ".join(p["mismatches"]) + ". " if p.get("mismatches") else "") +
                "Это просмотр профиля, не рекомендация. Исходные условия сохранены.")
    return missing_question(state) or "Не удалось сформировать ответ. Условия сохранены; можно повторить поиск."
