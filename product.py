"""Deterministic presentation helpers and demo business operations."""
import hashlib
import json
import re
from copy import deepcopy


def search_id(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def record_search(state, payload):
    state["active_result"] = payload
    key = search_id(payload["request"])
    history = [item for item in state.get("search_history", []) if search_id(item["request"]) != key]
    state["search_history"] = [deepcopy(payload), *history][:10]
    state["pending_form"] = payload["request"]


def toggle_shortlist(state, rec, request):
    shortlist = state.setdefault("shortlist", {})
    if rec["id"] in shortlist:
        del shortlist[rec["id"]]
    else:
        shortlist[rec["id"]] = {"contractor": deepcopy(rec), "request": deepcopy(request)}


def create_demo_request(state, rec, request):
    key = rec["id"] + ":" + search_id(request)
    requests = state.setdefault("demo_requests", {})
    if key not in requests:
        requests[key] = {"id": f"DEMO-REQ-{1024 + len(requests)}", "contractor": deepcopy(rec),
                         "request": deepcopy(request), "status": "Демо-черновик. Не отправлен."}
    return requests[key]


def comparison_rows(cards, request):
    rows = []
    for card in cards:
        quote = card.get("profile_quote", "")
        experience = re.search(r"[^.!?]*опыт[^.!?]*\d+\s*(?:лет|год|года)[^.!?]*", quote, re.I)
        rows.append({"Подрядчик": card["name"], "ID": card["id"],
                     "Match": f"{card['match_percent']}% подтверждено",
                     "Цена от": f"{card['price_from_kzt']:,} ₸".replace(",", " "),
                     "Языки": ", ".join(card["languages"]),
                     "Длительность": f"до {card['max_hours']} ч" if card["max_hours"] else "Не привязана ко времени",
                     "Опыт (со слов профиля)": experience.group(0).strip() if experience else "Не выделен из описания",
                     "Свободен по календарю": request["event_date"] or "Дата не задана — не проверено",
                     "Почему подходит": quote or "Описание отсутствует",
                     "Пожелания": ("Не заданы" if not request["preferences"] else
                                   card["preference_evidence"] or "Не подтверждены описанием")})
    return rows


def detailed_reasons(candidate, request):
    labels = {"заняты": f"Занят {request['event_date']}",
              "дороже бюджета": f"Цена от {candidate['price_from_kzt']:,} ₸ выше бюджета {request['budget']:,} ₸".replace(",", " ") if request['budget'] is not None else "",
              "формат не заявлен в каталоге": f"Формат «{request['event_format']}» не заявлен в каталоге",
              "не подходит язык": f"Нужен {request['language']}; в профиле: {', '.join(candidate['languages'])}",
              "не подходит длительность": f"Нужно {request['duration_hours']} ч; максимум {candidate['max_hours']} ч"}
    return [labels.get(reason, reason) for reason in candidate["reasons"]]
