import json
from datetime import date
from pathlib import Path

import streamlit as st

from assistant import AssistantConfig, AssistantError, SearchTools, request_dict
from assistant_ui import describe_request, render_assistant
from product import comparison_rows, record_search
from recommender import CALENDAR_END, SearchRequest, load_contractors
from results_ui import render_result
from semantic import EmbeddingRanker


ROOT = Path(__file__).parent


@st.cache_data
def get_contractors():
    return load_contractors(ROOT / "data/contractors.csv")


st.set_page_config(page_title="Nurseitai · Подрядчики для вашего события", page_icon="✦", layout="wide")
st.markdown("""<style>
.block-container {max-width:1280px;padding-top:2rem;}
[data-testid="stForm"] {border-radius:18px;}
[data-testid="stMetricValue"] {font-size:1.45rem;}
</style>""", unsafe_allow_html=True)
st.title("Nurseitai")
st.write("**Подрядчики для вашего события. С понятными основаниями выбора.**")
contractors = get_contractors()
try:
    settings = dict(st.secrets)
except FileNotFoundError:
    settings = {}
with st.sidebar:
    st.caption(f"{len(contractors)} профилей · календарь до 31.12.2026")
    with st.expander("Настройки AI"):
        provider = st.selectbox("Провайдер ИИ", ["openai", "nvidia"], key="ai_provider")
        embeddings_enabled = st.checkbox("Семантическое ранжирование", key="use_embeddings",
            help="Применяется к следующему поиску. Пожелания и описания прошедших фильтры профилей отправляются в OpenAI Embeddings API. Используется квота API.")
        debug = st.checkbox("Режим демонстрации: вызовы функций", key="debug")
        st.caption("Чат отправляет сообщения и результаты поиска выбранному провайдеру. "
                   "Embeddings используют OpenAI отдельно, независимо от провайдера чата.")

ranker = None
if embeddings_enabled:
    try:
        embedding_config = AssistantConfig.from_settings("openai", settings)
    except AssistantError:
        st.warning("Для embeddings нужен ключ OpenAI. Пока используется локальное сопоставление.")
    else:
        def ranker(query, documents):
            from openai import OpenAI
            with OpenAI(api_key=embedding_config.api_key, base_url="https://api.openai.com/v1",
                        timeout=20, max_retries=0) as client:
                return EmbeddingRanker(client, cache=st.session_state.setdefault("embedding_cache", {}))(query, documents)

# Category substitutions must be curated explicitly; no arbitrary service swaps.
mapping_path = ROOT / "data/related_categories.json"
related = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
service = SearchTools(contractors, semantic_ranker=ranker, related_categories=related)


def accept_result(payload):
    record_search(st.session_state, payload)
    st.session_state.search_request = service.parse_request(payload["request"])


def choose_request(args):
    request = service.parse_request(args)
    accept_result(service.search(request))
    chat_key = st.session_state.get("active_chat")
    session = st.session_state.get("ai_sessions", {}).get(chat_key)
    if session is not None:
        text = "Я выбрал эти условия: " + json.dumps(request_dict(request), ensure_ascii=False)
        session["history"].append({"role": "user", "content": text})
        session["display"].append({"role": "user", "text": "Условия изменены: " + describe_request(args)})


defaults = dict(city="Алматы", category="Ведущий", event_format="свадьба",
                event_date=date(2026, 10, 14), budget=1_500_000, budget_unlimited=False,
                languages_selected=["русский"], duration=6, preferences="")
for name, value in defaults.items():
    st.session_state.setdefault(name, value)
pending = st.session_state.pop("pending_form", None)
if pending:
    request = service.parse_request(pending)
    from recommender import requested_languages
    st.session_state.update(city=request.city, category=request.category, event_format=request.event_format,
                            event_date=request.event_date, budget=request.budget or st.session_state.budget,
                            budget_unlimited=request.budget is None, languages_selected=list(requested_languages(request)),
                            duration=request.duration_hours or "Неважно", preferences=request.preferences)

# Main entry point precedes all search controls.
st.session_state.pop("active_chat", None)
render_assistant(contractors, service, accept_result, provider, settings, debug)
if st.session_state.get("demo_notice"):
    st.success(st.session_state.demo_notice)
if "active_result" in st.session_state:
    render_result(st.session_state.active_result, choose_request, debug)

with st.expander("Ручная настройка", expanded=False):
    st.caption("Те же условия и тот же поиск. Используйте для точечной корректировки запроса.")
    with st.form("search"):
        left, middle, right = st.columns(3)
        with left:
            city = st.selectbox("Город", sorted({c.city for c in contractors}), key="city")
            event_date = st.date_input("Дата мероприятия", min_value=date(2026, 9, 23), max_value=CALENDAR_END, key="event_date")
            category = st.selectbox("Категория", sorted({v for c in contractors for v in c.categories}), key="category")
        with middle:
            event_format = st.selectbox("Формат", sorted({v for c in contractors for v in c.event_formats}), key="event_format")
            unlimited = st.checkbox("Без ограничения бюджета", key="budget_unlimited")
            budget = st.number_input("Бюджет, ₸", min_value=1, step=50_000, key="budget")
        with right:
            languages = st.multiselect("Языки — нужны все выбранные", sorted({v for c in contractors for v in c.languages}),
                                      key="languages_selected", help="Пустой список означает любой язык.")
            duration = st.selectbox("Длительность", ["Неважно", *range(1, 13)], key="duration",
                                    format_func=lambda v: v if v == "Неважно" else f"{v} ч")
        preferences = st.text_area("Пожелания", key="preferences")
        submitted = st.form_submit_button("Обновить подбор", key="manual_submit", type="primary")
    if submitted:
        request = SearchRequest(city, event_date, event_format, category, None if unlimited else int(budget),
                                None if duration == "Неважно" else int(duration), "/".join(languages) or None, preferences)
        choose_request(request_dict(request))
        st.rerun()

with st.sidebar:
    shortlist = st.session_state.get("shortlist", {})
    st.subheader(f"♡ Shortlist ({len(shortlist)})")
    for item_id, saved in list(shortlist.items()):
        st.write(saved["contractor"]["name"])
        st.caption(describe_request(saved["request"]))
        if st.button("Убрать", key="remove_" + item_id):
            del shortlist[item_id]
            st.rerun()
    if len(shortlist) > 1:
        st.checkbox("Сравнить shortlist", key="compare_shortlist")
    st.subheader("Последние запросы")
    for index, payload in enumerate(st.session_state.get("search_history", [])):
        st.button(describe_request(payload["request"]), key=f"history_{index}",
                  on_click=choose_request, args=(payload["request"],))
    demos = st.session_state.get("demo_requests", {})
    with st.expander(f"Демо-заявки ({len(demos)})"):
        for draft in demos.values():
            st.write(f"{draft['id']} · {draft['contractor']['name']}")
            st.caption(draft["status"])

if st.session_state.get("compare_shortlist") and len(shortlist) > 1:
    st.subheader("Сравнение сохранённых вариантов")
    st.caption("Снимки из разных запросов. Даты и условия могут различаться; повторите поиск перед выбором.")
    rows = [comparison_rows([item["contractor"]], item["request"])[0] for item in shortlist.values()]
    st.dataframe(rows, hide_index=True, use_container_width=True)
