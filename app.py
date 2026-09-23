from datetime import date
from pathlib import Path

import streamlit as st

from recommender import CALENDAR_END, SearchRequest, load_contractors, recommend
from assistant_ui import render_assistant


DATA_PATH = Path(__file__).parent / "data" / "contractors.csv"


@st.cache_data
def get_contractors():
    return load_contractors(DATA_PATH)


st.set_page_config(page_title="Подбор подрядчиков", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stForm"] {border: 1px solid #e5e7eb; border-radius: 18px; padding: 1.25rem;}
    [data-testid="stMetricValue"] {font-size: 1.45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

title, summary = st.columns([3, 1])
with title:
    st.title("Умный подбор подрядчиков")
    st.caption("Объяснимый подбор доступных event-подрядчиков по условиям мероприятия")
with summary:
    st.metric("Профилей в каталоге", len(get_contractors()))

contractors = get_contractors()
cities = sorted({item.city for item in contractors})
categories = sorted({value for item in contractors for value in item.categories})
formats = sorted({value for item in contractors for value in item.event_formats})
languages = sorted({value for item in contractors for value in item.languages})
st.session_state.setdefault("event_date", date(2026, 10, 14))
st.session_state.setdefault("budget", 1_500_000)
st.session_state.setdefault("budget_unlimited", False)
st.session_state.setdefault("category", "Ведущий")
st.session_state.setdefault("event_format", "свадьба")
st.session_state.setdefault("language", "русский")
st.session_state.setdefault("duration", 6)


def apply_suggestion(request):
    st.session_state.search_request = request
    for key, value in dict(city=request.city, event_date=request.event_date,
                           category=request.category, event_format=request.event_format,
                           budget=request.budget if request.budget is not None else st.session_state.budget,
                           budget_unlimited=request.budget is None,
                           language=request.language or "Неважно",
                           duration=request.duration_hours or "Неважно",
                           preferences=request.preferences).items():
        st.session_state[key] = value


with st.form("search"):
    left, middle, right = st.columns(3)
    with left:
        city = st.selectbox("Город", cities, key="city")
        event_date = st.date_input(
            "Дата мероприятия",
            min_value=date(2026, 9, 23),
            max_value=CALENDAR_END,
            key="event_date",
        )
        category = st.selectbox("Категория", categories, key="category")
    with middle:
        event_format = st.selectbox("Формат мероприятия", formats, key="event_format")
        budget_unlimited = st.checkbox("Без ограничения бюджета", key="budget_unlimited",
                                       help="При включении цена не ограничивает поиск; сумма ниже не используется.")
        budget = st.number_input("Бюджет, ₸", min_value=1, step=50_000, key="budget")
        language_options = ["Неважно", *languages]
        language = st.selectbox("Язык", language_options, key="language")
    with right:
        duration = st.selectbox(
            "Длительность",
            ["Неважно", *range(1, 13)],
            key="duration",
            format_func=lambda value: value if value == "Неважно" else f"{value} ч",
        )
    preferences = st.text_area(
        "Дополнительные пожелания",
        key="preferences",
        placeholder="Например: интеллигентный ведущий для деловой аудитории, без навязчивых конкурсов",
        help="NLP-модуль сравнит пожелания с описаниями доступных подрядчиков.",
    )
    submitted = st.form_submit_button("Подобрать", type="primary", use_container_width=True)

if submitted:
    request = SearchRequest(
        city=city,
        event_date=event_date,
        event_format=event_format,
        category=category,
        budget=None if budget_unlimited else int(budget),
        duration_hours=None if duration == "Неважно" else int(duration),
        language=None if language == "Неважно" else language,
        preferences=preferences,
    )
    st.session_state.search_request = request

if "search_request" in st.session_state:
    request = st.session_state.search_request
    result = recommend(contractors, request)

    if result.status == "matched":
        st.success(result.message)
        columns = st.columns(len(result.recommendations))
        for column, recommendation in zip(columns, result.recommendations):
            item = recommendation.contractor
            with column:
                with st.container(border=True):
                    st.subheader(item.name)
                    st.caption(" · ".join((item.city, ", ".join(item.categories))))
                    st.metric("Цена от", f"{item.price:,} ₸".replace(",", " "))
                    st.progress(min(1.0, recommendation.score / 100), text=f"Соответствие: {recommendation.score:.0f}/100")
                    st.info(recommendation.explanation)
                    if item.synthetic:
                        st.caption("Синтетический профиль")
                    with st.expander("Почему такая оценка"):
                        for factor, value in recommendation.factors:
                            st.write(f"{factor}: **{value:.1f}**")
                    with st.expander("Описание подрядчика"):
                        st.write(item.description)
    else:
        st.warning(result.message)
        for index, suggestion in enumerate(result.suggestions):
            st.info(suggestion.message)
            st.button("Применить перенос даты" if suggestion.request.event_date != request.event_date
                      else "Применить новый бюджет", key=f"suggestion_{index}",
                      on_click=apply_suggestion, args=(suggestion.request,))
        if result.status == "conditions_not_met" and not result.suggestions:
            st.info("Перенос даты вперёд до 31.12.2026 или увеличение только бюджета "
                    "не дают вариантов. Нужно пересмотреть другие условия.")

    if result.rejection_counts:
        with st.expander("Почему другие кандидаты не прошли"):
            for reason, count in result.rejection_counts:
                st.write(f"- {reason}: {count}")

render_assistant(contractors, apply_suggestion)
