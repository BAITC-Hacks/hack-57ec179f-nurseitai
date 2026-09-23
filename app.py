from datetime import date
from pathlib import Path

import streamlit as st

from recommender import SearchRequest, load_contractors, recommend


DATA_PATH = Path(__file__).parent / "data" / "contractors.csv"


@st.cache_data
def get_contractors():
    return load_contractors(DATA_PATH)


st.set_page_config(page_title="Подбор подрядчиков", page_icon="🔎", layout="wide")
st.title("Умный подбор подрядчиков")
st.caption("До трёх доступных вариантов с проверяемым объяснением выбора")

contractors = get_contractors()
cities = sorted({item.city for item in contractors})
categories = sorted({value for item in contractors for value in item.categories})
formats = sorted({value for item in contractors for value in item.event_formats})
languages = sorted({value for item in contractors for value in item.languages})

with st.form("search"):
    left, middle, right = st.columns(3)
    with left:
        city = st.selectbox("Город", cities)
        event_date = st.date_input(
            "Дата мероприятия",
            value=date(2026, 10, 14),
            min_value=date(2026, 9, 23),
            max_value=date(2026, 12, 31),
        )
        category = st.selectbox("Категория", categories)
    with middle:
        event_format = st.selectbox("Формат мероприятия", formats)
        budget = st.number_input("Бюджет, ₸", min_value=100_000, value=1_500_000, step=50_000)
        language = st.selectbox("Язык", ["Неважно", *languages])
    with right:
        use_duration = st.checkbox("Указать длительность")
        duration = st.slider("Длительность, часов", 1, 12, 6, disabled=not use_duration)
        st.write("")
        submitted = st.form_submit_button("Подобрать", type="primary", use_container_width=True)

if submitted:
    request = SearchRequest(
        city=city,
        event_date=event_date,
        event_format=event_format,
        category=category,
        budget=int(budget),
        duration_hours=duration if use_duration else None,
        language=None if language == "Неважно" else language,
    )
    result = recommend(contractors, request)

    if result.status == "matched":
        st.success(result.message)
        columns = st.columns(len(result.recommendations))
        for column, recommendation in zip(columns, result.recommendations):
            item = recommendation.contractor
            with column:
                st.subheader(item.name)
                st.write(f"**Категория:** {', '.join(item.categories)}")
                st.write(f"**Город:** {item.city}")
                st.write(f"**Цена от:** {item.price:,} ₸".replace(",", " "))
                st.write(f"**Оценка соответствия:** {recommendation.score:.0f}/100")
                st.info(recommendation.explanation)
                if item.synthetic:
                    st.caption("Синтетический профиль")
                with st.expander("Описание"):
                    st.write(item.description)
    else:
        st.warning(result.message)

    if result.rejection_counts:
        with st.expander("Почему другие кандидаты не прошли"):
            for reason, count in result.rejection_counts:
                st.write(f"- {reason}: {count}")

