import re
from html import escape

import streamlit as st

from assistant_ui import describe_request
from product import comparison_rows, detailed_reasons, search_id, toggle_shortlist


def render_comparison(rows):
    """Use wrapping HTML cells: the dataframe grid clips long profile text."""
    if not rows:
        return
    cell = lambda value: escape(str(value))
    headings = "".join(f'<th scope="col">{cell(row["Подрядчик"])}</th>' for row in rows)
    fields = list(dict.fromkeys(field for row in rows for field in row if field != "Подрядчик"))
    body = "".join(
        f'<tr><th scope="row">{cell(field)}</th>' +
        "".join(f'<td>{cell(row.get(field, "—"))}</td>' for row in rows) + "</tr>"
        for field in fields
    )
    minimum = max(560, 180 + 220 * len(rows)) if len(rows) > 3 else 560
    st.markdown(
        '<style>'
        '.comparison-scroll {width:100%; overflow-x:auto;}'
        '.contractor-comparison {width:100%; table-layout:fixed; border-collapse:collapse;}'
        '.contractor-comparison th,.contractor-comparison td {'
        'white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal;'
        'vertical-align:top; text-align:left; padding:12px 14px;'
        'border:1px solid #80808055; line-height:1.55; height:auto;}'
        '.contractor-comparison th:first-child {width:20%;}'
        '.contractor-comparison thead th {background:#80808018;}'
        '</style>'
        '<div class="comparison-scroll" role="region" aria-label="Сравнение подрядчиков" tabindex="0">'
        f'<table class="contractor-comparison" style="min-width:{minimum}px">'
        f'<thead><tr><th scope="col">Критерий</th>{headings}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def render_profile_feature(value):
    # Keep the original evidence unchanged; repair the extracted phrase only for display.
    feature = re.sub(r"^опытом\b", "Опыт", value.strip(), flags=re.I)
    st.write(f"**{feature}**" if re.match(r"^опыт\b", feature, flags=re.I) else feature or "Описание отсутствует.")


def render_result(payload, on_request, debug=False):
    request = payload["request"]
    key = search_id(request)
    st.header("Ваше мероприятие")
    st.write(describe_request(request))
    st.caption("Измените условия сообщением агенту или в «Ручной настройке».")
    (st.success if payload["status"] == "matched" else st.warning)(payload["message"])
    if payload.get("clarification"):
        st.info(payload["clarification"])
    if payload.get("ranking_mode") == "fallback":
        st.caption(payload["ranking_notice"])
    cards = payload["recommendations"][:3]
    if cards:
        st.caption("Match — доля подтверждённых условий, не оценка качества. Пожелания подтверждаются цитатами из профиля.")

    def card(rec):
        with st.container(border=True):
            st.subheader(rec["name"])
            st.caption(rec["id"] + (" · Синтетический профиль" if rec["synthetic"] else ""))
            st.caption("Город: " + (rec.get("city") or request.get("city") or "Не указан"))
            if rec.get("city_imputed"):
                st.caption("Город заполнен при подготовке датасета")
            st.metric("Цена от", f"{rec['price_from_kzt']:,} ₸".replace(",", " "))
            if rec.get("price_imputed"):
                st.caption("Цена заполнена при подготовке датасета")
            st.progress(rec["match_percent"] / 100, text=f"Match {rec['match_percent']}% · подтверждено")
            st.write(" · ".join("✓ " + value for value in rec["checks"] if value not in ("Город", "Категория")))
            st.write("**Особенность профиля**")
            if rec.get("profile_facts"):
                for fact in rec["profile_facts"]:
                    if fact["source"] == "description":
                        render_profile_feature(fact["value"])
                    else:
                        st.caption(f"{fact['label']}: {fact['value']}")
            else:
                render_profile_feature(rec["profile_quote"])
            if request["preferences"]:
                if rec["preference_evidence"]:
                    st.write(f"★ Фрагменты по пожеланию: «{rec['preference_evidence']}»")
                else:
                    st.info("Обязательные условия подходят, но пожелание не подтверждено описанием")
            else:
                st.caption("Дополнительные пожелания не заданы.")
            saved = rec["id"] in st.session_state.get("shortlist", {})
            st.button("Убрать из shortlist" if saved else "♡ В shortlist", key=f"save_{key}_{rec['id']}",
                      on_click=toggle_shortlist, args=(st.session_state, rec, request))

    if cards:
        columns = st.columns(min(3, len(cards)))
        for column, rec in zip(columns, cards[:3]):
            with column:
                card(rec)
        if len(cards) > 1:
            st.button("Сравнить этих подрядчиков", key=f"compare_button_{key}",
                      on_click=lambda: st.session_state.update({f"compare_{key}": True}))
            if st.session_state.get(f"compare_{key}"):
                st.subheader("Сравнение TOP-3")
                render_comparison(comparison_rows(cards[:3], request))
    if payload["suggestions"]:
        st.subheader("Что можно изменить")
        for index, suggestion in enumerate(payload["suggestions"]):
            st.button(suggestion["message"], key=f"suggestion_{index}" if not st.session_state.get("separate_results") else f"suggestion_{key}_{index}", on_click=on_request,
                      args=(suggestion["request"],))
    elif not cards and not payload.get("clarification"):
        st.info("Изменение только даты, бюджета, языка или длительности не дало вариантов. Уточните другие условия.")
    with st.expander(f"Почему не эти подрядчики? ({len(payload.get('near_matches', []))})"):
        st.caption("Эти профили не прошли обязательные условия и не входят в рекомендации. Причины могут пересекаться.")
        for candidate in payload.get("near_matches", []):
            st.write(f"**{candidate['name']} · {candidate['id']}**")
            for reason in detailed_reasons(candidate, request):
                st.write("✕ " + reason)
    if debug:
        with st.expander("Данные поиска для демонстрации"):
            st.json(payload)
