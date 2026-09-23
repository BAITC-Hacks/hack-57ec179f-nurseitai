"""AI-first dialogue. Search results render once, in the shared results panel."""
from datetime import date

import streamlit as st

from assistant import CHAT_VERSION, AssistantConfig, AssistantError, run_turn
from dialogue import empty_state, safe_text, search_arguments


def describe_request(request):
    parts = [request["city"] or "любой город", " + ".join(request.get("required_categories") or [request["category"]]),
             request["event_format"] or "любой формат",
             date.fromisoformat(request["event_date"]).strftime("%d.%m.%Y") if request["event_date"] else "любая дата (доступность не проверена)"]
    parts.append("бюджет без ограничений" if request["budget"] is None else
                 f"бюджет до {request['budget']:,} ₸".replace(",", " "))
    parts.append(f"язык: {request['language']}" if request["language"] else "любой язык")
    parts.append(f"длительность: {request['duration_hours']} ч" if request["duration_hours"] else "любая длительность")
    if request.get("preferences"):
        parts.append("пожелания: " + request["preferences"])
    return " · ".join(parts)


def current_session(provider, model):
    sessions = st.session_state.setdefault("ai_sessions", {})
    return sessions.setdefault((provider, model, CHAT_VERSION), {"history": [], "display": [], "state": empty_state()})


def render_assistant(contractors, service, on_result, provider, settings, debug=False):
    st.header("Расскажите о вашем мероприятии")
    st.caption("Агент уточнит условия, проверит каталог и объяснит, чем отличаются подходящие подрядчики.")
    try:
        config = AssistantConfig.from_settings(provider, settings)
    except AssistantError as exc:
        st.info(str(exc))
        st.chat_input("Чат пока недоступен — используйте ручную настройку ниже", disabled=True)
        return
    session = current_session(provider, config.model)
    st.session_state.active_chat = (provider, config.model, CHAT_VERSION)
    if st.button("Новый диалог", key="ai_reset"):
        session.update(history=[], display=[], state=empty_state())
        st.session_state.pop("active_result", None)
        st.session_state.pop("separate_results", None)
        st.rerun()
    with st.container():
        for entry in session["display"]:
            with st.chat_message(entry["role"]):
                st.write(safe_text(entry["text"], "Не удалось сформировать текст ответа.") if entry["role"] == "assistant" else entry["text"])
                if debug and entry.get("tool_results"):
                    with st.expander("Вызванные функции и данные"):
                        st.json(entry["tool_results"])
        prompt = st.chat_input("Нужен ведущий на свадьбу в Алматы…", key="ai_message")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.spinner("Проверяем условия и профили…"):
                turn = run_turn(config, contractors, prompt, session["history"], search_tools=service, state=session["state"])
        except AssistantError as exc:
            st.error(str(exc))
            st.caption("Сообщение не записано в историю. Можно повторить запрос.")
        else:
            session["history"] = turn.history
            if turn.state is not None:
                session["state"] = turn.state
                updated_conditions = search_arguments(turn.state)
                if updated_conditions is not None:
                    st.session_state.pending_form = updated_conditions
                st.session_state.pop("active_result", None)
                st.session_state.pop("separate_results", None)
            session["display"].extend([
                {"role": "user", "text": prompt},
                {"role": "assistant", "text": turn.text, "tool_results": turn.tool_results},
            ])
            results = (turn.state["last_results"] if turn.state is not None else
                       [o["result"] for o in turn.tool_results if o["name"] == "search_contractors" and "error" not in o["result"]])
            for result in results:
                on_result(result)
            if turn.state is not None and turn.state["service_mode"] == "separate" and len(results) > 1:
                st.session_state.separate_results = results
            st.rerun()
