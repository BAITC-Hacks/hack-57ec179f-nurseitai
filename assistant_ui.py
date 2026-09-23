"""AI-first dialogue. Search results render once, in the shared results panel."""
from datetime import date

import streamlit as st

from assistant import CHAT_VERSION, AssistantConfig, AssistantError, run_turn


def describe_request(request):
    parts = [request["city"], request["category"], request["event_format"],
             date.fromisoformat(request["event_date"]).strftime("%d.%m.%Y")]
    parts.append("бюджет без ограничений" if request["budget"] is None else
                 f"бюджет до {request['budget']:,} ₸".replace(",", " "))
    parts.append(f"язык: {request['language']}" if request["language"] else "любой язык")
    parts.append(f"длительность: {request['duration_hours']} ч" if request["duration_hours"] else "любая длительность")
    if request.get("preferences"):
        parts.append("пожелания: " + request["preferences"])
    return " · ".join(parts)


def current_session(provider, model):
    sessions = st.session_state.setdefault("ai_sessions", {})
    return sessions.setdefault((provider, model, CHAT_VERSION), {"history": [], "display": []})


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
        session.update(history=[], display=[])
        st.session_state.pop("active_result", None)
        st.rerun()
    with st.container():
        for entry in session["display"]:
            with st.chat_message(entry["role"]):
                st.write(entry["text"])
                if debug and entry.get("tool_results"):
                    with st.expander("Вызванные функции и данные"):
                        st.json(entry["tool_results"])
        prompt = st.chat_input("Нужен ведущий на свадьбу в Алматы…", key="ai_message")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.spinner("Проверяем условия и профили…"):
                turn = run_turn(config, contractors, prompt, session["history"], search_tools=service)
        except AssistantError as exc:
            st.error(str(exc))
            st.caption("Сообщение не записано в историю. Можно повторить запрос.")
        else:
            session["history"] = turn.history
            session["display"].extend([
                {"role": "user", "text": prompt},
                {"role": "assistant", "text": turn.text, "tool_results": turn.tool_results},
            ])
            for output in turn.tool_results:
                if output["name"] == "search_contractors" and "error" not in output["result"]:
                    on_result(output["result"])
            st.rerun()
