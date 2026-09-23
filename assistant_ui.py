"""Streamlit chat; credentials never enter chat history or tool arguments."""
import json
from datetime import date

import streamlit as st

from assistant import CHAT_VERSION, AssistantConfig, AssistantError, SearchTools, request_dict, run_turn


def describe_request(request):
    event_date = date.fromisoformat(request["event_date"]).strftime("%d.%m.%Y")
    parts = [request["city"], request["category"], request["event_format"], event_date]
    parts.append("бюджет без ограничений" if request["budget"] is None else
                 f"бюджет до {request['budget']:,} ₸".replace(",", " "))
    parts.append(f"язык: {request['language']}" if request["language"] else "любой язык")
    parts.append(f"длительность: {request['duration_hours']} ч" if request["duration_hours"] else "без ограничения длительности")
    return " · ".join(parts)


def render_assistant(contractors, apply_request):
    st.divider()
    st.header("ИИ-агент подбора подрядчиков")
    st.caption("Опишите мероприятие своими словами. Условия формы меняются только кнопкой «Применить». "
               "Сообщения и найденные сведения из каталога отправляются выбранному провайдеру.")
    provider = st.selectbox("Провайдер ИИ", ["openai", "nvidia"], key="ai_provider",
                            format_func=lambda name: "OpenAI" if name == "openai" else "NVIDIA NIM")
    try:
        settings = dict(st.secrets)
    except FileNotFoundError:
        settings = {}
    try:
        config = AssistantConfig.from_settings(provider, settings)
    except AssistantError as exc:
        st.caption(str(exc))
        st.chat_input("Сначала настройте API-ключ", disabled=True, key="ai_message_disabled")
        return
    st.caption(f"Модель: {config.model}")
    # Provider/model changes never forward a previous provider's conversation.
    sessions = st.session_state.setdefault("ai_sessions", {})
    session_key = (provider, config.model, CHAT_VERSION)
    session = sessions.setdefault(session_key, {"history": [], "display": []})
    if st.button("Новый диалог", key="ai_reset"):
        sessions[session_key] = {"history": [], "display": []}
        st.rerun()

    def apply_agent_request(request):
        apply_request(request)
        confirmation = "Я выбрал и применил к форме эти условия: " + json.dumps(request_dict(request), ensure_ascii=False)
        session["history"].append({"role": "user", "content": confirmation})
        session["display"].append({"role": "user", "text": confirmation})

    def render(entry, index):
        with st.chat_message(entry["role"]):
            st.write(entry["text"])
            for output_index, output in enumerate(entry.get("tool_results", [])):
                result = output["result"]
                if output["name"] != "search_contractors" or "error" in result:
                    continue
                with st.container(border=True):
                    st.write("**Результат проверки по каталогу**")
                    st.write(result["message"])
                    st.caption(describe_request(result["request"]))
                    for rec in result["recommendations"]:
                        st.write(f"**{rec['name']} · {rec['id']}**")
                        st.write(rec["explanation"])
                    options = [("Применить условия поиска к форме", result["request"])]
                    for suggestion in result["suggestions"]:
                        st.write(suggestion["message"])
                        label = (f"Применить дату {suggestion['request']['event_date']}"
                                 if suggestion["request"]["event_date"] != result["request"]["event_date"]
                                 else f"Применить бюджет {suggestion['request']['budget']:,} ₸".replace(",", " "))
                        options.append((label, suggestion["request"]))
                    for option_index, (label, request) in enumerate(options):
                        st.button(label, key=f"ai_apply_{index}_{output_index}_{option_index}",
                                  on_click=apply_agent_request, args=(SearchTools(contractors).parse_request(request),))
            if entry.get("tool_results"):
                with st.expander("Вызванные функции и данные"):
                    st.json(entry["tool_results"])

    for index, entry in enumerate(session["display"]):
        render(entry, index)
    prompt = st.chat_input("Например: ведущий на свадьбу в Алматы 3 октября 2026, бюджет 1,5 млн ₸", key="ai_message")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        try:
            with st.spinner("Агент проверяет условия…"):
                turn = run_turn(config, contractors, prompt, session["history"])
        except AssistantError as exc:
            st.error(str(exc))
            st.caption("Сообщение не добавлено в историю. Его можно отправить повторно.")
        else:
            session["history"] = turn.history
            session["display"].extend([
                {"role": "user", "text": prompt},
                {"role": "assistant", "text": turn.text, "tool_results": turn.tool_results},
            ])
            st.rerun()
