"""Russian date control, independent of the browser's calendar locale."""

import re
from datetime import date
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import declare_component


_calendar = declare_component(
    "russian_date_picker", path=str(Path(__file__).parent / "date_picker_frontend")
)


def decode_date(payload, min_date: date, max_date: date) -> date | None:
    """Validate component values again on the server before using them in search."""
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    value = payload.get("value")
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        selected = date.fromisoformat(value)
    except ValueError:
        return None
    return selected if min_date <= selected <= max_date else None


def russian_date_input(label: str, *, key: str, min_value: date, max_value: date,
                       reset: bool = False) -> date | None:
    """Keep form drafts separate from dates accepted by chat/history/actions."""
    current = st.session_state[key]
    state_key = f"_{key}_calendar"
    state = st.session_state.setdefault(state_key, {"value": current, "revision": 0})
    if reset or state["value"] != current:
        state = {"value": current, "revision": state["revision"] + 1}
        st.session_state[state_key] = state
    # A new key discards a stale form draft after an explicit server-side selection.
    payload = _calendar(
        label=label, value=current.isoformat(), min_date=min_value.isoformat(),
        max_date=max_value.isoformat(), revision=state["revision"],
        default={"value": current.isoformat(), "error": None},
        key=f"{key}_picker_{state['revision']}",
    )
    selected = decode_date(payload, min_value, max_value)
    if selected is not None:
        st.session_state[key] = selected
        state["value"] = selected
    return selected
