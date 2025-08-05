# app_streamlit.py
"""
Simple Streamlit interface that sends headlines to
localhost:8006/score_headlines and shows the sentiment results.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8006/score_headlines"

st.set_page_config(page_title="Headline Sentiment")
st.title("Headline Sentiment Demo")

# -------- session state --------
headlines = st.session_state.setdefault("headlines", [""])

st.write("Add, edit, or remove headlines and then click “Classify”:")

for i, txt in enumerate(headlines):
    col_text, col_del = st.columns([10, 1])

    # editable text box
    headlines[i] = col_text.text_input(f"Headline {i+1}", value=txt, key=f"h_{i}")

    # delete button
    if col_del.button("Delete", key=f"del_{i}"):
        headlines.pop(i)
        st.rerun()

# button to append a new input
if st.button("Add another"):
    headlines.append("")
    st.rerun()

st.markdown("---")

# -------- API call --------
if st.button("Classify"):
    clean = [h.strip() for h in headlines if h.strip()]

    if not clean:
        st.warning("Please enter at least one headline.")
    else:
        try:
            resp = requests.post(API_URL, json={"headlines": clean}, timeout=10)
            resp.raise_for_status()
            st.success("Results")
            st.dataframe(
                {"Headline": clean, "Sentiment": resp.json()["scores"]},
                use_container_width=True,
            )
        except Exception as err:
            st.error(f"API error: {err}")
