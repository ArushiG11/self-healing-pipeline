import requests
import streamlit as st

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://localhost:8000"

st.set_page_config(page_title="Review RAG", page_icon="🔎", layout="centered")
st.title("🔎 Ask the Reviews")
st.caption(
    "Grounded answers from 45k+ Amazon Electronics reviews — "
    "retrieval via pgvector + bge embeddings, generation via Llama 3.1. "
    "Answers cite their sources and refuse when the reviews don't cover it."
)

question = st.text_input(
    "Ask a shopper question",
    placeholder="e.g. Do wireless earbuds have battery problems?",
)

col1, col2 = st.columns([1, 2])
with col1:
    use_filter = st.checkbox("Only 4★+ reviews")
with col2:
    st.write("")  # spacing

if st.button("Ask", type="primary") and question.strip():
    payload = {"question": question}
    if use_filter:
        payload["min_rating"] = 4.0

    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"API unavailable: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(data["answer"])

    st.markdown("### Sources")
    for s in data["sources"]:
        with st.expander(
            f"[{s['rank']}] score {s['score']:.3f} · rating {s['rating']} · {s['asin']}"
        ):
            st.write(s["text"])