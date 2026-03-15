import re

import requests
import streamlit as st

st.set_page_config(page_title="Course RAG Chatbot", layout="centered")
st.title("Course RAG Chatbot (MVP)")
st.caption("Machine Learning course only (current MVP scope).")

api_base_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000").rstrip("/")
course_id = st.sidebar.text_input("Course ID", value="machine_learning")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _normalize_math_markdown(text: str) -> str:
    """
    Make model equations render reliably in Streamlit markdown.
    Converts LaTeX delimiters \\(...\\) and \\[...\\] into $...$ / $$...$$.
    """
    normalized = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    normalized = re.sub(r"\\\((.+?)\\\)", r"$\1$", normalized, flags=re.DOTALL)
    return normalized

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(_normalize_math_markdown(message["content"]))
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.markdown(
                        f"- `{citation['lecture_title']}` | `{citation['source_file']}` | `{citation['chunk_id']}`"
                        f"\n  - {citation['excerpt']}"
                    )

prompt = st.chat_input("Ask about the Machine Learning course")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            f"{api_base_url}/chat",
            json={"question": prompt, "course_id": course_id},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        with st.chat_message("assistant"):
            st.error(f"Could not reach API: {exc}")
        st.stop()

    answer = payload.get("answer", "")
    citations = payload.get("citations", [])
    grounded = bool(payload.get("grounded", False))

    with st.chat_message("assistant"):
        if grounded:
            st.success("Grounded answer from transcript evidence")
        else:
            st.warning("Insufficient evidence from transcripts")
        st.markdown(_normalize_math_markdown(answer))
        if citations:
            with st.expander("Sources"):
                for citation in citations:
                    st.markdown(
                        f"- `{citation['lecture_title']}` | `{citation['source_file']}` | `{citation['chunk_id']}`"
                        f"\n  - {citation['excerpt']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "grounded": grounded,
        }
    )
