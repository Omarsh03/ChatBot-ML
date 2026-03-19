import re
from concurrent.futures import ThreadPoolExecutor, Future

import requests
import streamlit as st

st.set_page_config(page_title="Course RAG Chatbot", layout="centered")
st.title("Course RAG Chatbot (MVP)")
st.caption("Multi-course assistant with automatic course routing.")

api_base_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000").rstrip("/")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False
if "future" not in st.session_state:
    st.session_state.future = None
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None


def _normalize_math_markdown(text: str) -> str:
    normalized = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    normalized = re.sub(r"\\\((.+?)\\\)", r"$\1$", normalized, flags=re.DOTALL)
    return normalized


def _call_api(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if message.get("grounded"):
                st.success("Grounded answer from transcript evidence")
            elif message.get("citations") == [] and message.get("content", "").startswith("Could not reach API:"):
                st.error(message["content"])
                continue
            elif message.get("content") == "Generation stopped by user.":
                st.info(message["content"])
                continue
            else:
                st.warning("Insufficient evidence from transcripts")
        st.markdown(_normalize_math_markdown(message["content"]))
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.markdown(
                        f"- `{citation['lecture_title']}` | `{citation['source_file']}` | `{citation['chunk_id']}`"
                        f"\n  - {citation['excerpt']}"
                    )

if st.session_state.is_processing:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Stop", type="secondary"):
            st.session_state.cancel_requested = True
            st.session_state.is_processing = False
            st.session_state.pending_prompt = None
            st.session_state.future = None
            st.session_state.messages.append(
                {"role": "assistant", "content": "Generation stopped by user.", "citations": [], "grounded": False}
            )
            st.rerun()

prompt = st.chat_input("Ask a question about the course material")

if prompt:
    if st.session_state.is_processing:
        st.session_state.queued_prompt = prompt
        st.rerun()
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.session_state.is_processing = True
        st.session_state.cancel_requested = False

        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m.get("role") in {"user", "assistant"} and m.get("content")
        ]

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            _call_api,
            f"{api_base_url}/chat",
            {"question": prompt, "chat_history": history_payload},
        )
        st.session_state.future = future
        st.rerun()

if st.session_state.is_processing and st.session_state.future is not None:
    import time

    future: Future = st.session_state.future

    with st.chat_message("assistant"):
        with st.spinner("Bot is thinking..."):
            while not future.done():
                time.sleep(0.3)

    if st.session_state.cancel_requested:
        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.session_state.future = None
        st.rerun()

    try:
        payload = future.result(timeout=0)
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Could not reach API: {exc}", "citations": [], "grounded": False}
        )
        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.session_state.future = None
        st.rerun()
    else:
        answer = payload.get("answer", "")
        citations = payload.get("citations", [])
        grounded = bool(payload.get("grounded", False))
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations, "grounded": grounded}
        )
        st.session_state.is_processing = False
        st.session_state.pending_prompt = None
        st.session_state.future = None
        st.rerun()

if not st.session_state.is_processing and st.session_state.queued_prompt:
    st.info(f"Queued message: {st.session_state.queued_prompt}")
    col_send, col_discard, _ = st.columns([1, 1, 4])
    with col_send:
        if st.button("Send", type="primary"):
            queued = st.session_state.queued_prompt
            st.session_state.queued_prompt = None
            st.session_state.messages.append({"role": "user", "content": queued})
            st.session_state.pending_prompt = queued
            st.session_state.is_processing = True
            st.session_state.cancel_requested = False

            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _call_api,
                f"{api_base_url}/chat",
                {"question": queued, "chat_history": history_payload},
            )
            st.session_state.future = future
            st.rerun()
    with col_discard:
        if st.button("Discard"):
            st.session_state.queued_prompt = None
            st.rerun()
