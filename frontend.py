"""
frontend.py — Streamlit UI for the OS Notes RAG Assistant
-----------------------------------------------------------
A polished, portfolio-ready chat interface on top of the existing rag.py /
rag_free.py backend. No changes to your working retrieval/generation logic --
this just gives it a real face for screenshots, GitHub README, and demos.

Run:
    streamlit run frontend.py
"""

import importlib
import streamlit as st

st.set_page_config(page_title="OS Notes RAG Assistant", page_icon="🖥️", layout="wide")

# ---------------- Theming (custom CSS) ----------------
st.markdown("""
<style>
    :root {
        --ink: #1B2430;
        --paper: #F4F6F8;
        --amber: #C97A1A;
        --amber-soft: #FCEFDD;
        --slate: #5C6773;
        --green: #2E8B57;
        --line: #D7DCE2;
    }
    .stApp { background-color: var(--paper); }
    h1, h2, h3 { color: var(--ink) !important; }
    .stChatMessage { border-radius: 4px; }
    .source-tag {
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        color: var(--slate);
        background: var(--amber-soft);
        border: 1px solid #EBD2AC;
        padding: 2px 8px;
        border-radius: 3px;
        margin-right: 6px;
        display: inline-block;
        margin-top: 4px;
    }
    .queue-row {
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        padding: 6px 0;
        border-bottom: 1px solid var(--line);
    }
    .queue-score {
        display: inline-block;
        background: var(--line);
        height: 7px;
        width: 70px;
        border-radius: 2px;
        overflow: hidden;
        vertical-align: middle;
        margin-right: 6px;
    }
    .queue-fill { background: var(--green); height: 100%; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    backend_choice = st.radio(
        "Generation backend",
        ["Free (local Ollama)", "Paid (Anthropic Claude API)"],
        index=0,
        help="Free uses your local Ollama model -- no API key needed. Paid uses the Anthropic API and requires credit in your account.",
    )
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=5, value=3)

    st.divider()
    st.caption("**Stack:** sentence-transformers embeddings → ChromaDB → "
               + ("Ollama (local LLM)" if "Free" in backend_choice else "Anthropic Claude API"))
    st.caption("Corpus: 8 Operating Systems note chunks (CPU scheduling, deadlocks, "
               "synchronization, paging, virtual memory, page replacement, file systems, processes/threads).")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner=False)
def load_backend(module_name):
    return importlib.import_module(module_name)


module_name = "rag_free" if "Free" in backend_choice else "rag"
rag = load_backend(module_name)

# ---------------- Header ----------------
st.title("🖥️ OS Notes RAG Assistant")
st.caption("Retrieval-Augmented Generation over Operating Systems concepts — grounded answers, every claim cited.")

if "messages" not in st.session_state:
    st.session_state.messages = []

col_chat, col_queue = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                tags = " ".join(f'<span class="source-tag">{s["id"]}</span>' for s in msg["sources"])
                st.markdown(tags, unsafe_allow_html=True)

    query = st.chat_input("Ask about OS concepts, e.g. 'what is a deadlock'...")

with col_queue:
    st.subheader("Retrieval queue")
    queue_placeholder = st.empty()
    if "last_sources" not in st.session_state:
        queue_placeholder.caption("Retrieved chunks will appear here after you ask a question.")
    else:
        rows_html = ""
        max_dist = max((s["distance"] for s in st.session_state.last_sources), default=1) or 1
        for s in st.session_state.last_sources:
            relevance_pct = max(0, 100 - int((s["distance"] / max_dist) * 100))
            rows_html += (
                f'<div class="queue-row"><b>{s["id"]}</b><br>'
                f'<span class="queue-score"><span class="queue-fill" style="width:{relevance_pct}%"></span></span>'
                f'{s["title"]} (distance={s["distance"]:.3f})</div>'
            )
        queue_placeholder.markdown(rows_html, unsafe_allow_html=True)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = rag.ask(query, k=k)
            st.markdown(result["answer"])
            tags = " ".join(f'<span class="source-tag">{s["id"]}</span>' for s in result["sources"])
            st.markdown(tags, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "sources": result["sources"]
    })
    st.session_state.last_sources = result["sources"]
    st.rerun()
