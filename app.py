"""
WaterOps RAG — Streamlit Interface
------------------------------------
Run:  streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import (
    build_rag_chain,
    build_vectorstore,
    load_documents,
    load_vectorstore,
)

st.set_page_config(
    page_title="WaterOps RAG Assistant",
    page_icon="💧",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💧 WaterOps RAG")
    st.caption("Intelligent Q&A over water infrastructure technical documents")

    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored. It lives only in this session.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    model_choice = st.selectbox(
        "LLM model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini is fast and cost-effective for most queries.",
    )

    top_k = st.slider(
        "Retrieved chunks (top-k)",
        min_value=2,
        max_value=8,
        value=4,
        help="How many document chunks to retrieve per query.",
    )

    st.divider()

    st.subheader("Index documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Build / Rebuild index", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            docs_dir = Path("data/docs")
            docs_dir.mkdir(parents=True, exist_ok=True)

            if uploaded_files:
                for f in uploaded_files:
                    (docs_dir / f.name).write_bytes(f.read())
                st.success(f"Saved {len(uploaded_files)} file(s).")

            with st.spinner("Loading and chunking documents…"):
                docs = load_documents(docs_dir)

            if not docs:
                st.warning(
                    "No documents found. Upload files above or add them to data/docs/."
                )
            else:
                with st.spinner(f"Embedding {len(docs)} document(s)…"):
                    vs = build_vectorstore(docs)
                st.session_state["vectorstore"] = vs
                st.success(f"Index built from {len(docs)} document(s).")

    st.divider()
    st.caption("Built by Elian Wira Sena · [GitHub](https://github.com/elianwirasena/waterops-rag)")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("WaterOps RAG Assistant")
st.markdown(
    "Ask questions about water treatment processes, sensor specifications, "
    "maintenance procedures, or operational guidelines — grounded in your documents."
)

# Example queries
st.subheader("Try an example")
example_cols = st.columns(3)
examples = [
    "What are the safe pressure thresholds for distribution pipes?",
    "Describe the chlorination dosing procedure for treated water.",
    "What sensor calibration steps are required for turbidity meters?",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["prefill"] = ex

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Source chunks"):
                for doc in msg["sources"]:
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    label = Path(src).name + (f" · page {page}" if page else "")
                    st.markdown(f"**{label}**")
                    st.text(doc.page_content[:400] + "…")

# Input
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask a question about your water infrastructure documents…")
if prefill and not question:
    question = prefill

if question:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # Load or retrieve vector store
    if "vectorstore" not in st.session_state:
        try:
            with st.spinner("Loading existing index…"):
                st.session_state["vectorstore"] = load_vectorstore()
        except FileNotFoundError:
            st.error(
                "No document index found. Upload documents and click "
                "**Build / Rebuild index** in the sidebar."
            )
            st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer…"):
            chain, retriever = build_rag_chain(
                vectorstore=st.session_state["vectorstore"],
                model=model_choice,
                top_k=top_k,
            )
            result = {"answer": "", "source_docs": retriever.invoke(question)}

            # Stream response
            response_placeholder = st.empty()
            full_answer = ""
            for chunk in chain.stream(question):
                full_answer += chunk
                response_placeholder.markdown(full_answer + "▌")
            response_placeholder.markdown(full_answer)

        with st.expander("Source chunks"):
            for doc in result["source_docs"]:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                label = Path(src).name + (f" · page {page}" if page else "")
                st.markdown(f"**{label}**")
                st.text(doc.page_content[:400] + "…")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_answer,
        "sources": result["source_docs"],
    })
