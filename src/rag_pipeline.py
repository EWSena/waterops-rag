"""
WaterOps RAG Pipeline
---------------------
Core retrieval-augmented generation pipeline for querying
water infrastructure technical documents.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


SYSTEM_PROMPT = """You are WaterOps Assistant, an expert AI system for water treatment
and distribution infrastructure operations.

You answer questions using ONLY the context retrieved from official technical documents,
maintenance manuals, and sensor specifications. If the answer cannot be found in the
context, say so clearly — do not hallucinate.

Always cite which document section your answer is based on.

Context:
{context}
"""

DATA_DIR = Path(__file__).parent.parent / "data" / "docs"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Load all supported documents from the docs directory."""
    docs = []

    # Load .txt files
    txt_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        silent_errors=True,
    )
    docs.extend(txt_loader.load())

    # Load .pdf files
    pdf_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=True,
    )
    docs.extend(pdf_loader.load())

    print(f"[WaterOps RAG] Loaded {len(docs)} document(s) from {data_dir}")
    return docs


def build_vectorstore(
    documents: List[Document],
    persist: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> FAISS:
    """
    Split documents into chunks and embed them into a FAISS vector store.

    Args:
        documents:     List of LangChain Document objects.
        persist:       Whether to save the vector store to disk.
        chunk_size:    Max tokens per chunk (tuned for technical manuals).
        chunk_overlap: Overlap between adjacent chunks to preserve context.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[WaterOps RAG] Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTORSTORE_DIR))
        print(f"[WaterOps RAG] Vector store saved to {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a persisted FAISS vector store from disk."""
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at {VECTORSTORE_DIR}. "
            "Run build_vectorstore() first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks with source metadata for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        ref = f"{Path(source).name}" + (f" (p.{page})" if page else "")
        parts.append(f"[{i}] {ref}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    vectorstore: Optional[FAISS] = None,
    model: str = "gpt-4o-mini",
    top_k: int = 4,
    temperature: float = 0.0,
):
    """
    Build the full RAG chain using LangChain LCEL.

    Pipeline:
        user query
            → retriever (FAISS similarity search, top_k chunks)
            → prompt (system prompt + context + question)
            → LLM (GPT-4o-mini by default)
            → string output

    Args:
        vectorstore: Pre-built FAISS store. Loads from disk if None.
        model:       OpenAI model name.
        top_k:       Number of chunks to retrieve per query.
        temperature: LLM sampling temperature (0 = deterministic).

    Returns:
        A LangChain runnable chain.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model=model, temperature=temperature)

    # LCEL chain with parallel context retrieval + passthrough question
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def query(question: str, chain=None, retriever=None) -> dict:
    """
    Run a question through the RAG pipeline.

    Returns:
        Dict with 'answer' (str) and 'source_docs' (List[Document]).
    """
    if chain is None:
        chain, retriever = build_rag_chain()

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    return {
        "answer": answer,
        "source_docs": source_docs,
    }
