# 💧 WaterOps RAG

**Retrieval-Augmented Generation (RAG) pipeline for water infrastructure technical documents**

An end-to-end AI assistant that lets operations teams query their water treatment and distribution manuals, sensor specifications, and maintenance procedures using natural language — with full source citation and no hallucination on out-of-scope questions.

Built with LangChain, OpenAI, and FAISS. Deployable locally or on any cloud VM.

---

## Demo

![WaterOps RAG demo screenshot](assets/demo_screenshot.png)

**Example queries:**
> *"What are the safe pressure thresholds for the distribution network?"*
> *"Describe the chlorination dosing procedure step by step."*
> *"What sensor calibration steps are required for turbidity meters?"*

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  RAG Pipeline (LangChain LCEL)       │
│                                                      │
│  Query ──► FAISS Retriever ──► Top-K Chunks          │
│                                     │                │
│                                     ▼                │
│              ChatPromptTemplate (system + context)   │
│                                     │                │
│                                     ▼                │
│                          GPT-4o-mini (OpenAI)        │
│                                     │                │
│                                     ▼                │
│                     Answer + Source Citations        │
└─────────────────────────────────────────────────────┘
```

**Key design decisions:**
- `RecursiveCharacterTextSplitter` with 800-token chunks and 150-token overlap — tuned for technical manuals with dense structured content
- `text-embedding-3-small` for cost-efficient semantic embeddings
- FAISS index persisted to disk — no re-embedding on restart
- Temperature = 0 for deterministic, reproducible answers
- Source citations injected via metadata at retrieval time

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangChain (LCEL) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (local) |
| Document Loaders | LangChain PDF + Text loaders |
| UI | Streamlit |
| Testing | pytest |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/elianwirasena/waterops-rag.git
cd waterops-rag
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI key:
# OPENAI_API_KEY=sk-...
```

### 3. Add your documents

Drop PDF or `.txt` files into `data/docs/`. Sample water infrastructure documents are included to get started immediately.

### 4. Build the vector index

```python
from src.rag_pipeline import load_documents, build_vectorstore
docs = load_documents()
build_vectorstore(docs)
```

Or use the Streamlit UI (see below) which handles this with a single button click.

### 5. Launch the UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Use the Python API directly

```python
from src.rag_pipeline import build_rag_chain, query

chain, retriever = build_rag_chain()

result = query(
    "What is the maximum allowable pressure before emergency shutdown?",
    chain=chain,
    retriever=retriever,
)

print(result["answer"])
# → "According to WOM-CH4-2023 (pressure_management_manual.txt), the emergency
#    shutdown threshold is 7.5 bar..."

for doc in result["source_docs"]:
    print(doc.metadata["source"], "—", doc.page_content[:100])
```

---

## Project Structure

```
waterops-rag/
├── app.py                        # Streamlit web UI
├── src/
│   └── rag_pipeline.py           # Core RAG pipeline (load → embed → retrieve → generate)
├── data/
│   ├── docs/                     # Source documents (PDF / TXT)
│   │   ├── pressure_management_manual.txt
│   │   └── treatment_procedures.txt
│   └── vectorstore/              # Persisted FAISS index (auto-created)
├── tests/
│   └── test_pipeline.py          # Unit tests (pytest)
├── notebooks/
│   └── demo_and_eval.ipynb       # Interactive demo + retrieval evaluation
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover document loading, chunking behaviour, context formatting, and chain construction — all with mocked LLM and embedding calls so no API key is needed for the test suite.

---

## Extending the Pipeline

**Swap the LLM:**
```python
chain, retriever = build_rag_chain(model="gpt-4o")  # or any OpenAI model
```

**Increase retrieval breadth:**
```python
chain, retriever = build_rag_chain(top_k=8)
```

**Add a new document type:**
Extend the `load_documents()` function in `src/rag_pipeline.py` with any LangChain document loader — the rest of the pipeline is loader-agnostic.

**Replace FAISS with a cloud vector DB (e.g. Pinecone, Weaviate):**
Swap `FAISS.from_documents(...)` for the relevant LangChain vector store integration — the LCEL chain is store-agnostic.

---

## Background

This project was built to demonstrate practical GenAI engineering skills — specifically:
- LLM integration via API (OpenAI)
- RAG pipeline design and implementation
- Vector store construction and similarity retrieval
- LangChain LCEL chain composition
- Production-minded code structure (typed functions, docstrings, unit tests)

The water infrastructure domain was chosen deliberately: it mirrors real operational AI use cases in utilities and infrastructure sectors, where AI assistants over technical documentation provide direct business value.

---

## Author

**Elian Wira Sena**
AI Engineer · Hamburg, Germany

---

## License

MIT License — see [LICENSE](LICENSE) for details.
