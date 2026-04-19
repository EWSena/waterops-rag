"""
Unit tests for WaterOps RAG pipeline.
Run: pytest tests/ -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDocumentLoading:
    """Tests for document loading and parsing."""

    def test_load_documents_returns_list(self, tmp_path):
        """load_documents should return a list (empty or populated)."""
        from rag_pipeline import load_documents
        docs = load_documents(tmp_path)
        assert isinstance(docs, list)

    def test_load_documents_reads_txt(self, tmp_path):
        """load_documents should pick up .txt files."""
        sample = tmp_path / "test_doc.txt"
        sample.write_text("Pressure threshold: 6.0 bar\nCalibrate weekly.")

        from rag_pipeline import load_documents
        docs = load_documents(tmp_path)
        assert len(docs) >= 1
        assert any("Pressure threshold" in d.page_content for d in docs)

    def test_load_documents_empty_dir(self, tmp_path):
        """load_documents should return [] for an empty directory."""
        from rag_pipeline import load_documents
        docs = load_documents(tmp_path)
        assert docs == []


class TestChunking:
    """Tests for text splitting and chunking logic."""

    def test_build_vectorstore_chunks_correctly(self, tmp_path):
        """Long documents should produce multiple chunks."""
        from langchain_core.documents import Document
        long_text = "Water pressure sensor reading. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

        embeddings_mock = MagicMock()
        embeddings_mock.embed_documents.return_value = [[0.1] * 1536] * 50
        embeddings_mock.embed_query.return_value = [0.1] * 1536

        with patch("rag_pipeline.FAISS") as mock_faiss, \
             patch("rag_pipeline.OpenAIEmbeddings", return_value=embeddings_mock):
            mock_faiss.from_documents.return_value = MagicMock()
            from rag_pipeline import build_vectorstore
            build_vectorstore(docs, persist=False)
            call_args = mock_faiss.from_documents.call_args
            chunks_passed = call_args[0][0]
            assert len(chunks_passed) > 1, "Long document should be split into chunks"


class TestFormatDocs:
    """Tests for the context formatter."""

    def test_format_docs_includes_source(self):
        from langchain_core.documents import Document
        from rag_pipeline import format_docs
        docs = [
            Document(
                page_content="Chlorine residual must be 0.5 mg/L.",
                metadata={"source": "/data/docs/treatment.txt", "page": 3},
            )
        ]
        result = format_docs(docs)
        assert "treatment.txt" in result
        assert "p.3" in result
        assert "0.5 mg/L" in result

    def test_format_docs_multiple_sources(self):
        from langchain_core.documents import Document
        from rag_pipeline import format_docs
        docs = [
            Document(page_content="Doc A content.", metadata={"source": "a.txt"}),
            Document(page_content="Doc B content.", metadata={"source": "b.txt"}),
        ]
        result = format_docs(docs)
        assert "a.txt" in result
        assert "b.txt" in result
        assert "[1]" in result
        assert "[2]" in result

    def test_format_docs_no_page_metadata(self):
        from langchain_core.documents import Document
        from rag_pipeline import format_docs
        docs = [Document(page_content="No page info.", metadata={"source": "doc.txt"})]
        result = format_docs(docs)
        assert "p." not in result


class TestRAGChain:
    """Integration-style tests for the RAG chain (mocked LLM + embeddings)."""

    def test_build_rag_chain_returns_chain_and_retriever(self):
        """build_rag_chain should return a (chain, retriever) tuple."""
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        with patch("rag_pipeline.OpenAIEmbeddings"), \
             patch("rag_pipeline.ChatOpenAI"):
            from rag_pipeline import build_rag_chain
            chain, retriever = build_rag_chain(vectorstore=mock_vs)
            assert chain is not None
            assert retriever is not None

    def test_query_returns_expected_keys(self):
        """query() should return a dict with 'answer' and 'source_docs'."""
        from langchain_core.documents import Document

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The safe pressure threshold is 6.0 bar."
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Threshold is 6.0 bar.", metadata={"source": "manual.txt"})
        ]

        from rag_pipeline import query
        result = query("What is the safe pressure?", chain=mock_chain, retriever=mock_retriever)
        assert "answer" in result
        assert "source_docs" in result
        assert isinstance(result["source_docs"], list)
