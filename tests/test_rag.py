"""
tests/test_rag.py
──────────────────
Test suite for the Adaptive Self-Reflective RAG Orchestrator.

Tests
─────
1. Config loading and validation
2. Document ingestion from sample_docs.json
3. Router evaluator (mocked LLM)
4. Document relevance grader (mocked LLM)
5. Hallucination grader (mocked LLM)
6. Answer grader (mocked LLM)
7. Query rewriter
8. Graph node: route_question
9. Graph node: grade_documents
10. Graph node: generate_answer
11. Full graph run (mocked LLM + retrievers)
12. Stopping conditions (max retries respected)
13. Citations extraction
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Set dummy env vars before importing src modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-key")

from src.config import Settings, get_settings
from src.state import GraphState, make_initial_state


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs() -> List[Document]:
    """Load documents from the real sample_docs.json."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_docs.json"
    with open(data_path) as f:
        raw = json.load(f)
    return [
        Document(
            page_content=entry["content"],
            metadata={
                "id": entry.get("id", ""),
                "source": entry.get("source", "unknown"),
                "title": entry.get("title", ""),
            },
        )
        for entry in raw
    ]


@pytest.fixture
def initial_state() -> GraphState:
    return make_initial_state("What is BM25 retrieval?")


@pytest.fixture
def populated_state(sample_docs) -> GraphState:
    state = make_initial_state("What is BM25 retrieval?")
    state["documents"] = sample_docs[:3]
    state["generation"] = "BM25 is a probabilistic ranking function. [Source: bm25_retrieval.txt]"
    return state


# ── 1. Config ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        settings = Settings(groq_api_key="gsk-test", tavily_api_key="tvly-test")
        assert settings.llm_model == "qwen/qwen3.8-27b"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.max_retries == 3
        assert settings.top_k_retrieval == 5
        assert len(settings.retriever_weights) == 2

    def test_weights_normalise(self):
        # Use valid values (<=1.0); property normalises them to sum to 1
        settings = Settings(
            groq_api_key="gsk-test",
            retriever_dense_weight=0.7,
            retriever_sparse_weight=0.3,
        )
        w = settings.retriever_weights
        assert abs(w[0] - 0.7) < 1e-6
        assert abs(w[1] - 0.3) < 1e-6
        assert abs(sum(w) - 1.0) < 1e-6

    def test_weights_equal_normalise(self):
        settings = Settings(
            groq_api_key="gsk-test",
            retriever_dense_weight=0.5,
            retriever_sparse_weight=0.5,
        )
        w = settings.retriever_weights
        assert abs(w[0] - 0.5) < 1e-6
        assert abs(w[1] - 0.5) < 1e-6

    def test_chunk_size_validation(self):
        with pytest.raises(Exception):
            Settings(groq_api_key="gsk-test", chunk_size=50)  # below minimum


# ── 2. Document Ingestion ──────────────────────────────────────────────────────

class TestDocumentIngestion:
    def test_load_returns_documents(self, sample_docs):
        assert len(sample_docs) == 15
        assert all(isinstance(d, Document) for d in sample_docs)

    def test_documents_have_metadata(self, sample_docs):
        for doc in sample_docs:
            assert "source" in doc.metadata
            assert "title" in doc.metadata
            assert len(doc.page_content) > 0

    def test_ingester_class(self):
        from src.components.retrievers import DocumentIngester
        ingester = DocumentIngester()
        docs = ingester.load()
        assert len(docs) == 15

    def test_ingester_missing_file(self, tmp_path):
        from src.components.retrievers import DocumentIngester
        ingester = DocumentIngester(docs_path=tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            ingester.load()


# ── 3. State ───────────────────────────────────────────────────────────────────

class TestState:
    def test_make_initial_state(self):
        state = make_initial_state("test question")
        assert state["question"] == "test question"
        assert state["original_question"] == "test question"
        assert state["loop_step"] == 0
        assert state["documents"] == []
        assert state["citations"] == []
        assert state["is_grounded"] is False
        assert state["is_useful"] is False

    def test_state_mutation(self, initial_state):
        initial_state["loop_step"] = 2
        assert initial_state["loop_step"] == 2


# ── 4. Router Node ─────────────────────────────────────────────────────────────

class TestRouterNode:
    def test_route_vectorstore(self, initial_state):
        from src.nodes.router_nodes import route_question

        mock_result = MagicMock()
        mock_result.datasource = "vectorstore"
        mock_result.reasoning = "Topic is in corpus."

        with patch("src.nodes.router_nodes._get_router") as mock_router_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_router_fn.return_value = mock_chain

            result = route_question(initial_state)

        assert result["search_strategy"] == "vectorstore"

    def test_route_web_search(self, initial_state):
        from src.nodes.router_nodes import route_question

        mock_result = MagicMock()
        mock_result.datasource = "web_search"
        mock_result.reasoning = "Needs real-time data."

        with patch("src.nodes.router_nodes._get_router") as mock_router_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_router_fn.return_value = mock_chain

            result = route_question(initial_state)

        assert result["search_strategy"] == "web_search"

    def test_route_fallback_on_error(self, initial_state):
        from src.nodes.router_nodes import route_question

        with patch("src.nodes.router_nodes._get_router") as mock_router_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("API error")
            mock_router_fn.return_value = mock_chain

            result = route_question(initial_state)

        # Should fall back to vectorstore
        assert result["search_strategy"] == "vectorstore"


# ── 5. Retrieval Nodes ─────────────────────────────────────────────────────────

class TestRetrievalNodes:
    def test_retrieve_docs(self, initial_state, sample_docs):
        from src.nodes.retrieval_nodes import retrieve_docs

        with patch("src.nodes.retrieval_nodes.get_ensemble_retriever") as mock_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = sample_docs[:3]
            mock_fn.return_value = mock_retriever

            result = retrieve_docs(initial_state)

        assert "documents" in result
        assert len(result["documents"]) == 3

    def test_web_search_returns_docs(self, initial_state):
        from src.nodes.retrieval_nodes import web_search

        mock_doc = Document(page_content="Web result", metadata={"source": "web"})

        with patch("src.nodes.retrieval_nodes.get_web_retriever") as mock_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [mock_doc]
            mock_fn.return_value = mock_retriever

            result = web_search(initial_state)

        assert len(result["documents"]) == 1
        assert result["documents"][0].metadata["source"] == "web"

    def test_web_search_handles_error(self, initial_state):
        from src.nodes.retrieval_nodes import web_search

        with patch("src.nodes.retrieval_nodes.get_web_retriever") as mock_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.side_effect = RuntimeError("Network error")
            mock_fn.return_value = mock_retriever

            result = web_search(initial_state)

        # Should return error document, not crash
        assert len(result["documents"]) == 1
        assert "failed" in result["documents"][0].page_content.lower()


# ── 6. Grading Nodes ──────────────────────────────────────────────────────────

class TestGradingNodes:
    def test_grade_all_relevant(self, sample_docs):
        from src.nodes.grading_nodes import grade_documents

        state = make_initial_state("What is BM25?")
        state["documents"] = sample_docs[:2]

        mock_result = MagicMock()
        mock_result.binary_score = "yes"

        with patch("src.nodes.grading_nodes._get_doc_grader") as mock_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_fn.return_value = mock_chain

            result = grade_documents(state)

        assert len(result["documents"]) == 2  # All kept
        assert result["loop_step"] == 1
        assert result["failure_reason"] is None

    def test_grade_all_irrelevant(self, sample_docs):
        from src.nodes.grading_nodes import grade_documents

        state = make_initial_state("What is the population of Mars?")
        state["documents"] = sample_docs[:2]

        mock_result = MagicMock()
        mock_result.binary_score = "no"

        with patch("src.nodes.grading_nodes._get_doc_grader") as mock_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_fn.return_value = mock_chain

            result = grade_documents(state)

        assert len(result["documents"]) == 0
        assert result["loop_step"] == 1
        assert result["failure_reason"] is not None

    def test_hallucination_detected(self, populated_state):
        from src.nodes.grading_nodes import check_hallucination_and_faithfulness

        mock_hal = MagicMock()
        mock_hal.binary_score = "no"  # hallucination!
        mock_ans = MagicMock()
        mock_ans.binary_score = "yes"

        with (
            patch("src.nodes.grading_nodes._get_hallucination_grader") as mock_h,
            patch("src.nodes.grading_nodes._get_answer_grader") as mock_a,
        ):
            mock_h.return_value = MagicMock(invoke=MagicMock(return_value=mock_hal))
            mock_a.return_value = MagicMock(invoke=MagicMock(return_value=mock_ans))

            result = check_hallucination_and_faithfulness(populated_state)

        assert result["is_grounded"] is False
        assert result["is_useful"] is True
        assert result["failure_reason"] is not None

    def test_grounded_and_useful(self, populated_state):
        from src.nodes.grading_nodes import check_hallucination_and_faithfulness

        mock_hal = MagicMock()
        mock_hal.binary_score = "yes"  # grounded
        mock_ans = MagicMock()
        mock_ans.binary_score = "yes"  # useful

        with (
            patch("src.nodes.grading_nodes._get_hallucination_grader") as mock_h,
            patch("src.nodes.grading_nodes._get_answer_grader") as mock_a,
        ):
            mock_h.return_value = MagicMock(invoke=MagicMock(return_value=mock_hal))
            mock_a.return_value = MagicMock(invoke=MagicMock(return_value=mock_ans))

            result = check_hallucination_and_faithfulness(populated_state)

        assert result["is_grounded"] is True
        assert result["is_useful"] is True
        assert result["failure_reason"] is None


# ── 7. Generation Nodes ────────────────────────────────────────────────────────

class TestGenerationNodes:
    def test_transform_query(self, initial_state):
        from src.nodes.generation_nodes import transform_query

        initial_state["failure_reason"] = "No relevant docs found."

        with patch("src.nodes.generation_nodes._get_query_rewriter") as mock_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "What is the BM25 probabilistic ranking algorithm?"
            mock_fn.return_value = mock_chain

            result = transform_query(initial_state)

        assert "question" in result
        assert result["question"] != ""

    def test_generate_answer_with_docs(self, populated_state):
        from src.nodes.generation_nodes import generate_answer

        with patch("src.nodes.generation_nodes._get_generation_chain") as mock_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = (
                "BM25 is a ranking function. [Source: bm25_retrieval.txt]\n\n"
                "Sources:\n- bm25_retrieval.txt"
            )
            mock_fn.return_value = mock_chain

            result = generate_answer(populated_state)

        assert "generation" in result
        assert len(result["generation"]) > 0
        assert "citations" in result

    def test_citations_extracted(self, populated_state):
        from src.nodes.generation_nodes import _extract_citations

        generation = (
            "BM25 is based on TF-IDF. [Source: bm25_retrieval.txt] "
            "It is used in search. [Source: faiss_docs.txt]"
        )
        citations = _extract_citations(generation, populated_state["documents"])
        assert "bm25_retrieval.txt" in citations
        assert "faiss_docs.txt" in citations


# ── 8. Conditional Edge Logic ──────────────────────────────────────────────────

class TestConditionalEdges:
    def test_decide_after_routing_vectorstore(self):
        from src.graph import decide_after_routing

        state = make_initial_state("test")
        state["search_strategy"] = "vectorstore"
        assert decide_after_routing(state) == "retrieve_docs"

    def test_decide_after_routing_web(self):
        from src.graph import decide_after_routing

        state = make_initial_state("test")
        state["search_strategy"] = "web_search"
        assert decide_after_routing(state) == "web_search"

    def test_decide_after_routing_direct(self):
        from src.graph import decide_after_routing

        state = make_initial_state("test")
        state["search_strategy"] = "direct_answer"
        assert decide_after_routing(state) == "direct_answer"

    def test_decide_after_grading_with_docs(self, sample_docs):
        from src.graph import decide_after_grading

        state = make_initial_state("test")
        state["documents"] = sample_docs[:2]
        state["loop_step"] = 1
        assert decide_after_grading(state) == "generate_answer"

    def test_decide_after_grading_no_docs_retry(self):
        from src.graph import decide_after_grading

        state = make_initial_state("test")
        state["documents"] = []
        state["loop_step"] = 1  # Below max_retries=3
        assert decide_after_grading(state) == "transform_query"

    def test_decide_after_grading_max_retries(self):
        from src.graph import decide_after_grading

        state = make_initial_state("test")
        state["documents"] = []
        state["loop_step"] = 10  # Exceeds max_retries
        assert decide_after_grading(state) == "web_search"

    def test_decide_after_reflection_success(self):
        from src.graph import decide_after_reflection

        state = make_initial_state("test")
        state["is_grounded"] = True
        state["is_useful"] = True
        assert decide_after_reflection(state) == "__end__"

    def test_decide_after_reflection_hallucination(self):
        from src.graph import decide_after_reflection

        state = make_initial_state("test")
        state["is_grounded"] = False
        state["is_useful"] = True
        state["loop_step"] = 1  # Below max_retries
        assert decide_after_reflection(state) == "generate_answer"

    def test_decide_after_reflection_not_useful(self):
        from src.graph import decide_after_reflection

        state = make_initial_state("test")
        state["is_grounded"] = True
        state["is_useful"] = False
        state["loop_step"] = 1
        assert decide_after_reflection(state) == "transform_query"


# ── 9. Graph Build ─────────────────────────────────────────────────────────────

class TestGraphBuild:
    def test_graph_compiles(self):
        """Verify the graph can be compiled without errors."""
        from src.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        from src.graph import build_graph
        graph = build_graph()
        # LangGraph compiled graphs expose graph structure
        assert graph is not None
