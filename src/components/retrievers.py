"""
src/components/retrievers.py
────────────────────────────
Hybrid retrieval stack:
  • DocumentIngester  — loads sample_docs.json and wraps each entry as a LangChain Document
  • build_faiss_retriever   — dense semantic search via FAISS + HuggingFace embeddings (local)
  • build_bm25_retriever    — sparse lexical search via rank-bm25
  • build_ensemble_retriever — fuses dense + sparse with configurable weights
  • build_web_retriever     — Tavily API with DuckDuckGo fallback
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.retrievers import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        from collections import defaultdict
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        class EnsembleRetriever(BaseRetriever):  # type: ignore[no-redef]
            """Self-contained Reciprocal Rank Fusion Ensemble Retriever fallback."""
            retrievers: list
            weights: list
            c: int = 60

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                rrf_score = defaultdict(float)
                doc_map = {}
                for retriever, weight in zip(self.retrievers, self.weights):
                    docs = retriever.invoke(query)
                    for rank, doc in enumerate(docs):
                        key = doc.page_content
                        doc_map[key] = doc
                        rrf_score[key] += weight * (1.0 / (rank + self.c))
                sorted_keys = sorted(rrf_score.keys(), key=lambda k: rrf_score[k], reverse=True)
                return [doc_map[k] for k in sorted_keys]

from src.config import get_settings

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SAMPLE_DOCS_PATH = _DATA_DIR / "sample_docs.json"
_FAISS_INDEX_PATH = _DATA_DIR / "faiss_index"


# ── Document Ingestion ─────────────────────────────────────────────────────────

class DocumentIngester:
    """
    Loads documents from data/sample_docs.json and converts them into
    LangChain Document objects with rich metadata.
    """

    def __init__(self, docs_path: Path = _SAMPLE_DOCS_PATH) -> None:
        self.docs_path = docs_path

    def load(self) -> List[Document]:
        """Read the JSON file and return a list of LangChain Documents."""
        if not self.docs_path.exists():
            raise FileNotFoundError(
                f"Sample docs not found at {self.docs_path}. "
                "Ensure data/sample_docs.json exists."
            )
        with open(self.docs_path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)

        documents: List[Document] = []
        for entry in raw:
            doc = Document(
                page_content=entry["content"],
                metadata={
                    "id": entry.get("id", ""),
                    "source": entry.get("source", "unknown"),
                    "title": entry.get("title", ""),
                },
            )
            documents.append(doc)
        return documents


# ── FAISS Dense Retriever ──────────────────────────────────────────────────────

def build_faiss_retriever(
    documents: List[Document],
    persist_path: Optional[Path] = _FAISS_INDEX_PATH,
) -> BaseRetriever:
    """
    Build (or reload from disk) a FAISS vector store and return a retriever.
    Uses HuggingFaceEmbeddings (sentence-transformers) — no API key required.

    Parameters
    ----------
    documents:
        The corpus to embed and index.
    persist_path:
        Directory to save/load the FAISS index. Pass None to skip persistence.
    """
    settings = get_settings()

    # HuggingFace embeddings run locally — no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # When switching embedding models the stored index is incompatible —
    # delete the persisted directory to force a fresh rebuild.
    if persist_path and persist_path.exists():
        # Reload existing index to avoid re-embedding on subsequent runs
        vectorstore = FAISS.load_local(
            str(persist_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        if persist_path:
            persist_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(persist_path))

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_retrieval},
    )


# ── BM25 Sparse Retriever ──────────────────────────────────────────────────────

def build_bm25_retriever(documents: List[Document]) -> BM25Retriever:
    """
    Build a BM25 sparse retriever from the document corpus.
    BM25 excels at exact keyword matching (e.g., named entities, IDs).
    """
    settings = get_settings()
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = settings.top_k_retrieval
    return retriever


# ── Ensemble (Hybrid) Retriever ───────────────────────────────────────────────

def build_ensemble_retriever(
    documents: List[Document],
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> EnsembleRetriever:
    """
    Combine FAISS (dense) and BM25 (sparse) into a single EnsembleRetriever.

    Parameters
    ----------
    documents:
        Corpus to index.
    dense_weight, sparse_weight:
        Override weights from settings (must sum to 1.0).
    """
    settings = get_settings()
    weights = settings.retriever_weights  # normalised [dense, sparse]
    if dense_weight is not None and sparse_weight is not None:
        total = dense_weight + sparse_weight
        weights = [dense_weight / total, sparse_weight / total]

    faiss_retriever = build_faiss_retriever(documents)
    bm25_retriever = build_bm25_retriever(documents)

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
    )


# ── Web Search Retriever ───────────────────────────────────────────────────────

def build_web_retriever() -> BaseRetriever:
    """
    Return a web-search retriever.
    Prefers Tavily if TAVILY_API_KEY is set; falls back to DuckDuckGo.
    """
    settings = get_settings()

    if settings.tavily_api_key:
        try:
            from langchain_community.retrievers import TavilySearchAPIRetriever

            return TavilySearchAPIRetriever(
                k=settings.top_k_retrieval,
                api_key=settings.tavily_api_key,
            )
        except ImportError:
            pass  # fall through to DuckDuckGo

    # DuckDuckGo fallback (no API key required)
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    class _DDGRetriever(BaseRetriever):
        """Minimal DuckDuckGo wrapper that satisfies the BaseRetriever interface."""

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
            tool = DuckDuckGoSearchRun()
            raw = tool.run(query)
            return [
                Document(
                    page_content=raw,
                    metadata={"source": "DuckDuckGo", "query": query},
                )
            ]

    return _DDGRetriever()


# ── Singleton cache so the expensive index is only built once ─────────────────

_ensemble_retriever_cache: Optional[EnsembleRetriever] = None


def get_ensemble_retriever() -> EnsembleRetriever:
    """Return a cached ensemble retriever, building it on first call."""
    global _ensemble_retriever_cache
    if _ensemble_retriever_cache is None:
        ingester = DocumentIngester()
        docs = ingester.load()
        _ensemble_retriever_cache = build_ensemble_retriever(docs)
    return _ensemble_retriever_cache


_web_retriever_cache: Optional[BaseRetriever] = None


def get_web_retriever() -> BaseRetriever:
    """Return a cached web retriever."""
    global _web_retriever_cache
    if _web_retriever_cache is None:
        _web_retriever_cache = build_web_retriever()
    return _web_retriever_cache
