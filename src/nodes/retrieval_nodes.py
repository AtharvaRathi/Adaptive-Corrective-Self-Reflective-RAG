"""
src/nodes/retrieval_nodes.py
─────────────────────────────
Graph nodes:
  • retrieve_docs  — hybrid FAISS+BM25 ensemble retrieval
  • web_search     — Tavily / DuckDuckGo fallback retrieval
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.components.retrievers import get_ensemble_retriever, get_web_retriever
from src.state import GraphState

logger = logging.getLogger(__name__)


def retrieve_docs(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: run hybrid retrieval (FAISS + BM25 ensemble).

    Uses the current ``state["question"]`` (which may have been rewritten)
    rather than the original question.

    Returns
    -------
    dict
        ``documents`` — list of retrieved LangChain Documents (replaces, not appends).
    """
    question = state["question"]
    logger.info("[RETRIEVAL] Hybrid retrieval for: %s", question)

    retriever = get_ensemble_retriever()
    docs: List[Document] = retriever.invoke(question)

    logger.info("[RETRIEVAL] Retrieved %d documents.", len(docs))
    for i, doc in enumerate(docs, 1):
        logger.debug(
            "  [%d] source=%s | preview=%s...",
            i,
            doc.metadata.get("source", "?"),
            doc.page_content[:80],
        )

    return {"documents": docs}


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: fallback web search via Tavily or DuckDuckGo.

    Returns
    -------
    dict
        ``documents`` — list of Documents from web search results.
    """
    question = state["question"]
    logger.info("[WEB SEARCH] Fetching results for: %s", question)

    retriever = get_web_retriever()

    try:
        docs: List[Document] = retriever.invoke(question)
    except Exception as exc:
        logger.error("[WEB SEARCH] Retrieval failed: %s", exc)
        docs = [
            Document(
                page_content=(
                    f"Web search failed for query: '{question}'. "
                    "Please try rephrasing your question or check your API keys."
                ),
                metadata={"source": "error", "query": question},
            )
        ]

    logger.info("[WEB SEARCH] Retrieved %d document(s).", len(docs))
    return {"documents": docs}
