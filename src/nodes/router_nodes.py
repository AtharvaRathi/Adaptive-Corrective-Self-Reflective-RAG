"""
src/nodes/router_nodes.py
──────────────────────────
Graph node: route_question

Classifies the incoming query and determines which retrieval strategy to use.
Updates state["search_strategy"] with one of:
  "vectorstore"   → hybrid FAISS+BM25 retrieval
  "web_search"    → Tavily/DuckDuckGo search
  "direct_answer" → LLM answers directly without retrieval
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from src.components.evaluators import build_router
from src.state import GraphState

logger = logging.getLogger(__name__)

# Build router chain once at module load (lazy per process)
_router_chain = None


def _get_router():
    global _router_chain
    if _router_chain is None:
        _router_chain = build_router()
    return _router_chain


def route_question(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: classify the question and set the search strategy.

    Parameters
    ----------
    state : GraphState
        Must contain ``question``.

    Returns
    -------
    dict
        Partial state update with ``search_strategy`` set.
    """
    question = state["question"]
    logger.info("[ROUTER] Classifying question: %s", question)

    try:
        router = _get_router()
        result = router.invoke({"question": question})
        strategy = result.datasource
        logger.info(
            "[ROUTER] → Strategy: %s | Reason: %s",
            strategy,
            result.reasoning,
        )
    except Exception as exc:
        # Graceful degradation: default to vectorstore on evaluator failure
        logger.warning("[ROUTER] Evaluator failed (%s), defaulting to vectorstore.", exc)
        strategy = "vectorstore"

    return {"search_strategy": strategy}
