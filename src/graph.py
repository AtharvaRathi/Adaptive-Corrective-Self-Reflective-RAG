"""
src/graph.py
─────────────
Full LangGraph StateGraph assembly for the Adaptive Self-Reflective RAG pipeline.

Graph Structure
───────────────
                      ┌─────────────────┐
            START ───►│  route_question  │
                      └────────┬────────┘
             ┌─────────────────┼──────────────────┐
             ▼                 ▼                  ▼
       retrieve_docs       web_search        [END direct]
       (hybrid FAISS+BM25)    │
             │                │
             ▼                │
      grade_documents          │
      ┌──────┴──────┐          │
      │ valid docs? │          │
      │  yes        │ no       │
      ▼             ▼          │
 generate_answer  transform_query
      │            │(loop < MAX)───► retrieve_docs
      │            │(loop >= MAX)──► web_search
      │
 check_hallucination_and_faithfulness
      ├── grounded & useful ────────► END ✅
      ├── not grounded (hallucinate) ► generate_answer (retry)
      └── grounded but not useful ──► transform_query

Conditional Edge Logic
──────────────────────
• decide_after_routing      : routes to retrieve_docs / web_search / END
• decide_after_grading      : routes to generate_answer / transform_query / web_search
• decide_after_reflection   : routes to END / generate_answer / transform_query
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.nodes.generation_nodes import (
    generate_answer,
    generate_direct_answer,
    transform_query,
)
from src.nodes.grading_nodes import (
    check_hallucination_and_faithfulness,
    grade_documents,
)
from src.nodes.retrieval_nodes import retrieve_docs, web_search
from src.nodes.router_nodes import route_question
from src.state import GraphState

logger = logging.getLogger(__name__)

# ── Conditional Edge Functions ─────────────────────────────────────────────────

def decide_after_routing(
    state: GraphState,
) -> Literal["retrieve_docs", "web_search", "direct_answer"]:
    """
    Branch from route_question based on the chosen search strategy.

    Returns
    -------
    "retrieve_docs"  → hybrid FAISS+BM25 retrieval
    "web_search"     → Tavily/DuckDuckGo search
    "direct_answer"  → LLM answers directly (no retrieval needed)
    """
    strategy = state.get("search_strategy", "vectorstore")
    logger.debug("[EDGE] route_question → strategy=%s", strategy)

    if strategy == "web_search":
        return "web_search"
    elif strategy == "direct_answer":
        return "direct_answer"
    else:
        return "retrieve_docs"


def decide_after_grading(
    state: GraphState,
) -> Literal["generate_answer", "transform_query", "web_search"]:
    """
    Branch from grade_documents:
    - If there are relevant documents → generate an answer.
    - If no relevant docs and retries remain → rewrite query and retry.
    - If max retries reached → escalate to web search.

    Returns
    -------
    "generate_answer" | "transform_query" | "web_search"
    """
    documents = state.get("documents", [])
    loop_step = state.get("loop_step", 0)
    settings = get_settings()
    max_retries = settings.max_retries

    logger.debug(
        "[EDGE] grade_documents → docs=%d, loop_step=%d, max_retries=%d",
        len(documents),
        loop_step,
        max_retries,
    )

    if documents:
        logger.info("[EDGE] ✅ Relevant docs found → generate_answer")
        return "generate_answer"
    elif loop_step < max_retries:
        logger.info(
            "[EDGE] ⚠️  No relevant docs (attempt %d/%d) → transform_query",
            loop_step,
            max_retries,
        )
        return "transform_query"
    else:
        logger.info(
            "[EDGE] 🔴 Max retries reached (%d) → web_search fallback",
            max_retries,
        )
        return "web_search"


def decide_after_reflection(
    state: GraphState,
) -> Literal["__end__", "generate_answer", "transform_query"]:
    """
    Branch from check_hallucination_and_faithfulness:
    - Grounded AND useful → return answer to user (END).
    - NOT grounded (hallucination) → regenerate with same context.
    - Grounded but NOT useful → rewrite query and re-retrieve.

    Returns
    -------
    "__end__" | "generate_answer" | "transform_query"
    """
    is_grounded = state.get("is_grounded", False)
    is_useful = state.get("is_useful", False)
    loop_step = state.get("loop_step", 0)
    settings = get_settings()

    logger.debug(
        "[EDGE] reflection → grounded=%s, useful=%s, loop_step=%d",
        is_grounded,
        is_useful,
        loop_step,
    )

    if is_grounded and is_useful:
        logger.info("[EDGE] ✅ Answer is grounded & useful → END")
        return "__end__"
    elif not is_grounded and loop_step < settings.max_retries:
        logger.info("[EDGE] ⚠️  Hallucination detected → regenerate")
        return "generate_answer"
    elif not is_useful and loop_step < settings.max_retries:
        logger.info("[EDGE] ⚠️  Answer not useful → rewrite query")
        return "transform_query"
    else:
        # Exceeded retries or both fail — return best effort
        logger.warning(
            "[EDGE] 🔴 Reflection limits reached — returning best-effort answer."
        )
        return "__end__"


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the full RAG StateGraph.

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph graph ready to invoke with a GraphState.
    """
    workflow = StateGraph(GraphState)

    # ── Register Nodes ─────────────────────────────────────────────────────────
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("generate_direct_answer", generate_direct_answer)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node(
        "check_hallucination_and_faithfulness",
        check_hallucination_and_faithfulness,
    )

    # ── Entry Point ────────────────────────────────────────────────────────────
    workflow.set_entry_point("route_question")

    # ── Conditional Routing from route_question ────────────────────────────────
    workflow.add_conditional_edges(
        "route_question",
        decide_after_routing,
        {
            "retrieve_docs": "retrieve_docs",
            "web_search": "web_search",
            "direct_answer": "generate_direct_answer",
        },
    )

    # ── Direct answer terminates at END ──────────────────────────────────────
    workflow.add_edge("generate_direct_answer", END)

    # ── After retrieval → grade documents ─────────────────────────────────────
    workflow.add_edge("retrieve_docs", "grade_documents")

    # ── After web search → generate directly (no grading) ────────────────────
    workflow.add_edge("web_search", "generate_answer")

    # ── After grading → branch based on document quality ─────────────────────
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate_answer": "generate_answer",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    # ── CRAG rewrite loop: transform_query → retrieve_docs ───────────────────
    workflow.add_edge("transform_query", "retrieve_docs")

    # ── After generation → self-reflection ───────────────────────────────────
    workflow.add_edge("generate_answer", "check_hallucination_and_faithfulness")

    # ── Self-RAG reflection branching ─────────────────────────────────────────
    workflow.add_conditional_edges(
        "check_hallucination_and_faithfulness",
        decide_after_reflection,
        {
            "__end__": END,
            "generate_answer": "generate_answer",
            "transform_query": "transform_query",
        },
    )

    compiled = workflow.compile()
    logger.info("[GRAPH] StateGraph compiled successfully.")
    return compiled


# ── Singleton ──────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return a cached compiled graph instance."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
