"""
src/nodes/grading_nodes.py
───────────────────────────
Graph nodes:
  • grade_documents                    — filter out irrelevant retrieved docs (CRAG)
  • check_hallucination_and_faithfulness — self-RAG reflection on generated answers
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.components.evaluators import build_doc_grader, build_hallucination_grader, build_answer_grader
from src.state import GraphState

logger = logging.getLogger(__name__)

# Lazy singletons
_doc_grader = None
_hallucination_grader = None
_answer_grader = None


def _get_doc_grader():
    global _doc_grader
    if _doc_grader is None:
        _doc_grader = build_doc_grader()
    return _doc_grader


def _get_hallucination_grader():
    global _hallucination_grader
    if _hallucination_grader is None:
        _hallucination_grader = build_hallucination_grader()
    return _hallucination_grader


def _get_answer_grader():
    global _answer_grader
    if _answer_grader is None:
        _answer_grader = build_answer_grader()
    return _answer_grader


# ── Document Grading (CRAG) ────────────────────────────────────────────────────

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: grade each retrieved document for relevance.

    Filters the document list to include only relevant documents.
    Increments ``loop_step`` to track CRAG retry count.
    Sets ``failure_reason`` if documents are insufficient.

    Returns
    -------
    dict
        ``documents``     — filtered list of relevant documents only
        ``loop_step``     — incremented retry counter
        ``failure_reason``— description if docs were insufficient
    """
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    loop_step: int = state.get("loop_step", 0)

    logger.info("[GRADER] Grading %d documents for relevance...", len(documents))

    grader = _get_doc_grader()
    relevant_docs: List[Document] = []

    for i, doc in enumerate(documents, 1):
        try:
            result = grader.invoke(
                {"document": doc.page_content, "question": question}
            )
            is_relevant = result.binary_score == "yes"
            logger.info(
                "  Doc [%d] source=%s → %s",
                i,
                doc.metadata.get("source", "?"),
                "✅ RELEVANT" if is_relevant else "❌ IRRELEVANT",
            )
            if is_relevant:
                relevant_docs.append(doc)
        except Exception as exc:
            logger.warning("  Doc [%d] grading failed (%s) — keeping doc.", i, exc)
            relevant_docs.append(doc)  # Keep on error to be safe

    n_relevant = len(relevant_docs)
    n_total = len(documents)
    logger.info(
        "[GRADER] %d/%d documents are relevant. Loop step: %d",
        n_relevant,
        n_total,
        loop_step + 1,
    )

    failure_reason = None
    if n_relevant == 0:
        failure_reason = (
            f"All {n_total} retrieved documents were graded as irrelevant "
            f"to the question: '{question}'. "
            "The query may be too ambiguous or use terminology not present in the corpus."
        )
        logger.warning("[GRADER] No relevant documents found. Will trigger rewrite.")

    return {
        "documents": relevant_docs,
        "loop_step": loop_step + 1,
        "failure_reason": failure_reason,
    }


# ── Hallucination & Faithfulness Check (Self-RAG) ─────────────────────────────

def check_hallucination_and_faithfulness(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: evaluate the generated answer for:
      1. Groundedness (no hallucination) — is every claim supported by the docs?
      2. Faithfulness/Utility           — does the answer address the question?

    Updates ``is_grounded`` and ``is_useful`` flags in state.

    Returns
    -------
    dict
        ``is_grounded`` — bool
        ``is_useful``   — bool
        ``failure_reason`` — description if checks failed
    """
    generation: str = state.get("generation", "")
    documents: List[Document] = state.get("documents", [])
    question: str = state.get("original_question", state.get("question", ""))

    logger.info("[REFLECTION] Evaluating answer for hallucination and faithfulness...")

    # Format documents for grader prompt
    docs_text = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )

    # ── Hallucination Check ────────────────────────────────────────────────────
    is_grounded = True
    try:
        hal_grader = _get_hallucination_grader()
        hal_result = hal_grader.invoke(
            {"documents": docs_text, "generation": generation}
        )
        is_grounded = hal_result.binary_score == "yes"
        logger.info(
            "[REFLECTION] Hallucination check → %s",
            "✅ GROUNDED" if is_grounded else "❌ HALLUCINATION DETECTED",
        )
    except Exception as exc:
        logger.warning("[REFLECTION] Hallucination grader failed (%s) — assuming grounded.", exc)

    # ── Faithfulness / Utility Check ───────────────────────────────────────────
    is_useful = True
    try:
        ans_grader = _get_answer_grader()
        ans_result = ans_grader.invoke(
            {"question": question, "generation": generation}
        )
        is_useful = ans_result.binary_score == "yes"
        logger.info(
            "[REFLECTION] Faithfulness check → %s",
            "✅ USEFUL" if is_useful else "❌ NOT USEFUL",
        )
    except Exception as exc:
        logger.warning("[REFLECTION] Answer grader failed (%s) — assuming useful.", exc)

    failure_reason = None
    if not is_grounded:
        failure_reason = (
            "The generated answer contains claims not supported by the retrieved documents. "
            "Regenerating with stricter grounding instructions."
        )
    elif not is_useful:
        failure_reason = (
            "The generated answer does not adequately address the user's question. "
            "Rewriting the query to retrieve more targeted context."
        )

    return {
        "is_grounded": is_grounded,
        "is_useful": is_useful,
        "failure_reason": failure_reason,
    }
