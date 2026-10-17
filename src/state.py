"""
src/state.py
────────────
Defines the shared LangGraph state (GraphState) used by every node in the
Adaptive Self-Reflective RAG pipeline.

All nodes receive this state dict and return a dict of partial updates.
LangGraph merges the partial updates into the running state automatically.
"""
from __future__ import annotations

from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document


def _append_docs(existing: List[Document], new: List[Document]) -> List[Document]:
    """Custom reducer: accumulate documents across nodes rather than overwriting."""
    return existing + new


class GraphState(TypedDict, total=False):
    """
    Shared state propagated through every node of the RAG graph.

    Fields
    ------
    question : str
        The *current* query string (may be rewritten by transform_query).
    original_question : str
        The user's original, unmodified question. Used for faithfulness grading.
    generation : str
        The LLM-generated answer text.
    documents : List[Document]
        Retrieved and/or filtered context documents.
    search_strategy : str
        Which retrieval path was chosen: "vectorstore" | "web_search" | "direct_answer".
    loop_step : int
        CRAG iteration counter. Prevents infinite rewrite→retrieve loops.
    is_grounded : bool
        True when the hallucination grader confirms the answer is fact-grounded.
    is_useful : bool
        True when the faithfulness grader confirms the answer addresses the question.
    citations : List[str]
        Source references extracted from retrieved documents.
    failure_reason : str
        Optional description of why the last retrieval/generation failed.
        Passed to the query rewriter for targeted improvement.
    """

    question: str
    original_question: str
    generation: str
    documents: List[Document]
    search_strategy: str  # "vectorstore" | "web_search" | "direct_answer"
    loop_step: int
    is_grounded: bool
    is_useful: bool
    citations: List[str]
    failure_reason: Optional[str]


def make_initial_state(question: str) -> GraphState:
    """Return a freshly initialised GraphState for a new query."""
    return GraphState(
        question=question,
        original_question=question,
        generation="",
        documents=[],
        search_strategy="vectorstore",
        loop_step=0,
        is_grounded=False,
        is_useful=False,
        citations=[],
        failure_reason=None,
    )
