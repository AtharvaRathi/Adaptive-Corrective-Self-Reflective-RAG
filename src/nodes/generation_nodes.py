"""
src/nodes/generation_nodes.py
──────────────────────────────
Graph nodes:
  • transform_query — rewrite the question when retrieval failed
  • generate_answer — produce a cited, grounded response from context
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.components.rewriter import build_query_rewriter
from src.config import get_settings
from src.state import GraphState

logger = logging.getLogger(__name__)

# Lazy singletons
_query_rewriter = None
_generation_chain = None


def _get_query_rewriter():
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = build_query_rewriter()
    return _query_rewriter


def _build_generation_chain():
    """Construct the RAG answer generation chain."""
    settings = get_settings()

    system_prompt = """\
You are a knowledgeable assistant that provides accurate, well-cited answers.

IMPORTANT RULES:
1. Base your answer EXCLUSIVELY on the provided context documents.
2. Cite your sources inline using [Source: <source_name>] after each claim.
3. If the context does not contain enough information, explicitly state what is missing.
4. Do not introduce external knowledge not present in the context.
5. Structure your answer clearly with logical flow.
6. End your answer with a brief "Sources:" list summarizing the documents used.

Context Documents:
{context}
"""

    human_prompt = "Question: {question}\n\nAnswer (with inline citations):"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.1,  # Low temp for factual, grounded answers
        groq_api_key=settings.groq_api_key,
    )

    return prompt | llm | StrOutputParser()


def _get_generation_chain():
    global _generation_chain
    if _generation_chain is None:
        _generation_chain = _build_generation_chain()
    return _generation_chain


def _format_context(documents: List[Document]) -> str:
    """Format retrieved documents into a numbered context block."""
    if not documents:
        return "No context documents available."
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        header = f"[{i}] {title} (source: {source})" if title else f"[{i}] source: {source}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_citations(generation: str, documents: List[Document]) -> List[str]:
    """
    Extract unique citation strings from the generated text.
    Looks for patterns like [Source: xxx] and also builds a list from doc metadata.
    """
    citations: List[str] = []

    # Extract inline citations from generation text
    inline_pattern = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)
    for match in inline_pattern.finditer(generation):
        cite = match.group(1).strip()
        if cite not in citations:
            citations.append(cite)

    # If no inline citations found, fall back to document sources
    if not citations:
        seen = set()
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")
            cite = f"{title} ({source})" if title else source
            if cite not in seen:
                citations.append(cite)
                seen.add(cite)

    return citations


# ── Node Functions ─────────────────────────────────────────────────────────────

def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: rewrite the current query to improve retrieval.

    Uses the failure_reason from document grading or self-reflection
    to produce a more targeted search query.

    Returns
    -------
    dict
        ``question`` — the rewritten query string
    """
    question = state["question"]
    original_question = state.get("original_question", question)
    failure_reason = state.get("failure_reason") or "Retrieved documents were not relevant."

    logger.info("[REWRITER] Rewriting query...")
    logger.info("  Original: %s", original_question)
    logger.info("  Current:  %s", question)
    logger.info("  Reason:   %s", failure_reason)

    try:
        rewriter = _get_query_rewriter()
        new_question = rewriter.invoke(
            {
                "question": question,
                "original_question": original_question,
                "failure_reason": failure_reason,
            }
        )
        new_question = new_question.strip().strip('"').strip("'")
        logger.info("[REWRITER] → New query: %s", new_question)
    except Exception as exc:
        logger.error("[REWRITER] Rewrite failed (%s) — keeping original question.", exc)
        new_question = question

    return {"question": new_question}


def generate_answer(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: generate a cited answer from the retrieved context.

    Formats the context, calls the generation chain, extracts citations,
    and updates state with the generation and citation list.

    Returns
    -------
    dict
        ``generation`` — the LLM-generated answer string
        ``citations``  — list of source citation strings
    """
    question = state.get("original_question", state["question"])
    documents: List[Document] = state.get("documents", [])

    logger.info(
        "[GENERATOR] Generating answer from %d context document(s)...", len(documents)
    )

    context = _format_context(documents)

    try:
        chain = _get_generation_chain()
        generation = chain.invoke({"context": context, "question": question})
        logger.info("[GENERATOR] Answer generated (%d chars).", len(generation))
    except Exception as exc:
        logger.error("[GENERATOR] Generation failed: %s", exc)
        generation = (
            f"I encountered an error while generating an answer: {exc}. "
            "Please check your API key and try again."
        )

    citations = _extract_citations(generation, documents)
    logger.info("[GENERATOR] Extracted %d citation(s).", len(citations))

    if not generation or not generation.strip():
        generation = "I could not find sufficient information in the provided context to answer your question."

    return {"generation": generation, "citations": citations}


def generate_direct_answer(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node: generate a direct answer without document retrieval
    for simple factual, logical, mathematical, or conversational questions.

    Returns
    -------
    dict
        ``generation`` — the LLM-generated answer string
        ``citations``  — source indicator
    """
    question = state.get("original_question", state["question"])
    logger.info("[GENERATOR] Generating direct answer without retrieval for: %s", question)
    settings = get_settings()

    system_prompt = (
        "You are a helpful, knowledgeable assistant. "
        "Answer the user's question directly, accurately, and concisely."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.1,
        groq_api_key=settings.groq_api_key,
    )
    chain = prompt | llm | StrOutputParser()

    try:
        generation = chain.invoke({"question": question})
        logger.info("[GENERATOR] Direct answer generated (%d chars).", len(generation))
    except Exception as exc:
        logger.error("[GENERATOR] Direct generation failed: %s", exc)
        generation = f"I encountered an error generating a direct answer: {exc}"

    if not generation or not generation.strip():
        generation = "I was unable to formulate a response to this question."

    return {
        "generation": generation,
        "citations": ["Direct LLM Response (no external retrieval needed)"],
    }
