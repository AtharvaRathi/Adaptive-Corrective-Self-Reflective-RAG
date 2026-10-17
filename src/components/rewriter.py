"""
src/components/rewriter.py
───────────────────────────
Query Rewriting chain for Corrective RAG.

When retrieved documents are graded as insufficient, the query rewriter
reformulates the user's question to improve the next retrieval attempt.
It takes into account:
  - The original user question
  - The (optionally) rewritten current question
  - A failure reason describing why the last retrieval was insufficient

The output is an improved query string ready for re-retrieval.

LLM Backend: Groq (llama-3.3-70b-versatile by default)
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings


_REWRITER_SYSTEM = """\
You are an expert query optimizer for a RAG (Retrieval-Augmented Generation) system.

Your task is to rewrite a user's question to improve retrieval performance.

Context about what went wrong with the previous retrieval:
- Failure reason: {failure_reason}

Guidelines for rewriting:
1. Make the query more specific and unambiguous.
2. Use synonyms or alternative phrasings to broaden or focus semantic coverage.
3. Add key terms that might appear in relevant documents.
4. Remove irrelevant or overly broad terms.
5. If the failure was a keyword mismatch, include explicit terminology from the domain.
6. Keep the rewritten query concise (1–2 sentences max).

Return ONLY the rewritten query string. Do not include explanations or preamble.
"""

_REWRITER_HUMAN = """\
Original question: {original_question}
Current question (may have been previously rewritten): {question}

Rewritten query:
"""


def build_query_rewriter():
    """
    Return a chain: ChatPromptTemplate | ChatGroq | StrOutputParser

    Input keys:
        question         (str) — current (possibly rewritten) question
        original_question(str) — the user's original question
        failure_reason   (str) — why the last retrieval failed

    Output: str — the rewritten query
    """
    settings = get_settings()

    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.3,  # Slight creativity to produce varied rewrites
        groq_api_key=settings.groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _REWRITER_SYSTEM),
            ("human", _REWRITER_HUMAN),
        ]
    )

    return prompt | llm | StrOutputParser()
