"""
src/components/evaluators.py
─────────────────────────────
Four structured-output LLM evaluator chains used throughout the RAG pipeline.

Each evaluator uses Pydantic schemas + `with_structured_output` for deterministic,
parsable binary responses — no regex or freeform parsing needed.

Evaluators
──────────
1. build_router()              → RouteQuery:       route = "vectorstore" | "web_search" | "direct_answer"
2. build_doc_grader()          → GradeDocuments:   binary_score = "yes" | "no"
3. build_hallucination_grader()→ GradeHallucinations: binary_score = "yes" | "no"
4. build_answer_grader()       → GradeAnswer:      binary_score = "yes" | "no"

LLM Backend: Groq (llama-3.3-70b-versatile by default)
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import get_settings


# ── Pydantic Output Schemas ────────────────────────────────────────────────────

class RouteQuery(BaseModel):
    """
    Router decision: which retrieval strategy to use for the incoming query.

    Attributes
    ----------
    datasource:
        - "vectorstore"   → Use hybrid FAISS+BM25 retrieval
        - "web_search"    → Use Tavily/DuckDuckGo for real-time info
        - "direct_answer" → LLM can answer directly without retrieval
    reasoning:
        Brief justification for the routing decision.
    """

    datasource: Literal["vectorstore", "web_search", "direct_answer"] = Field(
        description=(
            "Route the query to 'vectorstore' for document retrieval, "
            "'web_search' for real-time internet search, "
            "or 'direct_answer' if the LLM can answer without retrieval."
        )
    )
    reasoning: str = Field(
        description="One-sentence justification for the routing decision."
    )


class GradeDocuments(BaseModel):
    """
    Relevance grade for a single retrieved document with respect to the query.

    binary_score = "yes"  → document is relevant and should be used in generation
    binary_score = "no"   → document is off-topic and should be filtered out
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the document is relevant to the question, 'no' otherwise."
        )
    )


class GradeHallucinations(BaseModel):
    """
    Hallucination check: is the generated answer grounded in the retrieved context?

    binary_score = "yes" → answer is fully supported by the documents (no hallucination)
    binary_score = "no"  → answer contains claims not supported by context
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the generation is grounded in the provided facts, "
            "'no' if it contains hallucinated or unsupported claims."
        )
    )


class GradeAnswer(BaseModel):
    """
    Faithfulness / utility check: does the answer address the user's question?

    binary_score = "yes" → answer is useful and resolves the question
    binary_score = "no"  → answer is off-topic, incomplete, or unhelpful
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer addresses the question, 'no' if it does not."
        )
    )


# ── LLM Factory ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.llm_model,
        temperature=0,
        groq_api_key=settings.groq_api_key,
    )


# ── Evaluator Builders ─────────────────────────────────────────────────────────

def build_router():
    """
    Returns a structured-output chain that classifies an incoming query
    into one of three retrieval strategies.
    """
    system_prompt = (
        "You are an expert at routing user questions to the correct information source.\n\n"
        "Guidelines:\n"
        "- Route to 'vectorstore' for questions about AI, ML, RAG, LangChain, "
        "embeddings, retrieval, transformers, or any topic likely covered in a "
        "curated document corpus.\n"
        "- Route to 'web_search' for questions requiring real-time data, "
        "current events, recent news, or live prices.\n"
        "- Route to 'direct_answer' for simple factual questions, math, "
        "definitions, or anything the LLM can answer confidently without retrieval."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}"),
        ]
    )

    llm = _get_llm().with_structured_output(RouteQuery)
    return prompt | llm


def build_doc_grader():
    """
    Returns a structured-output chain that grades a single document's
    relevance to the user's question.
    """
    system_prompt = (
        "You are a document relevance grader for a RAG system.\n"
        "Given a retrieved document and a user question, determine whether the "
        "document contains information relevant to answering the question.\n\n"
        "Be lenient: even tangential relevance counts as 'yes'.\n"
        "Grade 'no' only if the document is clearly unrelated."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved document:\n\n{document}\n\n"
                "User question: {question}",
            ),
        ]
    )

    llm = _get_llm().with_structured_output(GradeDocuments)
    return prompt | llm


def build_hallucination_grader():
    """
    Returns a structured-output chain that checks whether the generated
    answer is fully grounded in the provided context documents.
    """
    system_prompt = (
        "You are a hallucination detector for a RAG system.\n"
        "Given a set of source documents and a generated answer, determine "
        "whether the answer is fully supported by the documents.\n\n"
        "Grade 'yes' if every factual claim in the answer can be traced to "
        "the provided documents.\n"
        "Grade 'no' if the answer introduces facts, numbers, names, or claims "
        "that are absent from or contradict the provided documents."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Source documents:\n\n{documents}\n\n"
                "Generated answer:\n\n{generation}",
            ),
        ]
    )

    llm = _get_llm().with_structured_output(GradeHallucinations)
    return prompt | llm


def build_answer_grader():
    """
    Returns a structured-output chain that checks whether the generated
    answer actually addresses the user's original question.
    """
    system_prompt = (
        "You are a faithfulness evaluator for a RAG system.\n"
        "Given a user question and a generated answer, determine whether the "
        "answer is useful and directly addresses what was asked.\n\n"
        "Grade 'yes' if the answer resolves the question.\n"
        "Grade 'no' if the answer is off-topic, evasive, or fails to address "
        "the core of the question."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User question: {question}\n\n"
                "Generated answer:\n\n{generation}",
            ),
        ]
    )

    llm = _get_llm().with_structured_output(GradeAnswer)
    return prompt | llm
