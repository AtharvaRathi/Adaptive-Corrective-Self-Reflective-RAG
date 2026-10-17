"""
app.py
──────
Interactive Streamlit frontend for the Adaptive, Corrective (CRAG) &
Self-Reflective RAG Orchestrator powered by Groq and local HuggingFace embeddings.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import streamlit as st
from langchain_core.documents import Document

# Sync Streamlit Community Cloud secrets into os.environ if present
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            if isinstance(_v, str) and _k not in os.environ:
                os.environ[_k] = _v
except Exception:
    pass

from src.config import get_settings
from src.graph import get_graph
from src.state import make_initial_state


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive & Self-Reflective RAG (Groq)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished badge and container styling
st.markdown(
    """
    <style>
    .node-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .citation-card {
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4CAF50;
        background-color: rgba(76, 175, 80, 0.08);
        border-radius: 2px;
    }
    .metric-container {
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        settings = get_settings()

        api_key_input = st.text_input(
            "Groq API Key",
            value=settings.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Get your free key at https://console.groq.com/keys",
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            settings.groq_api_key = api_key_input

        if not settings.groq_api_key:
            st.warning("⚠️ Please provide a `GROQ_API_KEY` to run queries.", icon="⚠️")
            st.markdown(
                "[👉 Get a free Groq API key here](https://console.groq.com/keys)"
            )

        st.divider()
        st.subheader("Model Configuration")
        model_options = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "allam-2-7b",
            "Custom...",
        ]
        current_model = settings.llm_model
        default_idx = model_options.index(current_model) if current_model in model_options else len(model_options) - 1

        selected_model = st.selectbox("Groq Chat Model", model_options, index=default_idx)
        if selected_model == "Custom...":
            chosen_model = st.text_input("Enter Groq Model Name", value=current_model)
        else:
            chosen_model = selected_model

        if chosen_model and chosen_model != settings.llm_model:
            os.environ["LLM_MODEL"] = chosen_model
            settings.llm_model = chosen_model

        st.caption(f"🧠 **Active Model:** `{settings.llm_model}`")
        st.caption(f"🔤 **Embeddings:** `{settings.embedding_model}` (HuggingFace CPU)")
        st.caption(f"⚖️ **Weights:** Dense {settings.retriever_weights[0]:.1f} | Sparse {settings.retriever_weights[1]:.1f}")
        st.caption(f"🔁 **Max Retries:** {settings.max_retries}")

        st.divider()
        st.subheader("Architecture Capabilities")
        st.markdown(
            """
            - 🧭 **Adaptive Routing**: Classifies queries to vectorstore, web search, or direct LLM.
            - 🔄 **Corrective RAG (CRAG)**: Filters low-relevance docs and automatically rewrites queries.
            - 🪞 **Self-RAG Reflection**: Evaluates hallucination and question usefulness before outputting.
            - 📚 **Citation Grounding**: Inline verified document citations.
            """
        )

        st.divider()
        st.subheader("💡 Example Queries")
        st.markdown(
            """
            - **Semantic:** *"What are transformer attention mechanisms?"*
            - **Keyword:** *"What is BM25 retrieval?"*
            - **Hybrid/CRAG:** *"How does Corrective RAG handle bad docs?"*
            - **Web:** *"Latest developments in AI"*
            - **Direct:** *"What is 25 * 4 + 100?"*
            """
        )

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

    return settings


# ── Helper to format node output for live status streaming ─────────────────────
def format_event_update(node_name: str, node_output: Dict[str, Any]) -> str:
    """Format node output into a human-readable live update string."""
    if node_name == "route_question":
        strategy = node_output.get("search_strategy", "unknown")
        icons = {"vectorstore": "📚 Hybrid Vector/BM25", "web_search": "🌐 Web Search", "direct_answer": "⚡ Direct LLM"}
        display_strategy = icons.get(strategy, strategy)
        return f"🟢 **Executing Node: route_question** → 👉 Search Strategy Chosen: **{display_strategy}**"

    elif node_name == "retrieve_docs":
        docs = node_output.get("documents", [])
        return f"🟢 **Executing Node: retrieve_docs** → 🔍 Retrieved **{len(docs)}** context documents via FAISS + BM25"

    elif node_name == "web_search":
        docs = node_output.get("documents", [])
        return f"🟢 **Executing Node: web_search** → 🌐 Web retrieved **{len(docs)}** fallback results"

    elif node_name == "grade_documents":
        docs = node_output.get("documents", [])
        step = node_output.get("loop_step", 1)
        failure = node_output.get("failure_reason")
        status = f"🟢 **Executing Node: grade_documents** → 📋 Filtered: **{len(docs)}** relevant doc(s) kept (Attempt {step})"
        if failure:
            status += f"\n  - ⚠️ *Context insufficient: {failure}*"
        return status

    elif node_name == "transform_query":
        new_q = node_output.get("question", "")
        return f"🟢 **Executing Node: transform_query** → ✍️ Rewritten Query: *\"{new_q}\"*"

    elif node_name == "generate_answer":
        return "🟢 **Executing Node: generate_answer** → 💡 Generated cited response with Groq LLM"

    elif node_name == "generate_direct_answer":
        return "🟢 **Executing Node: generate_direct_answer** → ⚡ Generated direct response with Groq LLM"

    elif node_name == "check_hallucination_and_faithfulness":
        is_grounded = node_output.get("is_grounded", False)
        is_useful = node_output.get("is_useful", False)
        g_badge = "✅ Grounded" if is_grounded else "❌ Hallucination Detected"
        u_badge = "✅ Answers Question" if is_useful else "❌ Off-Topic/Incomplete"
        return f"🟢 **Executing Node: check_hallucination_and_faithfulness** → 🪞 Reflection: {g_badge} | {u_badge}"

    return f"🟢 **Executing Node: {node_name}**"


# ── Help & Example Queries Section ─────────────────────────────────────────────
def render_help_section():
    """Renders the interactive example queries and routing help guide."""
    is_empty_chat = len(st.session_state.get("messages", [])) == 0
    with st.expander("💡 **Example Queries & Routing Guide**", expanded=is_empty_chat):
        st.markdown(
            """
            This orchestrator uses an **Adaptive Router** to choose the best retrieval strategy for your question:
            
            | Query Type | What It Exercises | Expected Route |
            |---|---|---|
            | **Semantic Search** | Dense vector similarity via local HuggingFace embeddings + FAISS | `vectorstore` |
            | **Keyword Search** | Exact technical terms, acronyms, and formulas via BM25 | `vectorstore` |
            | **Hybrid Fusion** | Complex questions combining concepts and exact terminology | `vectorstore` |
            | **CRAG Self-Correction** | Intentionally obscure or low-relevance topics trigger rewriter/web fallback | `transform_query` → `web_search` |
            | **Live Web Search** | Real-time news, current events, or live information | `web_search` |
            | **Direct LLM Answer** | Basic factual questions, math, logic, or programming syntax | `direct_answer` |
            """
        )
        st.markdown("**Click any sample query below to test the pipeline immediately:**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 What are transformer attention mechanisms?", use_container_width=True):
                st.session_state.pending_query = "What are transformer attention mechanisms?"
                st.rerun()
            if st.button("🔍 What is BM25 retrieval and what are parameters k1 and b?", use_container_width=True):
                st.session_state.pending_query = "What is BM25 retrieval and what are parameters k1 and b?"
                st.rerun()
            if st.button("⚡ What is 25 * 4 + 100?", use_container_width=True):
                st.session_state.pending_query = "What is 25 * 4 + 100?"
                st.rerun()

        with col2:
            if st.button("🔄 How does Corrective RAG (CRAG) handle low-relevance docs?", use_container_width=True):
                st.session_state.pending_query = "How does Corrective RAG (CRAG) handle low-relevance docs?"
                st.rerun()
            if st.button("🪞 How does Self-RAG evaluate hallucination and utility?", use_container_width=True):
                st.session_state.pending_query = "How does Self-RAG evaluate hallucination and utility?"
                st.rerun()
            if st.button("🌐 What are the latest developments in generative AI models?", use_container_width=True):
                st.session_state.pending_query = "What are the latest developments in generative AI models?"
                st.rerun()


# ── Main Application ──────────────────────────────────────────────────────────
def main():
    settings = render_sidebar()

    st.title("🤖 Adaptive, CRAG & Self-Reflective RAG Orchestrator")
    st.caption(f"Powered by **Groq** (`{settings.llm_model}`), **HuggingFace** (`{settings.embedding_model}`), and **LangGraph**")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Render Example Queries & Help section
    render_help_section()

    # Display past chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for i, cite in enumerate(msg["citations"], 1):
                        st.markdown(f"**[{i}]** {cite}")
            if "context_docs" in msg and msg["context_docs"]:
                with st.expander("📄 Retrieved Context Chunks", expanded=False):
                    for i, doc in enumerate(msg["context_docs"], 1):
                        st.markdown(f"**Document {i}:** `{doc.get('title', 'Untitled')}` (*{doc.get('source', 'Unknown')}*)")
                        st.text(doc.get("content", ""))
                        st.divider()

    # Chat input
    user_query = st.chat_input("Ask a question about AI, RAG, Machine Learning, or general knowledge...")

    # Determine query to run: from input or clicked sample button
    active_query = user_query
    if not active_query and st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if active_query:
        # Display user question
        with st.chat_message("user"):
            st.markdown(active_query)

        st.session_state.messages.append({"role": "user", "content": active_query})

        if not settings.groq_api_key:
            with st.chat_message("assistant"):
                st.error("⚠️ Groq API Key is missing. Please set `GROQ_API_KEY` in the sidebar or `.env` file.")
            return

        # Assistant processing with live LangGraph state machine streaming
        with st.chat_message("assistant"):
            status_container = st.status("🚀 Orchestrating LangGraph State Machine...", expanded=True)
            
            graph = get_graph()
            initial_state = make_initial_state(active_query)

            final_state = dict(initial_state)
            start_time = time.perf_counter()

            try:
                for event in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            continue
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
                        
                        # Print live update in the status container
                        update_msg = format_event_update(node_name, node_output)
                        status_container.markdown(update_msg)

                elapsed = time.perf_counter() - start_time
                status_container.update(
                    label=f"✅ Execution Complete ({elapsed:.2f}s) — Strategy: {final_state.get('search_strategy', 'N/A')}",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                status_container.update(label="❌ Error during graph execution", state="error", expanded=True)
                st.error(f"Execution Error: {e}")
                return

            # Retrieve final answers and metadata
            generation = (final_state.get("generation") or "").strip()
            if not generation:
                generation = "⚠️ No response was produced for this query. Please check your query or model connection."
            citations = final_state.get("citations", [])
            documents: List[Document] = final_state.get("documents", [])

            # 1. Render Final Answer
            st.markdown(generation)

            # 2. Render Retrieved Context & Citations in st.expander()
            if citations or documents:
                with st.expander("📚 Retrieved Context & Citations", expanded=False):
                    if citations:
                        st.markdown("### Citations")
                        for i, cite in enumerate(citations, 1):
                            st.markdown(f"- **[{i}]** {cite}")
                        st.markdown("---")

                    if documents:
                        st.markdown("### Context Documents")
                        for i, doc in enumerate(documents, 1):
                            title = doc.metadata.get("title", f"Document {i}")
                            source = doc.metadata.get("source", "unknown")
                            st.markdown(f"**[{i}] {title}** (`{source}`)")
                            st.caption(doc.page_content)

            # Save message to session state
            saved_docs = [
                {
                    "title": doc.metadata.get("title", f"Document {i}"),
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content,
                }
                for i, doc in enumerate(documents, 1)
            ]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": generation,
                    "citations": citations,
                    "context_docs": saved_docs,
                }
            )


if __name__ == "__main__":
    main()
