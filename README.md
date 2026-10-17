# 🤖 Agentic Adaptive, Corrective (CRAG) & Self-Reflective RAG Orchestrator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adaptive-corrective-self-reflective-rag.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq LPU](https://img.shields.io/badge/Inference-Groq%20LPU-f55036.svg)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace%20MiniLM-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests Passing](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](#-test-suite--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A production-grade **Agentic RAG State Machine** that autonomously routes, retrieves, grades, self-corrects, and reflects on evidence using **LangGraph**, **Groq LPUs** (`qwen/qwen3.8-27b`), and local **HuggingFace** sentence-transformers (`all-MiniLM-L6-v2`).

👉 **[🚀 Experience the Live Interactive Web Demo](https://adaptive-corrective-self-reflective-rag.streamlit.app/)**

---

## 🌟 Executive Overview

Traditional naive RAG architectures suffer from four critical vulnerabilities:
1. **Semantic Misses & Keyword Blindspots**: Vector-only search misses exact technical IDs or acronyms, while keyword search misses contextual synonyms.
2. **Context Poisoning**: Irrelevant documents injected into the prompt lead to degraded answers or model confusion.
3. **Hallucination & Grounding Drift**: LLMs generate plausible-sounding facts not grounded in the retrieved sources.
4. **Rigid Execution**: Naive pipelines cannot recover when retrieved context is poor or missing.

This orchestrator unifies **three modern RAG paradigms into a resilient, cyclical graph**:

* 🧭 **Adaptive Query Routing**: Evaluates incoming queries using deterministic Pydantic schemas to route dynamically between **Hybrid Vectorstore** (FAISS + BM25), **Real-Time Web Search** (Tavily / DuckDuckGo), or **Direct Zero-Retrieval LLM** (for arithmetic, logic, and definitions).
* 🔄 **Corrective RAG (CRAG)**: Evaluates evidence relevance chunk-by-chunk. If context quality is low or off-topic, it strips noise, triggers an intelligent **Query Reformulation node**, and automatically retries retrieval or escalates to live web search upon hitting retry limits.
* 🪞 **Self-RAG Reflection**: Evaluates the drafted answer against retrieved context for **grounding/hallucination**, followed by a second evaluator checking **faithfulness/utility** against the user's original query. Low-quality answers trigger an internal self-correction loop before anything is returned to the user.
* 📚 **Verified In-Text Citation Grounding**: Answers include strict inline citations (`[Source: document_name.txt]`), verified and presented in collapsible citation inspectors.

---

## 🚀 Live Demo & Interactive Features

Try the production deployment live in your browser:  
👉 **[https://adaptive-corrective-self-reflective-rag.streamlit.app/](https://adaptive-corrective-self-reflective-rag.streamlit.app/)**

### Streamlit Web UI Highlights
* 🟢 **Real-Time State Machine Streaming**: An interactive `st.status()` container streams graph transitions step-by-step (routing decision, document filtering, query rewriting, reflection checks).
* 💡 **One-Click Example Queries**: Preloaded test queries covering dense semantic search, exact keyword search, hybrid fusion, query rewriting loops, and direct math answers.
* 📚 **Context & Citations Inspector**: Collapsible drawer with verified inline citations and raw retrieved source chunks.
* ⚙️ **Secure Sidebar Key Management**: Uses server-side secrets by default without exposing the host's API key; allows visitors to optionally enter their own Groq API key to test custom sessions.

---

## 📐 Architecture & State Machine

The pipeline is implemented as a cyclical directed state graph (`StateGraph`) using **LangGraph**:

```text
                                  ┌────────────────────────┐
                        START ───►│  route_question (LLM)  │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
            [vectorstore route]       [web_search route]     [direct_answer route]
                      │                       │                       │
                      ▼                       │                       ▼
                retrieve_docs                 │            generate_direct_answer
             (FAISS + BM25 RRF)               │            (Math, logic, direct)
                      │                       │                       │
                      ▼                       │                       ▼
               grade_documents                │                    END ✅
              (LLM Binary Grader)             │
               ┌──────┴──────┐                │
               │ Valid Docs? │                │
               │             │                │
            Yes│             │No              │
               ▼             ▼                │
        generate_answer  transform_query      │
               │         (Query Rewriter)     │
               │         ┌───┴────────────┐   │
               │         │ loop < MAX?    │   │
               │         │ Yes ──► retrieve_docs
               │         │ No  ──► web_search │
               │         └────────────────┘   │
               ▼                              ▼
  check_hallucination_and_faithfulness ◄──────┘
         (Self-RAG Dual Reflection)
         ┌───────────┴────────────────────────┐
         │ • Grounded & Useful?  ───────────► END ✅
         │ • Hallucination?      ───────────► generate_answer (Regenerate)
         │ • Not useful/off-topic? ─────────► transform_query (Re-retrieve)
         └────────────────────────────────────┘
```

---

## 🧩 Graph Node Breakdown

| Node | Purpose | Execution Logic & Guarantees |
|---|---|---|
| `route_question` | Adaptive Classifier | Analyzes query intent via `ChatGroq.with_structured_output(RouteQuery)` to branch to `vectorstore`, `web_search`, or `direct_answer`. |
| `retrieve_docs` | Hybrid Retrieval | Queries dense FAISS index (`all-MiniLM-L6-v2`) and sparse BM25 index, combining results using weighted Reciprocal Rank Fusion (RRF). |
| `grade_documents` | Evidence Filter & CRAG | Grades each chunk (`yes`/`no`) via `GradeDocuments` schema. Strips irrelevant text. Increments retry counter if context is insufficient. |
| `transform_query` | Query Rewriter | Rewrites the search query using the failure reason to expand or focus vocabulary when initial retrieval yields poor evidence. |
| `web_search` | Fallback Retrieval | Fetches real-time internet context via Tavily API with zero-config DuckDuckGo fallback when local documents are missing or exhausted. |
| `generate_answer` | Grounded Synthesis | Generates cited answers strictly from validated context documents, enforcing inline citations and producing source lists. |
| `generate_direct_answer` | Zero-Retrieval Path | Directly answers simple math expressions, logic questions, or greetings without unneeded retrieval overhead. |
| `check_hallucination_and_faithfulness` | Self-RAG Evaluator | Runs dual structured evaluations: (1) Is generation grounded in evidence? (2) Does generation answer the user's prompt? |

---

## 🛠️ Tech Stack & Technical Specifications

| Layer | Technology | Details / Implementation |
|---|---|---|
| **Orchestration** | **LangGraph** (`>=0.2.0`) + **LangChain** (`>=0.3.0`) | Stateful directed graph with cyclical edges, typed state dictionary (`GraphState`), and streaming update mode. |
| **LLM Provider** | **Groq LPU** (`langchain-groq`) | Ultra-fast inference with `qwen/qwen3.8-27b` using native structured outputs (`with_structured_output(PydanticSchema)`). |
| **Dense Search** | **FAISS** (`faiss-cpu`) | Local vector index with L2 similarity; serialized to disk for instant subsequent loads. |
| **Embeddings** | **HuggingFace** (`all-MiniLM-L6-v2`) | 384-dimensional dense embeddings running locally on CPU. Zero OpenAI API costs; private vector generation. |
| **Sparse Search** | **BM25** (`rank-bm25`) | Lexical search with document length normalization and term saturation ($k_1=1.5, b=0.75$). |
| **Hybrid Fusion** | **Reciprocal Rank Fusion** | Dynamically normalizes and fuses vector similarity with lexical scores via configurable weights (default 60/40). |
| **Web Search** | **Tavily API** + **DuckDuckGo** | High-precision search fallback for live queries or when local document retrieval retries are exhausted. |
| **Evaluators** | **Pydantic v2** (`pydantic`) | Strict schemas (`RouteQuery`, `GradeDocuments`, `GradeHallucinations`, `GradeAnswer`) eliminating regex/parsing errors. |
| **Web UI** | **Streamlit** (`>=1.35.0`) | Modern UI with dynamic status streaming, interactive query pills, and responsive expandable drawers. |
| **CLI Interface** | **Rich** (`>=13.7.0`) | Color-coded terminal trace with step badges, execution timers, and formatted panels. |

---

## 📂 Repository Structure

```text
Adaptive-Corrective-Self-Reflective-RAG/
├── README.md                      # Comprehensive documentation & architecture guide
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Excludes secrets, indexes, caches, and OS artifacts
├── app.py                         # Interactive Streamlit Web UI with live execution streaming
├── main.py                        # Rich CLI terminal entrypoint with real-time logging
├── data/
│   ├── sample_docs.json           # Curated 15-document AI/ML corpus (Transformers, CRAG, BM25, etc.)
│   └── faiss_index/               # Local FAISS index (auto-built from corpus on first launch)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic-settings config (models, weights, thresholds, keys)
│   ├── state.py                   # GraphState TypedDict definition (all pipeline state fields)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── retrievers.py          # Hybrid FAISS + BM25 with RRF Ensemble and Web fallback
│   │   ├── evaluators.py          # 4 structured-output Groq evaluators using Pydantic schemas
│   │   └── rewriter.py            # Context-aware query optimizer for retrieval retry loops
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── router_nodes.py        # route_question node
│   │   ├── retrieval_nodes.py     # retrieve_docs, web_search nodes
│   │   ├── grading_nodes.py       # grade_documents, check_hallucination_and_faithfulness nodes
│   │   └── generation_nodes.py    # generate_answer, generate_direct_answer, transform_query nodes
│   └── graph.py                   # Complete LangGraph StateGraph construction and edge compilation
└── tests/
    └── test_rag.py                # 34 comprehensive unit & integration tests
```

---

## ⚡ Quickstart & Local Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/AtharvaRathi/Adaptive-Corrective-Self-Reflective-RAG.git
cd Adaptive-Corrective-Self-Reflective-RAG

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the example configuration file:
```bash
cp .env.example .env
```
Open `.env` and provide your **Groq API Key**:
```env
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_key_here

# Optional: Tavily key for web search (falls back to DuckDuckGo if omitted)
TAVILY_API_KEY=tvly_your_key_here

# Default model configuration
LLM_MODEL=qwen/qwen3.8-27b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 🖥️ Running the Application

### Option A: Streamlit Web UI (Recommended)
```bash
python3 -m streamlit run app.py
```
* Opens automatically at `http://localhost:8501`.
* Allows testing queries with visual state machine tracing.

### Option B: Interactive Terminal CLI
```bash
python3 main.py
```

#### Terminal Execution Sample
```text
❓ Enter your question: What are transformer attention mechanisms?

── Execution Trace ──────────────────────────────────────────────────────────
  🧭 [ROUTER]     → Strategy → vectorstore | Reason: Transformer attention mechanisms are a core AI topic.
  🔍 [RETRIEVAL]  → Retrieved 8 documents via FAISS + BM25 Ensemble.
  📋 [DOC GRADER] → 2 relevant docs kept | Filtered 6 irrelevant chunks | Attempt 1
  💬 [GENERATOR]  → Answer generated (1,280 chars) with verified inline citations.
  🪞 [REFLECTION] → ✅ Grounded in context | ✅ Faithfully answers question

╭──────────────────────── Answer (strategy: vectorstore, retries: 1) ────────────────────────╮
│                                                                                            │
│  Transformer attention mechanisms are a core component of the Transformer                  │
│  architecture, which dispenses with recurrence and convolutions entirely [Source:          │
│  transformer_paper.txt]. It uses multi-head self-attention to draw global dependencies      │
│  between input and output, allowing for significantly more parallelization [Source:        │
│  transformer_paper.txt].                                                                   │
│                                                                                            │
│  The fundamental formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V            │
│  [Source: transformer_paper.txt].                                                          │
│                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────╯

📚 Citations:
  [1] transformer_paper.txt: Scaled Dot-Product Attention
```

---

## 🧪 Test Suite & Verification

The project includes an extensive test suite verifying every component using mocked LLM outputs and unit tests:

```bash
python3 -m pytest tests/test_rag.py -v
```

### Test Coverage Highlights (34/34 Passing)
* ✅ **Configuration & Validation**: Weight normalization, chunking constraints, environment resolution.
* ✅ **Document Ingestion**: JSON parsing, schema validation, metadata preservation.
* ✅ **State Management**: Immutability, transitions, retry counters.
* ✅ **Router Decisions**: Vectorstore, Web Search, and Direct LLM branches.
* ✅ **Hybrid Retrieval**: Document fusion, error resilience, fallback behavior.
* ✅ **Document Grading**: Chunk filtering, threshold enforcement, failure annotation.
* ✅ **Self-RAG Evaluators**: Groundedness verification, hallucination tripping, utility evaluation.
* ✅ **Query Rewriter**: Reformulation logic and context carry-over.
* ✅ **Graph Topology**: Cycle validation, conditional edge branching, compilation verification.

---

## ⚙️ Configuration Reference

| Setting | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | Groq API Key for LLM inference |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Groq model for routing, generation, grading, and reflection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace sentence-transformers model (CPU) |
| `CHUNK_SIZE` | `500` | Target chunk character count |
| `CHUNK_OVERLAP` | `50` | Character overlap between consecutive chunks |
| `RETRIEVER_DENSE_WEIGHT` | `0.6` | FAISS dense vector weight in ensemble fusion |
| `RETRIEVER_SPARSE_WEIGHT` | `0.4` | BM25 sparse keyword weight in ensemble fusion |
| `TOP_K_RETRIEVAL` | `5` | Number of context documents to retrieve |
| `MAX_RETRIES` | `3` | Maximum CRAG loop retries before escalating to web search |
| `TAVILY_API_KEY` | `""` | Optional Tavily web search key (falls back to DuckDuckGo) |

---

## 👤 Author & Connect

**Atharva Rathi**  
* **GitHub**: [@AtharvaRathi](https://github.com/AtharvaRathi)  
* **Repository**: [Adaptive-Corrective-Self-Reflective-RAG](https://github.com/AtharvaRathi/Adaptive-Corrective-Self-Reflective-RAG)  
* **Live Web App**: [https://adaptive-corrective-self-reflective-rag.streamlit.app/](https://adaptive-corrective-self-reflective-rag.streamlit.app/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
