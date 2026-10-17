# Adaptive, Corrective & Self-Reflective RAG Orchestrator

A production-grade **LangGraph**-based Retrieval-Augmented Generation system powered by **Groq** (`qwen/qwen3.8-27b`) and **HuggingFace** local embeddings (`all-MiniLM-L6-v2`), implementing three advanced RAG paradigms:

- 🧭 **Adaptive Query Routing** — classifies queries and routes to dense, sparse, hybrid, or web search
- 🔄 **Corrective RAG (CRAG)** — grades retrieved evidence and rewrites queries when context quality is low
- 🪞 **Self-RAG Reflection** — evaluates generated answers for hallucination and faithfulness before returning them
- 🖥️ **Live Streamlit UI** — real-time visualization of the LangGraph state machine execution steps and citations

---

## Architecture

```
                      ┌─────────────────┐
            START ───►│  route_question  │
                      └────────┬────────┘
             ┌─────────────────┼──────────────────┐
             ▼                 ▼                  ▼
       retrieve_docs       web_search        [direct END]
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
      ├── not grounded (hallucination)► generate_answer (retry)
      └── grounded but not useful ──► transform_query
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (`langgraph`) + LangChain (`langchain-core`) |
| LLM Provider | **Groq** (`langchain-groq`, default: `qwen/qwen3.8-27b`) |
| Dense Retrieval | **FAISS** (`faiss-cpu`) + **HuggingFace** (`all-MiniLM-L6-v2`, local CPU) |
| Sparse Retrieval | **BM25** (`rank-bm25`) |
| Hybrid Fusion | `EnsembleRetriever` |
| Web Fallback | Tavily API + DuckDuckGo |
| Evaluation | Structured-output LLM chains with Pydantic (`ChatGroq.with_structured_output`) |
| Web Frontend | **Streamlit** (`streamlit`) |
| Config & State | `pydantic-settings` + `.env` + `TypedDict` GraphState |
| CLI Output | `rich` |

---

## Project Structure

```text
adaptive-self-reflective-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                      # Interactive Streamlit Web UI
├── main.py                     # Rich CLI Terminal Entrypoint
├── data/
│   ├── sample_docs.json        # Bootstrap corpus (15 docs across AI/ML topics)
│   └── faiss_index/            # Local FAISS vector index (auto-generated)
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic-settings configuration
│   ├── state.py                # GraphState TypedDict definition
│   ├── components/
│   │   ├── retrievers.py       # FAISS (HuggingFace) + BM25 + Ensemble + Web
│   │   ├── evaluators.py       # 4 structured-output Groq evaluators
│   │   └── rewriter.py         # Groq query rewriting chain
│   ├── nodes/
│   │   ├── router_nodes.py     # route_question
│   │   ├── retrieval_nodes.py  # retrieve_docs, web_search
│   │   ├── grading_nodes.py    # grade_documents, check_hallucination_and_faithfulness
│   │   └── generation_nodes.py # transform_query, generate_answer
│   └── graph.py                # LangGraph StateGraph compilation
└── tests/
    └── test_rag.py             # Complete test suite (34 unit & integration tests)
```

---

## Setup & Quickstart

### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd "Adaptive RAG AI"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment & API Keys

Create a `.env` file by copying `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

- **Groq API Key (Required for LLM generation & evaluation):**
  1. Visit [https://console.groq.com/keys](https://console.groq.com/keys).
  2. Sign in or sign up (free tier available).
  3. Create an API Key and set:
     ```env
     GROQ_API_KEY=gsk_...
     ```
- **Tavily API Key (Optional for web search fallback):**
  - Sign up at [https://tavily.com](https://tavily.com) to obtain a key.
  - If omitted, the orchestrator automatically falls back to DuckDuckGo search.
- **Embeddings:**
  - `all-MiniLM-L6-v2` runs locally via HuggingFace and Sentence-Transformers on CPU. **No OpenAI API key required.**

---

## Running the Application

### Option A: Interactive Streamlit Web UI (Recommended)

Launch the visual web demo with real-time LangGraph execution tracing:

```bash
python3 -m streamlit run app.py
```
*(or `streamlit run app.py` if your Python bin directory is on your PATH)*

Features:
- 💬 **Interactive Chat Interface**: Ask any question and watch the state machine stream steps in real time.
- 🟢 **Live Graph State Stream**: Visual `st.status()` container tracking node routing, document filtering, query rewriting, and self-reflection checks.
- 📚 **Retrieved Context & Citations**: Collapsible expander displaying cited sources and original document text chunks.
- ⚙️ **Sidebar Key Management**: Enter or update your Groq API key directly from the UI without restarting.

### Option B: Rich Terminal CLI

Run the interactive terminal interface:

```bash
python main.py
```

Example terminal session:

```text
❓ Enter your question: What are transformer attention mechanisms?

[ROUTER]    → Strategy: vectorstore (hybrid)
[RETRIEVAL] → Retrieved 5 documents
[GRADER]    → 4/5 documents relevant
[GENERATOR] → Generating answer with 4 context chunks...
[REFLECTION]→ Grounded: ✅ | Useful: ✅
[ANSWER]    → Transformer attention mechanisms compute...

📚 Citations:
  [1] transformer_paper.txt: Scaled Dot-Product Attention
  [2] attention_is_all_you_need.txt: Multi-Head Attention
```

---

## Running Tests

Run the full test suite with pytest:

```bash
python3 -m pytest tests/test_rag.py -v
```

All 34 tests execute with mocked LLM outputs and test:
- Configuration validation and weight normalisation
- Document ingestion and metadata handling
- Routing logic (vectorstore, web_search, direct_answer)
- Hybrid FAISS + BM25 retrieval
- Document relevance grading
- Hallucination and faithfulness reflection
- Query rewriting loops and max retry escalation
- LangGraph graph compilation

---

## Key Design Decisions

### 1. Groq Ultra-Fast Inference
All routing, grading, rewriting, and final generation tasks run against Groq using `qwen/qwen3.8-27b` with low latency and structured outputs (`with_structured_output(PydanticSchema)`).

### 2. Local HuggingFace Embeddings
Dense vectors are embedded with `all-MiniLM-L6-v2` via `langchain-huggingface` and indexed in FAISS locally on CPU, ensuring zero embedding API cost and private local processing.

### 3. Adaptive Routing
The `route_question` node dynamically classifies the incoming question into:
- **`vectorstore`**: AI/ML, NLP, transformers, and topics in the corpus.
- **`web_search`**: Current events, live data, or unknown domains.
- **`direct_answer`**: Basic factual questions or greetings.

### 4. Corrective RAG (CRAG)
Retrieved evidence is evaluated for relevance. If zero relevant documents are found, `transform_query` rewrites the question using the failure context. If retries hit `MAX_RETRIES` (default: 3), retrieval automatically escalates to web search.

### 5. Self-RAG Reflection
Generations are checked for:
1. **Groundedness**: Are all claims supported by context?
2. **Faithfulness & Utility**: Does it answer the user's specific prompt?

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | Groq API Key |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Groq chat model (configurable in `.env` or Streamlit sidebar) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `CHUNK_SIZE` | `500` | Document chunk size |
| `RETRIEVER_DENSE_WEIGHT` | `0.6` | FAISS dense weight in ensemble |
| `RETRIEVER_SPARSE_WEIGHT` | `0.4` | BM25 sparse weight in ensemble |
| `TOP_K_RETRIEVAL` | `5` | Documents retrieved per search |
| `MAX_RETRIES` | `3` | Max CRAG loop retry attempts |
| `TAVILY_API_KEY` | `""` | Optional Tavily search key (falls back to DuckDuckGo) |

---

## License

MIT
