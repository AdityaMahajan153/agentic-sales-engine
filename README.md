# Agentic Sales & Research Outreach Engine

A systems-engineering progression building a production-grade **AI Sales & Research Outreach Assistant** across 8 incremental architectural stages. Rather than starting with high-level agent frameworks, this repository builds every reliability primitive from the ground up—from raw async SDK transport resilience and grounded citation retrieval to deterministic tool execution, stateful workflows, multi-agent coordination, and automated evaluation suites.

---

## Roadmap & Progress

| Stage | Project Folder | Core Engineering Focus | Status |
| :--- | :--- | :--- | :--- |
| **01** | [`01-resilient-model-integration`](./01-resilient-model-integration) | Native async SDK integration (`google-genai`), explicit timeout enforcement, exponential backoff with jitter, `MAX_TOKENS` truncation detection, JSON structured logging, and async SSE streaming | **Completed** |
| **02** | `02-grounded-evidence-rag` | Overlapping document chunking, dense vector embeddings, vectorized cosine similarity indexing, strict inline `[chunk_id]` citation enforcement, and similarity-threshold refusal gates | In Progress |
| **03** | `03-structured-schema-extraction` | Strict Pydantic output contracts, JSON schema enforcement, validation-repair retry loops, and deterministic company/prospect profiling | Planned |
| **04** | `04-deterministic-tool-use` | Typed function-calling schemas, explicit tool registry, runtime argument validation, sandboxed execution, and structured tool error recovery | Planned |
| **05** | `05-stateful-workflow-orchestration` | Explicit state machines, step and token budgets, infinite-loop guards, and execution checkpointing | Planned |
| **06** | `06-persistent-memory-context` | Episodic and semantic prospect memory, context window compression, and cross-session recall | Planned |
| **07** | `07-multi-agent-coordination` | Specialized agent roles (Researcher, Analyst, Outreach Writer, Critic/Verifier) with bounded, typed handoffs | Planned |
| **08** | `08-evaluation-observability-production` | Automated evaluation benchmarks, end-to-end latency/token cost tracking, regression suites, and production hardening | Planned |

---

## Repository Structure

All stages share a single root environment (`pyproject.toml`, `uv.lock`, and `.env`), while each project is isolated in its own self-contained directory with dedicated source code, unit tests, and documentation:

```text
agentic-sales-engine/
├── .gitignore                                # Excludes .env, .venv, and cache artifacts
├── pyproject.toml                            # Shared dependencies managed via uv
├── uv.lock                                   # Deterministic dependency lockfile
├── README.md                                 # Collective roadmap and setup guide
│
├── 01-resilient-model-integration/           # [Stage 01] Completed
│   ├── README.md                             # Architecture, failure modes & test docs
│   ├── pyproject.toml                        # Local pytest import resolution
│   ├── src/
│   │   ├── core/
│   │   │   ├── config.py                     # Pydantic-settings environment validation
│   │   │   └── logging.py                    # JSON structlog telemetry setup
│   │   ├── services/
│   │   │   └── llm_client.py                 # ResilientLLMClient + LLMResponse schema
│   │   └── main.py                           # Blocking & streaming demo entrypoint
│   └── tests/
│       └── test_llm_client.py                # 6-test async mock regression suite
│
├── 02-grounded-evidence-rag/                 # [Stage 02] In Progress
├── 03-structured-schema-extraction/          # [Stage 03] Planned
├── 04-deterministic-tool-use/                # [Stage 04] Planned
├── 05-stateful-workflow-orchestration/       # [Stage 05] Planned
├── 06-persistent-memory-context/             # [Stage 06] Planned
├── 07-multi-agent-coordination/              # [Stage 07] Planned
└── 08-evaluation-observability-production/   # [Stage 08] Planned
```

---

## Engineering Principles

- **Direct SDK Control First:** Interfaces directly with the native `google-genai` SDK (`v1beta`) to maintain full control over system instructions, token budgets, timeouts, and retry boundaries before introducing orchestration abstractions.
- **Deterministic Failure Handling:** Disables hidden internal SDK retries (`HttpRetryOptions(attempts=1)`) and enforces explicit `asyncio.wait_for` timeout ceilings paired with `tenacity` exponential backoff—retrying transient `429`/`5xx`/timeout spikes while failing fast on permanent `4xx` errors.
- **Observability by Default:** Emits structured JSON logs via `structlog` capturing request lifecycle events, retry backoffs, candidate `finish_reason` flags, and token accounting.
- **Mock-Isolated Regression Testing:** Every stage includes an asynchronous `pytest` suite (`pytest-asyncio`, `pytest-mock`) validating edge cases and failure modes without consuming live API quota.

---

## Tech Stack

- **Language & Package Management:** Python 3.13+, `uv`
- **LLM & Embedding SDK:** `google-genai` (`gemini-3.5-flash-lite`, `gemini-embedding-001`)
- **Validation & Resilience:** `pydantic`, `pydantic-settings`, `tenacity`, `structlog`, `numpy`
- **Testing:** `pytest`, `pytest-asyncio`, `pytest-mock`

---

## Quickstart

### 1. Clone and Install Dependencies

```bash
git clone [https://github.com/](https://github.com/)<your-username>/agentic-sales-engine.git
cd agentic-sales-engine
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root (this file is git-ignored):

```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-3.5-flash-lite
LLM_TIMEOUT_SECONDS=45.0
LLM_MAX_RETRIES=5
LOG_LEVEL=INFO
```

### 3. Run Project 01 Tests & Live Demo

```bash
# Execute the unit test suite
uv run --directory 01-resilient-model-integration pytest -v

# Run the live blocking & streaming verification script
uv run --directory 01-resilient-model-integration python -m src.main
```
