# Project 01: Resilient Model Integration

Foundational async LLM transport layer built directly on the native `google-genai` SDK without high-level agent framework abstractions.

## Engineering Objectives

- **Direct SDK Control:** Configures system instructions, temperature, and output token budgets directly via `GenerateContentConfig` on the `v1beta` endpoint.
- **Deterministic Timeout & Retry Ownership:** Disables hidden internal SDK retries (`HttpRetryOptions(attempts=1)`) and enforces explicit `asyncio.wait_for` timeout boundaries wrapped in `tenacity` exponential backoff with jitter.
- **Failure Classification:** Retries only transient faults (`asyncio.TimeoutError`, HTTP `429`, `5xx`) while failing fast on permanent `4xx` client errors.
- **Truncation & Safety Guardrails:** Inspects candidate `finish_reason` metadata to flag `MAX_TOKENS` truncation and safely defaults empty safety-blocked candidate lists.
- **Structured JSON Telemetry:** Emits machine-readable JSON logs via `structlog` capturing latency events, retry backoffs, finish reasons, and token usage.
- **Async SSE Streaming:** Exposes an async generator (`stream_generate`) filtering empty metadata deltas.

## Folder Structure

```text
01-resilient-model-integration/
├── README.md
├── pyproject.toml
├── src/
│   ├── core/
│   │   ├── config.py        # Pydantic-settings environment validation
│   │   └── logging.py       # JSON structlog configuration
│   ├── services/
│   │   └── llm_client.py    # ResilientLLMClient + LLMResponse contract
│   └── main.py              # Blocking & streaming verification entrypoint
└── tests/
    └── test_llm_client.py   # 6-test async mock suite
