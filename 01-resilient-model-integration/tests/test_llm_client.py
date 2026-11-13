import asyncio
from unittest.mock import AsyncMock, MagicMock
from google.genai.errors import APIError
import pytest
from src.core.config import settings
from src.services.llm_client import ResilientLLMClient


@pytest.mark.asyncio
async def test_truncation_detection(mocker):
    """Verifies that finish_reason='MAX_TOKENS' flags the response as truncated."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = "MAX_TOKENS"

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 50
    mock_usage.candidates_token_count = 100
    mock_usage.total_token_count = 150

    mock_response = MagicMock()
    mock_response.text = "Incomplete generated output..."
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage

    mocker.patch.object(
        client.client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_response),
    )

    result = await client.generate("System instructions", "User query")

    assert result.truncated is True
    assert result.finish_reason == "MAX_TOKENS"
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 100
    assert result.total_tokens == 150
    assert result.content == "Incomplete generated output..."


@pytest.mark.asyncio
async def test_transient_error_retry(mocker):
    """Verifies that transient 503/429 errors trigger a retry before succeeding."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = "STOP"

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 20
    mock_usage.candidates_token_count = 40
    mock_usage.total_token_count = 60

    mock_success_response = MagicMock()
    mock_success_response.text = "Success after initial server spike"
    mock_success_response.candidates = [mock_candidate]
    mock_success_response.usage_metadata = mock_usage

    transient_error = APIError(
        503,
        {"error": {"code": 503, "message": "Model temporarily unavailable"}},
    )

    mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_generate = AsyncMock(side_effect=[transient_error, mock_success_response])
    mocker.patch.object(client.client.aio.models, "generate_content", new=mock_generate)

    result = await client.generate("System instructions", "User query")

    assert result.content == "Success after initial server spike"
    assert result.truncated is False
    assert result.finish_reason == "STOP"
    assert mock_generate.call_count == 2


@pytest.mark.asyncio
async def test_non_retryable_client_error_fails_fast(mocker):
    """Verifies that 4xx client errors (400/404) do not trigger retries."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    client_error = APIError(
        404,
        {"error": {"code": 404, "message": "Model not found"}},
    )

    mock_sleep = mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_generate = AsyncMock(side_effect=client_error)
    mocker.patch.object(client.client.aio.models, "generate_content", new=mock_generate)

    with pytest.raises(APIError):
        await client.generate("System instructions", "User query")

    assert mock_generate.call_count == 1
    assert mock_sleep.call_count == 0


@pytest.mark.asyncio
async def test_timeout_retry_budget_exhausted_reraises(mocker):
    """Verifies that continuous timeouts stop at max_retries and re-raise."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    mocker.patch("asyncio.sleep", new=AsyncMock())
    mock_generate = AsyncMock(side_effect=asyncio.TimeoutError())
    mocker.patch.object(client.client.aio.models, "generate_content", new=mock_generate)

    with pytest.raises(asyncio.TimeoutError):
        await client.generate("System instructions", "User query")

    assert mock_generate.call_count == settings.llm_max_retries


@pytest.mark.asyncio
async def test_stream_generate_filters_empty_chunks(mocker):
    """Verifies that streaming yields only non-empty text deltas in order."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    async def _fake_stream():
        for value in ["Enterprise ", None, "", "intelligence ", "verified."]:
            chunk = MagicMock()
            chunk.text = value
            yield chunk

    mocker.patch.object(
        client.client.aio.models,
        "generate_content_stream",
        new=AsyncMock(return_value=_fake_stream()),
    )

    chunks = [
        delta
        async for delta in client.stream_generate("System instructions", "User query")
    ]

    assert chunks == ["Enterprise ", "intelligence ", "verified."]


@pytest.mark.asyncio
async def test_empty_candidates_and_null_usage_defaults(mocker):
    """Verifies graceful fallback when safety filters return empty candidates."""
    client = ResilientLLMClient(api_key="mock-api-key", model="gemini-3.8-flash")

    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = []
    mock_response.usage_metadata = None

    mocker.patch.object(
        client.client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_response),
    )

    result = await client.generate("System instructions", "Blocked query")

    assert result.content == ""
    assert result.finish_reason == "UNKNOWN"
    assert result.total_tokens == 0
    assert result.truncated is False