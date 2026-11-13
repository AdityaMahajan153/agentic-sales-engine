import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.core.config import settings
from src.core.logging import logger


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    truncated: bool


def _is_transient_error(exc: BaseException) -> bool:
    """Retry timeouts, rate limits (429), or transient 5xx server drops."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        return code in (429, 500, 502, 503, 504)
    return False


def _log_retry(retry_state: Any) -> None:
    """Emits structured JSON retry logs compatible with structlog."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_time = getattr(retry_state.next_action, "sleep", 0.0)
    error_detail = (
        f"{type(exception).__name__}: {exception}" if exception else "UnknownError"
    )
    logger.warning(
        "llm_request_retry",
        attempt=retry_state.attempt_number,
        delay_seconds=round(sleep_time, 2),
        error=error_detail,
    )


class ResilientLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.llm_model
        # Disable internal SDK retries (attempts=1) so tenacity owns the retry budget
        self.client = genai.Client(
            api_key=api_key or settings.gemini_api_key,
            http_options=types.HttpOptions(
                api_version="v1beta",
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_random_exponential(min=2.0, multiplier=2.0, max=15.0),
        retry=retry_if_exception(_is_transient_error),
        before_sleep=_log_retry,
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        logger.info(
            "llm_request_started",
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            ),
            timeout=settings.llm_timeout_seconds,
        )

        candidate = response.candidates[0] if response.candidates else None
        raw_finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
        finish_reason = getattr(raw_finish_reason, "name", str(raw_finish_reason))
        is_truncated = finish_reason == "MAX_TOKENS"

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        total_tokens = usage.total_token_count if usage else 0

        if is_truncated:
            logger.warning(
                "llm_response_truncated",
                finish_reason=finish_reason,
                tokens_used=completion_tokens,
            )

        logger.info(
            "llm_request_completed",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            truncated=is_truncated,
        )

    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        logger.info("llm_stream_started", model=self.model)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        async for chunk in stream:
            if chunk.text:
                yield chunk.text