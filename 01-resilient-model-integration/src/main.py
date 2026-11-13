import sys
from pathlib import Path

# Add project root directory to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from src.core.logging import logger, setup_logging
from src.services.llm_client import ResilientLLMClient


async def main() -> None:
    setup_logging()
    client = ResilientLLMClient()

    logger.info("starting_agent_demo")

    # 1. Standard validated generation
    response = await client.generate(
        system_prompt="You are a precise corporate intelligence assistant.",
        user_prompt="Summarize the core business value of Snowflake in 2 sentences.",
    )

    print(f"\nResponse:\n{response.content}")
    print(f"\nMetadata: Tokens={response.total_tokens}, Truncated={response.truncated}")

    # 2. Token Streaming
    print("\nStreaming test:")
    async for chunk in client.stream_generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="List 3 key metrics to track when evaluating B2B SaaS companies.",
    ):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())