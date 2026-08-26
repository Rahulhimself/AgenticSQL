"""
LLM initialization and management.

Provides a factory function to create the Gemini LLM with proper
error handling for API key and network issues.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Config

logger = logging.getLogger(__name__)


def create_llm(config: Config) -> ChatGoogleGenerativeAI:
    """
    Initialize the Google Gemini LLM.

    Args:
        config: Application configuration with API key and model settings.

    Returns:
        An initialized ChatGoogleGenerativeAI instance.

    Raises:
        RuntimeError: If initialization fails, with troubleshooting tips.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            google_api_key=config.google_api_key,
        )
        logger.info("LLM initialized: %s (temperature=%.1f)", config.llm_model, config.llm_temperature)
        return llm

    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize LLM ({config.llm_model}): {e}\n\n"
            f"Troubleshooting:\n"
            f"  1. Is GOOGLE_API_KEY valid in .env?\n"
            f"  2. Do you have internet access?\n"
            f"  3. Is the model name '{config.llm_model}' correct?\n"
            f"  4. Check your API quota at https://aistudio.google.com/"
        ) from e
