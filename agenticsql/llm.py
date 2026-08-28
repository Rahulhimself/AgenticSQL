"""
LLM initialization and multi-provider management for AgenticSQL.

Supports:
- Groq API (ChatGroq with Llama 3.3 70B, Llama 3.1 8B, etc.)
- Google Gemini (ChatGoogleGenerativeAI)
- OpenAI / Azure / DeepSeek / OpenRouter (ChatOpenAI)
- Mock / Fake LLM for zero-token deterministic CI testing
"""

import logging
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Config

logger = logging.getLogger(__name__)


def create_llm(config: Config) -> BaseChatModel:
    """
    Initialize the configured LLM provider (Groq, Gemini, OpenAI, or Mock).

    Args:
        config: Application configuration with provider, API key, and model settings.

    Returns:
        An initialized LangChain BaseChatModel instance.

    Raises:
        RuntimeError: If initialization fails, with helpful troubleshooting tips.
    """
    provider = (config.llm_provider or "groq").lower().strip()

    # 1. Groq (Primary recommended provider)
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq

            model_name = config.llm_model or "openai/gpt-oss-120b"
            # Fallback to standard Groq model if previous Gemini model name was left in config
            if "gemini" in model_name.lower():
                model_name = "openai/gpt-oss-120b"

            llm = ChatGroq(
                model=model_name,
                temperature=config.llm_temperature,
                api_key=config.groq_api_key,
            )
            logger.info("Groq LLM initialized: %s (temperature=%.1f)", model_name, config.llm_temperature)
            return llm

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Groq LLM ({config.llm_model}): {e}\n\n"
                f"Troubleshooting:\n"
                f"  1. Is GROQ_API_KEY valid in your .env file? (Get a free key at https://console.groq.com/keys)\n"
                f"  2. Is 'langchain-groq' installed? (Run: pip install langchain-groq groq)\n"
                f"  3. Is the model name '{config.llm_model}' valid on Groq (e.g. 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant')?\n"
                f"  4. Check your internet connection."
            ) from e

    # 2. Google Gemini
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = config.llm_model or "gemini-2.5-flash"
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=config.llm_temperature,
                google_api_key=config.google_api_key,
            )
            logger.info("Gemini LLM initialized: %s (temperature=%.1f)", model_name, config.llm_temperature)
            return llm

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Gemini LLM ({config.llm_model}): {e}\n\n"
                f"Troubleshooting:\n"
                f"  1. Is GOOGLE_API_KEY valid in .env?\n"
                f"  2. Check your API quota at https://aistudio.google.com/\n"
                f"  3. Is the model name '{config.llm_model}' valid?"
            ) from e

    # 3. OpenAI / Azure / DeepSeek / OpenRouter
    elif provider in ("openai", "azure", "deepseek", "openrouter"):
        try:
            from langchain_openai import ChatOpenAI

            model_name = config.llm_model or "gpt-4o-mini"
            llm = ChatOpenAI(
                model=model_name,
                temperature=config.llm_temperature,
                api_key=config.openai_api_key,
            )
            logger.info("OpenAI-compatible LLM initialized: %s", model_name)
            return llm

        except Exception as e:
            raise RuntimeError(f"Failed to initialize {provider} LLM: {e}") from e

    # 4. Mock / Offline Testing LLM
    elif provider in ("mock", "fake"):
        try:
            from langchain_community.chat_models import FakeListChatModel
            return FakeListChatModel(
                responses=[
                    "Action: sql_db_query\nAction Input: SELECT 1;",
                    "Final Answer: Query executed successfully.",
                ]
            )
        except Exception:
            from unittest.mock import MagicMock
            mock_llm = MagicMock()
            mock_llm.model = "mock-model"
            return mock_llm

    else:
        raise ValueError(
            f"Unsupported LLM provider '{config.llm_provider}'. "
            f"Supported providers: 'groq', 'gemini', 'openai', 'mock'."
        )
