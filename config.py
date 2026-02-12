"""Configuration for the email agent workflow."""
import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel

load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")


def get_model() -> OpenAIChatCompletionsModel:
    """Create and return the Ollama-backed model."""
    return OpenAIChatCompletionsModel(
        model=OLLAMA_MODEL,
        openai_client=AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # Dummy key - Ollama doesn't require auth
        )
    )
