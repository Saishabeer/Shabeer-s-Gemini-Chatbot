import logging
import os
from typing import Callable, Tuple, TypeVar

from google.api_core.exceptions import (InvalidArgument, PermissionDenied,
                                        ResourceExhausted)
from langchain_google_genai._common import GoogleGenerativeAIError

logger = logging.getLogger(__name__)


# --- API Key Rotation Manager ---
class ApiKeyManager:
    """Manages a pool of API keys, rotating them when one is exhausted."""

    def __init__(self):
        keys_str = os.getenv("GEMINI_API_KEYS")
        if not keys_str:
            raise ValueError("GEMINI_API_KEYS not found in .env. Please provide a comma-separated list of keys.")
        self.keys = [key.strip() for key in keys_str.split(',') if key.strip()]
        if not self.keys:
            raise ValueError(
                "GEMINI_API_KEYS was found but contained no valid keys after parsing. Please check your .env file.")
        self.current_index = 0
        logger.info(f"Loaded {len(self.keys)} API keys for rotation.")

    def get_current_key(self) -> str:
        """Returns the currently active API key."""
        return self.keys[self.current_index]

    def switch_to_next_key(self):
        """Rotates to the next key in the pool."""
        if len(self.keys) > 1:
            self.current_index = (self.current_index + 1) % len(self.keys)
            logger.info(f"Switched to API key at index {self.current_index}.")


# --- Decorator for API Key Rotation ---
T = TypeVar('T')
RETRYABLE_EXCEPTIONS: Tuple = (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError)
api_key_manager = ApiKeyManager()  # Global instance for the app


def with_api_key_rotation(func: Callable[..., T]) -> Callable[..., T]:
    """
    A decorator that wraps a function making a Google AI API call, providing
    automatic API key rotation and retries upon specific failures.
    """

    def wrapper(*args, **kwargs) -> T:
        for attempt in range(len(api_key_manager.keys)):
            try:
                return func(*args, **kwargs)
            except RETRYABLE_EXCEPTIONS as e:
                logger.warning(
                    f"API call failed with key index {api_key_manager.current_index} (Attempt {attempt + 1}/{len(api_key_manager.keys)}). Reason: {type(e).__name__}."
                )
                if attempt == len(api_key_manager.keys) - 1:
                    logger.error("All available API keys have been tried and failed.")
                    raise e
                api_key_manager.switch_to_next_key()

    return wrapper