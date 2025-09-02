import os
import logging
from functools import wraps
from threading import Lock
from google.api_core.exceptions import ResourceExhausted, PermissionDenied
from langchain_google_genai._common import GoogleGenerativeAIError

logger = logging.getLogger(__name__)


class ApiKeyManager:
    """Manages a pool of API keys for rotation."""

    def __init__(self):
        """
        Initializes the key manager by loading keys from the environment.
        The settings.py file is responsible for loading the .env file for local dev.
        This class just reads from the environment.
        """
        keys_str = os.getenv("GEMINI_API_KEYS")
        if not keys_str:
            # This error message is now clear for both local and production environments.
            raise ValueError(
                "GEMINI_API_KEYS not found in the environment. "
                "Please set it in your .env file (for local development) "
                "or in your hosting provider's environment variables (for production)."
            )

        self.keys = [key.strip() for key in keys_str.split(',') if key.strip()]
        if not self.keys:
            raise ValueError("GEMINI_API_KEYS environment variable is empty or contains only whitespace.")

        self.current_key_index = 0
        self.lock = Lock()
        logger.info(f"Loaded {len(self.keys)} API keys for rotation.")

    def get_key(self) -> str:
        """Get the current key."""
        with self.lock:
            return self.keys[self.current_key_index]

    def rotate_key(self) -> str:
        """Rotate to the next key in the list and return it."""
        with self.lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            logger.warning(f"API key rotated. Now using key index {self.current_key_index}.")
            return self.keys[self.current_key_index]


# Create a single, global instance for the application to use.
# This will raise the ValueError on startup if keys are not configured.
api_key_manager = ApiKeyManager()


def with_api_key_rotation(func):
    """A decorator to handle API key rotation on specific exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):  
        # Try to execute the function with the current key.
        # We loop through all available keys to find one that works.
        for i in range(len(api_key_manager.keys)):
            try:
                # The decorated function (e.g., gemini_chat_stream) will use
                # api_key_manager.get_key() to get the current key.
                return func(*args, **kwargs)
            except (ResourceExhausted, PermissionDenied, GoogleGenerativeAIError) as e:
                logger.warning(f"API call failed with key index {api_key_manager.current_key_index}. Reason: {e}")
                # If it's the last key in the list, we've tried them all. Re-raise the exception.
                if i == len(api_key_manager.keys) - 1:
                    logger.error("All API keys have failed. No more keys to rotate to.")
                    raise e
                # Otherwise, rotate to the next key and the loop will retry.
                api_key_manager.rotate_key()
        # This part should not be reachable if there's at least one key, but it's good practice.
        raise Exception("Failed to execute function after trying all API keys.")
    return wrapper