# --- Python Standard Library Imports ---
import os
import logging
from functools import wraps
from threading import Lock

# --- Third-Party Library Imports ---
# Specific exceptions from Google's API client and LangChain's wrapper.
# These are used to detect when an API key has failed (e.g., due to rate limits).
from google.api_core.exceptions import ResourceExhausted, PermissionDenied
from langchain_google_genai._common import GoogleGenerativeAIError

# Get a logger instance for this file.
logger = logging.getLogger(__name__)


class ApiKeyManager:
    """
    A thread-safe class to manage a pool of API keys.
    It allows for automatic rotation to the next key when one fails.
    """

    def __init__(self):
        """
        Initializes the key manager by loading keys from the environment.
        This method runs when the application starts up.
        """
        # 1. Read the comma-separated keys from the environment variable.
        keys_str = os.getenv("GEMINI_API_KEYS")
        if not keys_str:
            # If the environment variable is not set, raise an error to prevent the app from starting incorrectly.
            raise ValueError(
                "GEMINI_API_KEYS not found in the environment. "
                "Please set it in your .env file (for local development) "
                "or in your hosting provider's environment variables (for production)."
            )

        # 2. Split the string into a list of keys, removing any extra whitespace or empty entries.
        self.keys = [key.strip() for key in keys_str.split(",") if key.strip()]
        if not self.keys:
            # If the variable is set but empty, raise an error.
            raise ValueError(
                "GEMINI_API_KEYS environment variable is empty or contains only whitespace."
            )

        # 3. Set the initial key to the first one in the list.
        self.current_key_index = 0
        # 4. Create a lock to ensure that key rotation is thread-safe (prevents race conditions).
        self.lock = Lock()
        logger.info(f"Loaded {len(self.keys)} API keys for rotation.")

    def get_key(self) -> str:
        """Safely get the currently active API key."""
        # The `with self.lock:` block ensures that no other thread can change the index while we are reading it.
        with self.lock:
            return self.keys[self.current_key_index]

    def rotate_key(self) -> str:
        """Rotate to the next key in the list and return it."""
        # This block ensures that the rotation logic is atomic and thread-safe.
        with self.lock:
            # Use the modulo operator to cycle through the keys.
            # (0+1)%3=1, (1+1)%3=2, (2+1)%3=0. This wraps the index back to the start.
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            logger.warning(
                f"API key rotated. Now using key index {self.current_key_index}."
            )
            return self.keys[self.current_key_index]


# Create a single, global instance for the application to use.
# This ensures all parts of the app share the same key manager.
# It also acts as a startup check; if keys aren't configured, the app will fail to start.
api_key_manager = ApiKeyManager()


def with_api_key_rotation(func):
    """
    A decorator that wraps a function making an API call.
    It automatically handles API key rotation on specific exceptions.
    """
    # @wraps ensures the decorated function keeps its original name and docstring.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Loop a number of times equal to the number of available keys.
        # This gives each key one chance to succeed.
        for i in range(len(api_key_manager.keys)):
            try:
                # Attempt to execute the original function (e.g., gemini_chat_stream).
                # That function will internally call `api_key_manager.get_key()`.
                return func(*args, **kwargs)
            except (
                ResourceExhausted,
                PermissionDenied,
                GoogleGenerativeAIError,
            ) as e:
                # If the API call fails with a key-related error, catch it.
                logger.warning(
                    f"API call failed with key index {api_key_manager.current_key_index}. Reason: {e}"
                )
                # If we've already tried the last key in our list, give up and raise the error.
                if i == len(api_key_manager.keys) - 1:
                    logger.error(
                        "All API keys have failed. No more keys to rotate to."
                    )
                    raise e
                # Otherwise, rotate to the next key. The loop will then retry the function call.
                api_key_manager.rotate_key()
        # This line should theoretically not be reached if there's at least one key, but it's a safeguard.
        raise Exception("Failed to execute function after trying all API keys.")

    return wrapper