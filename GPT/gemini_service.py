import logging
import google.generativeai as genai
from typing import Iterable, List, Dict

from .utils import api_key_manager, with_api_key_rotation

logger = logging.getLogger(__name__)

# --- Model Configuration ---
# These settings can be adjusted to control the model's behavior.
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


@with_api_key_rotation
def gemini_chat_stream(prompt: str, history: List[Dict]) -> Iterable[str]:
    """
    Generates content from the Gemini model with streaming and API key rotation.
    This function is decorated to automatically handle API key errors.
    """
    logger.info("Attempting to generate content with Gemini API.")
    
    # Get the current API key from the manager
    current_key = api_key_manager.get_key()
    genai.configure(api_key=current_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(prompt, stream=True)

    for chunk in response:
        if chunk.text:
            yield chunk.text