import logging
from typing import Dict, Iterable, List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .utils import api_key_manager, with_api_key_rotation

logger = logging.getLogger(__name__)

# Configure safety settings to be less restrictive for a general-purpose chatbot.
# Adjust these as needed for your specific use case.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


@with_api_key_rotation
def gemini_chat_stream(prompt: str, history: List[Dict[str, str]]) -> Iterable[str]:
    """
    Generates a response from the Gemini model in a stream, with automatic API key rotation.

    This function is decorated with `with_api_key_rotation`, which handles retries
    with different API keys upon encountering quota or permission errors.
    """
    logger.info("Attempting to generate content with Gemini API.")
    
    # This function will be re-executed by the decorator on failure,
    # so we get the current (potentially new) key on each attempt.
    current_key = api_key_manager.get_current_key()
    genai.configure(api_key=current_key)

    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=SAFETY_SETTINGS)
    chat = model.start_chat(history=history)
    response_stream = chat.send_message(prompt, stream=True)

    for chunk in response_stream:
        yield chunk.text