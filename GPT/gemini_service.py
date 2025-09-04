# --- Python Standard Library Imports ---
import logging
# --- Third-Party Library Imports ---
import google.generativeai as genai
from typing import Iterable, List, Dict

# --- Local Application Imports ---
# Import the API key manager and the rotation decorator from our utils file.
from .utils import api_key_manager, with_api_key_rotation

# Get a logger instance for this file.
logger = logging.getLogger(__name__)

# --- Model Configuration ---
# These settings can be adjusted to control the model's behavior.
generation_config = {
    # Controls randomness. Lower is more deterministic, higher is more creative.
    "temperature": 0.9,
    # Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds top_p.
    "top_p": 1,
    # Top-k sampling: selects the next token from the top k most likely tokens.
    "top_k": 1,
    # The maximum number of tokens to generate in the response.
    "max_output_tokens": 8192,
}

# Configures the content safety filters for the API.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


# This decorator automatically wraps the function in a try/except block.
# If an API key fails, it will rotate to the next one and retry the function.
@with_api_key_rotation
def gemini_chat_stream(prompt: str, history: List[Dict]) -> Iterable[str]:
    """
    Generates content from the Gemini model with streaming and API key rotation.
    This function is decorated to automatically handle API key errors.
    """
    logger.info("Attempting to generate content with Gemini API.")  

    # 1. Get the current, active API key from our key manager.
    current_key = api_key_manager.get_key()
    # 2. Configure the `genai` library with this key for the upcoming request.
    genai.configure(api_key=current_key)

    # 3. Initialize the generative model with our desired settings.
    model = genai.GenerativeModel(
        # Specify the model to use. 'flash' models are optimized for speed.
        model_name="gemini-2.5-flash-lite",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # 4. Start a conversational chat session, providing the previous messages for context.
    chat_session = model.start_chat(history=history)
    # 5. Send the new prompt to the model. `stream=True` is essential for the typing effect.
    # It tells the API to send back the response in chunks as it's generated.
    response = chat_session.send_message(prompt, stream=True)

    # 6. Loop through the streaming response from the API.
    for chunk in response:
        # The response object may contain other data, so we check for the text content.
        if chunk.text:
            # `yield` sends the piece of text back to the calling function (in views.py)
            # and pauses, waiting for the next chunk. This is what makes streaming possible.
            yield chunk.text