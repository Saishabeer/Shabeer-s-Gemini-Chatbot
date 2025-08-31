import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# Import the key manager and exceptions from your RAG service
from .rag_service import api_key_manager
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

# Load .env
load_dotenv()

# Use the same environment variable as rag_service for consistency
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def gemini_chat(prompt: str, history: List[dict] = None) -> str:
    """
    Handles general knowledge chat without RAG, with API key rotation and retry logic.
    """
    system_instruction = (
        "You are a helpful and friendly conversational AI. "
        "Answer the user's question clearly and concisely."
    )

    # This loop will retry with the next API key if the current one is exhausted or invalid.
    for _ in range(len(api_key_manager.keys)):
        try:
            # Configure the library with the current key for this attempt
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_instruction
            )

            # Convert our history list to Gemini's format
            gemini_history = []
            for m in history or []:
                role = "model" if m.get("role") == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [m.get("content", "")]})

            chat = model.start_chat(history=gemini_history)
            resp = chat.send_message(prompt)

            return (getattr(resp, "text", "") or "").strip() # Success!

        except (ResourceExhausted, PermissionDenied, InvalidArgument) as e:
            print(f"WARNING: API key ending in '...{api_key_manager.get_current_key()[-4:]}' failed during general chat. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e # Re-raise the exception to be handled by the view
            print("INFO: Switching to the next API key for general chat.")

    # This part is reached if all keys fail
    raise ResourceExhausted("All available API keys failed during general chat.")
