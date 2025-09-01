import os
import logging
from typing import List, Iterable

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted
from langchain_google_genai._common import GoogleGenerativeAIError

from .utils import api_key_manager, with_api_key_rotation
from .web_search_service import web_search_manager

# --- Basic Setup ---
logger = logging.getLogger(__name__)
load_dotenv()

# --- Global Instances & Tunables ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _should_search_web(prompt: str) -> bool:
    """Uses heuristics to decide if a web search is necessary."""
    prompt_lower = prompt.lower().strip()
    conversational_starters = ('hi', 'hello', 'hey', 'my name is', 'what is my name', 'thank you', 'thanks',
                               "what's my name")
    if prompt_lower.startswith(conversational_starters) or len(prompt.split()) < 3:
        logger.info(f"Skipping web search for simple/conversational prompt: '{prompt}'")
        return False
    return True


@with_api_key_rotation
def gemini_chat_stream(prompt: str, history: List[dict] = None) -> Iterable[str]:
    """
    Handles general knowledge chat, streaming the response chunk by chunk.
    """
    logger.info(f"-- General Chat (Stream) Pipeline Initiated --")
    logger.info(f"User prompt: '{prompt}'")

    current_key = api_key_manager.get_current_key()
    genai.configure(api_key=current_key)

    # Web search is performed first and its context is fully gathered before streaming.
    web_context, web_sources = "", []
    if _should_search_web(prompt) and web_search_manager.is_enabled():
        logger.info(f"Step 1: Performing web search for query: '{prompt}'")
        search_results = web_search_manager.search(prompt)
        web_context = "\n\n".join([f"[WEB-{i + 1}] {r.get('content')}" for i, r in enumerate(search_results)])
        web_sources = [f"[{r.get('title')}]({r.get('url')})" for r in search_results]
        logger.info(f"Web search found {len(search_results)} results.")
    else:
        logger.info("Step 1: Web search skipped based on prompt or settings.")

    logger.info("Step 2: Building prompt and calling Gemini model in streaming mode.")
    system_instruction = "You are a helpful assistant..."  # Keep short for log

    model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_instruction)
    chat = model.start_chat(
        history=[{"role": "model" if m.get("role") == "assistant" else "user", "parts": [m.get("content", "")]}
                 for m in history or []])

    # Use stream=True to get response chunks
    response_stream = chat.send_message(prompt, stream=True)

    # Yield each chunk as it arrives
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text

    # After the content stream, yield the sources if any were found
    if web_sources:
        unique_srcs = sorted(list(set(s for s in web_sources if s)))
        source_header = "\n\n**Source:**\n"
        source_list = "- " + "\n- ".join(unique_srcs)
        yield source_header + source_list

    logger.info("-- General Chat (Stream) Pipeline Finished --\n")
