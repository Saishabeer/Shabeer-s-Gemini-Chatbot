import os
import logging
from typing import List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted
from langchain_google_genai._common import GoogleGenerativeAIError

from .rag_service import api_key_manager
from .web_search_service import web_search_manager

# --- Basic Setup ---
logger = logging.getLogger(__name__)
load_dotenv()

# --- Global Instances & Tunables ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _should_search_web(prompt: str) -> bool:
    """Uses heuristics to decide if a web search is necessary."""
    prompt_lower = prompt.lower().strip()
    conversational_starters = ('hi', 'hello', 'hey', 'my name is', 'what is my name', 'thank you', 'thanks', "what's my name")
    if prompt_lower.startswith(conversational_starters) or len(prompt.split()) < 3:
        logger.info(f"Skipping web search for simple/conversational prompt: '{prompt}'")
        return False
    return True


def gemini_chat(prompt: str, history: List[dict] = None) -> Tuple[str, List[str]]:
    """
    Handles general knowledge chat, intelligently deciding whether to fetch web results.
    """
    logger.info(f"-- General Chat Pipeline Initiated --")
    logger.info(f"User prompt: '{prompt}'")

    for _ in range(len(api_key_manager.keys)):
        try:
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            # 1. Web Search
            web_context, web_sources = "", []
            if _should_search_web(prompt) and web_search_manager.is_enabled():
                logger.info(f"Step 1: Performing web search for query: '{prompt}'")
                search_results = web_search_manager.search(prompt)
                web_context = "\n\n".join([f"[WEB-{i + 1}] {r.get('content')}" for i, r in enumerate(search_results)])
                web_sources = [f"[{r.get('title')}]({r.get('url')})" for r in search_results]
                logger.info(f"Web search found {len(search_results)} results.")
            else:
                logger.info("Step 1: Web search skipped based on prompt or settings.")

            # 2. Build Prompt and Generate
            logger.info("Step 2: Building prompt and calling Gemini model.")
            system_instruction = "You are a helpful assistant..."  # Keep short for log

            model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_instruction)
            chat = model.start_chat(history=[{"role": "model" if m.get("role") == "assistant" else "user", "parts": [m.get("content", "")]} for m in history or []])
            resp = chat.send_message(prompt)
            raw_answer = (getattr(resp, "text", "") or "").strip()
            logger.info(f"Model raw response: '{raw_answer[:100]}...'")

            # 3. Parse Response
            logger.info("Step 3: Parsing response.")
            final_answer, sources_to_return = raw_answer, []

            if web_context:
                if "SOURCE: WEB" in raw_answer:
                    final_answer = raw_answer.split("---", 1)[-1].strip()
                    sources_to_return = web_sources
                    logger.info("Source determined: WEB. Answer is grounded in web search results.")
                else:
                    final_answer = raw_answer.split("---", 1)[-1].strip()
                    logger.info("Source determined: KNOWLEDGE. Answer is from model's internal knowledge.")

            logger.info(f"Final Answer: '{final_answer[:100]}...'")
            logger.info("-- General Chat Pipeline Finished --\n")
            return final_answer, sources_to_return

        except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
            logger.warning(f"API key at index {api_key_manager.current_index} failed during general chat. Reason: {type(e).__name__}")
            if not api_key_manager.switch_to_next_key():
                logger.error("All available API keys are invalid or have reached their quota.")
                raise e
            logger.info("Switching to the next API key for general chat.")

    raise ResourceExhausted("All available API keys failed during general chat.")
