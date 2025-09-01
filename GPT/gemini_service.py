import os
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai._common import GoogleGenerativeAIError

# Import the key manager and exceptions from your RAG service
from .web_search_service import web_search_manager
from .rag_service import api_key_manager
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

# Load .env
load_dotenv()

# Use the same environment variable as rag_service for consistency
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _should_search_web(prompt: str) -> bool:
    """
    Uses the LLM to quickly decide if a web search is necessary for a given prompt.
    This avoids unnecessary web searches for simple conversational questions.
    """
    # --- Heuristic Improvement ---
    # Check for common conversational patterns to avoid an unnecessary LLM call.
    # This is more robust than just checking the length.
    prompt_lower = prompt.lower().strip()
    conversational_starters = ('hi', 'hello', 'hey', 'my name is', 'what is my name', 'thank you', 'thanks', "what's my name")
    if prompt_lower.startswith(conversational_starters):
        print(f"INFO: Skipping web search for conversational prompt: '{prompt}'")
        return False

    try:
        # Use a simple, constrained prompt for the decision-making.
        router_model = genai.GenerativeModel(GEMINI_MODEL)

        router_prompt = (
            "You are an intelligent search query classifier. Your task is to determine if a user's query "
            "requires a real-time web search to be answered accurately. "
            "Answer with only a single word: 'YES' or 'NO'.\n\n"
            "== Reasons to say YES (search the web) ==\n"
            "1. The query is about recent events, news, or current affairs (e.g., 'latest movie releases', 'today's weather').\n"
            "2. The query asks for specific, factual information about a person, place, or thing that might have recent updates (e.g., 'who is the CEO of OpenAI?', 'what is Thalapathy Vijay's upcoming movie?').\n"
            "3. The query asks for code or technical instructions involving a **specific, named library, framework, or API** (e.g., 'how to use pandas read_csv', 'django authentication example', 'google maps api key setup').\n\n"
            "== Reasons to say NO (do NOT search the web) ==\n"
            "1. The query is a simple greeting, conversational filler, or a personal question (e.g., 'hi', 'how are you?', 'what is my name?').\n"
            "2. The query is a **general programming or algorithmic question** that does not require knowledge of a specific, named library (e.g., 'write a python function to find duplicates', 'how does bubble sort work?', 'javascript for loop syntax').\n"
            "3. The query is a broad, philosophical, or creative request (e.g., 'what is the meaning of life?', 'write a poem about the sea').\n\n"
            f"User Query: \"{prompt}\"\n\n"
            "Requires Web Search? (YES/NO):"
        )

        response = router_model.generate_content(router_prompt)
        decision = (getattr(response, "text", "") or "").strip().upper()

        print(f"INFO: Web search router decision for '{prompt}': {decision}")

        return "YES" in decision

    except Exception as e:
        # If the router fails for any reason, it's safer to default to not searching to save resources.
        print(f"WARNING: Web search router failed: {e}. Defaulting to NO search.")
        return False


def gemini_chat(prompt: str, history: List[dict] = None) -> Tuple[str, List[str]]:
    """
    Handles general knowledge chat. Intelligently decides whether to fetch
    web results to provide up-to-date answers.
    Includes API key rotation and retry logic.
    """
    # This loop will retry with the next API key if the current one is exhausted or invalid.
    for _ in range(len(api_key_manager.keys)):
        try:
            # Configure the library with the current key for this attempt
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            # --- 1. Web Search ---
            web_context_blocks = []
            web_sources = []
            # Use the router to make an intelligent decision on whether to search.
            if _should_search_web(prompt) and web_search_manager.is_enabled():
                print(f"INFO: Performing web search for general chat: '{prompt}'")
                search_results = web_search_manager.search(prompt)
                for i, result in enumerate(search_results, 1):
                    web_context_blocks.append(f"[WEB-{i}] {result.get('content')}")
                    web_sources.append(f"[{result.get('title')}]({result.get('url')})")
            web_context = "\n\n".join(web_context_blocks)

            # --- 2. Build Prompt ---
            if web_context:
                system_instruction = (
                    "You are a helpful assistant. Your response MUST be in two parts, separated by '---'.\n"
                    "Part 1: State the source of your answer. It must be one of: `SOURCE: WEB` or `SOURCE: KNOWLEDGE`.\n"
                    "Part 2: Provide the answer to the user's question. **Do NOT include citations like [WEB-1] in the answer itself.** The answer should be clean text.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Examine the WEB SEARCH RESULTS. If they contain the answer, your first line MUST be `SOURCE: WEB`. Then, after '---', provide the answer.\n"
                    "2. If the web results are not relevant, but you know the answer from chat HISTORY or general knowledge, your first line MUST be `SOURCE: KNOWLEDGE`. Then, after '---', provide the answer.\n\n"
                    f"WEB SEARCH RESULTS:\n{web_context}\n\n"
                )
            else:  # No web search enabled or no results
                system_instruction = (
                    "You are a helpful and friendly conversational AI. "
                    "Answer the user's question clearly and concisely."
                )

            # --- 3. Call API ---
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

            raw_answer = (getattr(resp, "text", "") or "").strip()

            # --- 4. Parse Response ---
            # This robustly parses the model's output to separate the answer from the source header.
            final_answer = raw_answer
            sources_to_return = []

            if web_context:
                answer_parts = raw_answer.split('---', 1)
                if len(answer_parts) == 2 and "SOURCE:" in answer_parts[0]:
                    header = answer_parts[0].strip()
                    final_answer = answer_parts[1].strip()
                    if header == "SOURCE: WEB":
                        sources_to_return = web_sources
                else:
                    # Fallback if '---' is missing. Check for a header at the start of the string.
                    if raw_answer.startswith("SOURCE: WEB"):
                        final_answer = raw_answer.replace("SOURCE: WEB", "", 1).strip()
                        sources_to_return = web_sources
                    elif raw_answer.startswith("SOURCE: KNOWLEDGE"):
                        final_answer = raw_answer.replace("SOURCE: KNOWLEDGE", "", 1).strip()

            return final_answer, sources_to_return

        except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
            print(f"WARNING: API key at index {api_key_manager.current_index} (ending in '...{api_key_manager.get_current_key()[-4:]}') failed during general chat. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e # Re-raise the exception to be handled by the view
            print("INFO: Switching to the next API key for general chat.")

    # This part is reached if all keys fail
    raise ResourceExhausted("All available API keys failed during general chat.")
