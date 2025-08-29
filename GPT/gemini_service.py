import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env and configure SDK
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# Choose your model here (flash is faster/cheaper; pro is stronger)
_GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"

def _to_gemini_messages(history_list, user_prompt):
    """
    Convert our history + current prompt into Gemini's expected format.
    history_list: [{"role": "user"|"assistant", "content": "..."}, ...]
    """
    msgs = []
    for m in history_list or []:
        role = "model" if m.get("role") == "assistant" else "user"
        msgs.append({"role": role, "parts": [m.get("content", "")]})
    # Append current prompt as a new user turn
    if user_prompt:
        msgs.append({"role": "user", "parts": [user_prompt]})
    return msgs

def gemini_chat(prompt, history=None):
    """
    Generate a reply from Gemini given current prompt and prior history.
    - prompt: str (current user input)
    - history: list of dicts with keys: role ('user'|'assistant') and content (str)
    Returns: str (model reply text)
    """
    try:
        messages = _to_gemini_messages(history or [], prompt)
        model = genai.GenerativeModel(_GEMINI_MODEL)
        resp = model.generate_content(messages)
        # response.text is convenient; handle empty gracefully
        return (resp.text or "").strip() if hasattr(resp, "text") else ""
    except Exception as e:
        # Surface a readable error back to UI
        return f"⚠️ Error contacting Gemini: {str(e)}"
