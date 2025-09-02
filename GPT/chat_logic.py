import logging
from typing import Tuple, List, Dict

from .gemini_service import gemini_chat_stream
from .models import ChatSession
from .rag_service import get_rag_context, has_vectorstore
from .web_search_service import web_search_manager

logger = logging.getLogger(__name__)

GREETINGS = {"hi", "hello", "hlo", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"}


def process_chat_prompt(session: ChatSession, prompt: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Processes a user prompt by performing query analysis, rewriting, RAG, and building the final prompt.

    Returns:
        A tuple containing (final_prompt, gemini_history).
    """
    is_simple_query = prompt.lower().strip() in GREETINGS
    search_query = prompt

    # --- Query Rewriting (for follow-up questions) ---
    if not is_simple_query and history:
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in history[-6:]])
        rewrite_prompt = f"""You are an expert at rephrasing a follow-up question into a standalone question, using the provided chat history. Do not answer the question, just provide the rephrased, standalone question.

**Example 1:**
Chat History:
user: Who is the CEO of Tesla?
assistant: Elon Musk is the CEO of Tesla.
Follow-up Question: What other companies does he run?
Standalone Question: What other companies does Elon Musk run?

**Your Task:**

Chat History:
{history_text}

Follow-up Question: {prompt}

Standalone Question:"""
        try:
            rewriter_stream = gemini_chat_stream(rewrite_prompt, history=[])
            rewritten_query = "".join(list(rewriter_stream)).strip().replace('"', '')
            if rewritten_query and '\n' not in rewritten_query:
                search_query = rewritten_query
            logger.info(f"Rewritten query: {search_query}")
        except Exception as e:
            logger.error(f"Query rewrite failed, using original prompt. Error: {e}")

    # --- Information Retrieval ---
    doc_context, web_context = "", ""
    if not is_simple_query:
        if has_vectorstore(session.id):
            doc_snippets = get_rag_context(search_query, session.id)
            if doc_snippets:
                doc_context = "\n\n".join(doc_snippets)
        if web_search_manager.is_enabled():
            web_results = web_search_manager.search(search_query)
            web_context = "\n\n".join([r.get('content', '') for r in web_results if r.get('content')])

    # --- Build Final Prompt ---
    if doc_context or web_context:
        system_instruction = "You are a helpful assistant. Use the provided context to answer the user's question."
        context_parts = [system_instruction]
        if doc_context:
            context_parts.append(f"--- DOCUMENT CONTEXT ---\n{doc_context}")
        if web_context:
            context_parts.append(f"--- WEB SEARCH CONTEXT ---\n{web_context}")
        final_prompt = "\n\n".join(context_parts) + f"\n\n--- USER QUESTION ---\n{prompt}"
    else:
        final_prompt = prompt

    gemini_history = [{'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [{'text': msg['content']}]} for msg in history]

    return final_prompt, gemini_history