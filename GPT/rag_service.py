import os
from typing import Tuple, List
import shutil
from django.conf import settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

import google.generativeai as genai
from dotenv import load_dotenv
from .web_search_service import web_search_manager

# --- Load .env and configure Gemini ---
load_dotenv()


# --- API Key Rotation Manager ---
class ApiKeyManager:
    """Manages a pool of API keys, rotating them when one is exhausted."""

    def __init__(self):
        keys_str = os.getenv("GEMINI_API_KEYS")
        if not keys_str:
            raise ValueError("GEMINI_API_KEYS not found in .env. Please provide a comma-separated list of keys.")
        self.keys = [key.strip() for key in keys_str.split(',') if key.strip()]
        if not self.keys:
            raise ValueError(
                "GEMINI_API_KEYS was found but contained no valid keys after parsing. Please check your .env file.")
        self.current_index = 0
        print(f"INFO: Loaded {len(self.keys)} API keys for rotation.")

    def get_current_key(self) -> str:
        """Returns the currently active API key."""
        return self.keys[self.current_index]

    def switch_to_next_key(self) -> bool:
        """Rotates to the next key. Returns False if it has cycled through all keys, otherwise True."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        # A return value of False indicates we've tried every key once in this cycle.
        return self.current_index != 0


# Create a single instance of the manager for the application
api_key_manager = ApiKeyManager()

# Tunables (override via .env if needed)
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")  # free tier
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "5"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# --- Utilities ---
def _session_chroma_dir(session_id: int) -> str:
    base = str(getattr(settings, "CHROMA_DIR", settings.BASE_DIR / "chroma"))
    path = os.path.join(base, f"session_{session_id}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_documents(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md", ".log"):
        loader = TextLoader(file_path, autodetect_encoding=True)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (use PDF/DOCX/TXT)")
    return loader.load()


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key_manager.get_current_key())


# --- Ingestion: split + embed + persist in Chroma ---
def ingest_document_for_session(session_id: int, file_path: str) -> Tuple[str, int]:
    docs = _load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    chroma_dir = _session_chroma_dir(session_id)

    # This loop will retry with the next API key if the current one is exhausted or invalid.
    for _ in range(len(api_key_manager.keys)):
        try:
            # Get embeddings with the current key
            embeddings = _get_embeddings()

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=chroma_dir,
            )

            return chroma_dir, len(chunks)  # Success!

        except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
            print(
                f"WARNING: API key at index {api_key_manager.current_index} (ending in '...{api_key_manager.get_current_key()[-4:]}') failed during ingestion. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e  # Re-raise the exception to be handled by the view
            print("INFO: Switching to the next API key for ingestion.")

    # This part is reached if all keys fail
    raise ResourceExhausted("All available API keys failed during document ingestion.")


def delete_vectorstore_for_session(session_id: int):
    """Deletes the entire ChromaDB directory for a given session to allow for re-indexing."""
    chroma_dir = _session_chroma_dir(session_id)
    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            print(f"Cleaned up vector store for session {session_id}")
        except OSError as e:
            print(f"Error cleaning up vector store for session {session_id}: {e}")


def has_vectorstore(session_id: int) -> bool:
    chroma_dir = _session_chroma_dir(session_id)
    try:
        return any(os.scandir(chroma_dir))
    except FileNotFoundError:
        return False


def _get_vectordb(session_id: int):
    return Chroma(
        embedding_function=_get_embeddings(),
        persist_directory=_session_chroma_dir(session_id)
    )


# --- Retrieval + Gemini generation ---
def rag_answer(question: str, session_id: int, history: List[dict] = None, top_k: int = TOP_K_DEFAULT) -> Tuple[
    str, List[str]]:
    # This loop will retry the request with the next API key if the current one is exhausted.
    for _ in range(len(api_key_manager.keys)):
        try:
            # Configure the library with the current key for this attempt
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            # --- 1. Document Retrieval ---
            vectordb = _get_vectordb(session_id)
            docs = vectordb.similarity_search(question, k=top_k)

            doc_context_blocks = []
            doc_sources = []
            for i, d in enumerate(docs, 1):
                meta = d.metadata or {}
                src = meta.get("source", "uploaded document")
                page = meta.get("page")
                label = f"{os.path.basename(src)}" + (f" • p.{page + 1}" if isinstance(page, int) else "")
                doc_context_blocks.append(f"[{i}] {d.page_content}")
                doc_sources.append(label)
            doc_context = "\n\n".join(doc_context_blocks)

            # --- 2. Web Search ---
            web_context_blocks = []
            web_sources = []
            if web_search_manager.is_enabled():
                print(f"INFO: Performing web search for RAG query: '{question}'")
                search_results = web_search_manager.search(question)
                for i, result in enumerate(search_results, 1):
                    web_context_blocks.append(f"[WEB-{i}] {result.get('content')}")
                    # Format as a Markdown link for the final citation
                    web_sources.append(f"[{result.get('title')}]({result.get('url')})")
            web_context = "\n\n".join(web_context_blocks)

            # --- 3. Combine Contexts and Build Prompt ---
            system_instruction = (
                "You are a helpful assistant. Your response MUST be in two parts, separated by '---'.\n"
                "Part 1: State the source of your answer. It must be one of: `SOURCE: DOCUMENT`, `SOURCE: WEB`, or `SOURCE: KNOWLEDGE`.\n"
                "Part 2: Provide the answer to the user's question. **Do NOT include citations like [1] or [WEB-1] in the answer itself.** The answer should be clean text.\n\n"
                "INSTRUCTIONS:\n"
                "1. First, check the DOCUMENT CONTEXT. If it answers the question, your first line MUST be `SOURCE: DOCUMENT`. Then, after '---', provide the answer.\n"
                "2. If the document doesn't help, check the WEB SEARCH RESULTS. If they answer the question, your first line MUST be `SOURCE: WEB`. Then, after '---', provide the answer.\n"
                "3. If neither context helps, but you know the answer from chat HISTORY or general knowledge, your first line MUST be `SOURCE: KNOWLEDGE`. Then, after '---', provide the answer.\n\n"
                f"DOCUMENT CONTEXT:\n{doc_context}\n\n"
                f"WEB SEARCH RESULTS:\n{web_context}\n\n"
            )

            # Convert our history list to Gemini's format
            gemini_history = []
            for m in history or []:
                role = "model" if m.get("role") == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [m.get("content", "")]})

            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_instruction
            )

            # Start a chat session with the history and send the new question
            chat = model.start_chat(history=gemini_history)
            resp = chat.send_message(question)

            raw_answer = (getattr(resp, "text", "") or "").strip()

            # --- 4. Parse Response ---
            # This robustly parses the model's output to separate the answer from the source header.
            final_answer = raw_answer
            sources_to_return = []

            answer_parts = raw_answer.split('---', 1)
            if len(answer_parts) == 2 and "SOURCE:" in answer_parts[0]:
                header = answer_parts[0].strip()
                final_answer = answer_parts[1].strip()
                if header == "SOURCE: DOCUMENT":
                    sources_to_return = doc_sources
                elif header == "SOURCE: WEB":
                    sources_to_return = web_sources
            else:
                # Fallback if '---' is missing. Check for a header at the start of the string.
                if raw_answer.startswith("SOURCE: DOCUMENT"):
                    final_answer = raw_answer.replace("SOURCE: DOCUMENT", "", 1).strip()
                    sources_to_return = doc_sources
                elif raw_answer.startswith("SOURCE: WEB"):
                    final_answer = raw_answer.replace("SOURCE: WEB", "", 1).strip()
                    sources_to_return = web_sources
                elif raw_answer.startswith("SOURCE: KNOWLEDGE"):
                    final_answer = raw_answer.replace("SOURCE: KNOWLEDGE", "", 1).strip()

            return final_answer, sources_to_return

        except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
            print(f"WARNING: API key at index {api_key_manager.current_index} (ending in '...{api_key_manager.get_current_key()[-4:]}') failed during RAG. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e  # Re-raise the exception to be handled by the view
            print("INFO: Switching to the next API key for RAG answer.")

    # This part should not be reached unless something went wrong, but it's a safe fallback.
    raise ResourceExhausted("All available API keys have been tried and failed.")
