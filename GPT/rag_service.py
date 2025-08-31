import os
from typing import Tuple, List
import shutil
from django.conf import settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

import google.generativeai as genai
from dotenv import load_dotenv

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

            return chroma_dir, len(chunks) # Success!

        except (ResourceExhausted, PermissionDenied, InvalidArgument) as e:
            print(f"WARNING: API key ending in '...{api_key_manager.get_current_key()[-4:]}' failed during ingestion. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e # Re-raise the exception to be handled by the view
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
def rag_answer(question: str, session_id: int, history: List[dict] = None, top_k: int = TOP_K_DEFAULT) -> Tuple[str, List[str]]:
    # This loop will retry the request with the next API key if the current one is exhausted.
    for _ in range(len(api_key_manager.keys)):
        try:
            # Configure the library with the current key for this attempt
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            vectordb = _get_vectordb(session_id)
            docs = vectordb.similarity_search(question, k=top_k)

            context_blocks = []
            sources = []
            for i, d in enumerate(docs, 1):
                meta = d.metadata or {}
                src = meta.get("source", "uploaded document")
                page = meta.get("page")
                label = f"{os.path.basename(src)}" + (f" • p.{page+1}" if isinstance(page, int) else "")
                context_blocks.append(f"[{i}] {d.page_content}")
                sources.append(label)

            context = "\n\n".join(context_blocks)
            
            system_instruction = (
                "You are a helpful assistant. Your primary goal is to answer the user's question based *exclusively* on the provided CONTEXT.\n"
                "1. Read the user's QUESTION and the chat HISTORY to understand the full query.\n"
                "2. Examine the CONTEXT provided. If the CONTEXT contains enough information to fully and directly answer the user's question, then generate a comprehensive answer based *only* on the CONTEXT and cite the sources like [1], [2], etc.\n"
                "3. If the CONTEXT does *not* contain enough information to answer the question, you MUST start your response with the exact phrase 'I don't have that information in the provided knowledge base, but I can answer using my general knowledge.' and then proceed to answer the question using your own general knowledge.\n\n"
                f"CONTEXT:\n{context}\n\n"
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

            answer = (getattr(resp, "text", "") or "").strip()
            return answer, sources # Success! Exit the function.

        except (ResourceExhausted, PermissionDenied, InvalidArgument) as e:
            print(f"WARNING: API key ending in '...{api_key_manager.get_current_key()[-4:]}' failed during RAG. Reason: {type(e).__name__}")
            can_switch = api_key_manager.switch_to_next_key()
            if not can_switch:
                print("ERROR: All available API keys are invalid or have reached their quota.")
                raise e # Re-raise the exception to be handled by the view
            print("INFO: Switching to the next API key for RAG answer.")

    # This part should not be reached unless something went wrong, but it's a safe fallback.
    raise ResourceExhausted("All available API keys have been tried and failed.")
