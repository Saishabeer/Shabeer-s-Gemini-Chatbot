import os
import logging
import shutil
from typing import List, Tuple, Iterable

import google.generativeai as genai
from django.conf import settings
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .web_search_service import web_search_manager

# --- Basic Setup ---
logger = logging.getLogger(__name__)
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
        logger.info(f"Loaded {len(self.keys)} API keys for rotation.")

    def get_current_key(self) -> str:
        """Returns the currently active API key."""
        return self.keys[self.current_index]

    def switch_to_next_key(self) -> bool:
        """Rotates to the next key. Returns False if it has cycled through all keys, otherwise True."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.current_index != 0


# --- Global Instances & Tunables ---
api_key_manager = ApiKeyManager()
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
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
    logger.info(f"Loading document with extension: {ext}")
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


# --- Core RAG Service Functions ---
def ingest_document_for_session(session_id: int, file_path: str) -> Tuple[str, int]:
    logger.info(f"[RAG] Starting ingestion for session {session_id}, file: {file_path}")
    docs = _load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[RAG] Document split into {len(chunks)} chunks.")
    chroma_dir = _session_chroma_dir(session_id)

    for _ in range(len(api_key_manager.keys)):
        try:
            embeddings = _get_embeddings()
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=chroma_dir,
            )
            logger.info(f"[RAG] Ingestion complete. Vector store created at {chroma_dir} with {len(chunks)} chunks.")
            return chroma_dir, len(chunks)

        except (ResourceExhausted, PermissionDenied, InvalidArgument) as e:
            logger.warning(
                f"API key at index {api_key_manager.current_index} failed during ingestion. Reason: {type(e).__name__}")
            if not api_key_manager.switch_to_next_key():
                logger.error("All available API keys are invalid or have reached their quota.")
                raise e
            logger.info("Switching to the next API key for ingestion.")

    raise ResourceExhausted("All available API keys failed during document ingestion.")


def delete_vectorstore_for_session(session_id: int):
    """Deletes the ChromaDB directory for a session."""
    chroma_dir = _session_chroma_dir(session_id)
    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            logger.info(f"Cleaned up vector store for session {session_id}")
        except OSError as e:
            logger.error(f"Error cleaning up vector store for session {session_id}: {e}")


def has_vectorstore(session_id: int) -> bool:
    chroma_dir = _session_chroma_dir(session_id)
    try:
        return any(os.scandir(chroma_dir))
    except FileNotFoundError:
        return False


def _get_vectordb(session_id: int):
    return Chroma(embedding_function=_get_embeddings(), persist_directory=_session_chroma_dir(session_id))


def rag_answer_stream(question: str, session_id: int, history: List[dict] = None, top_k: int = TOP_K_DEFAULT) -> \
Iterable[str]:
    logger.info(f"-- RAG Pipeline (Stream) Initiated for Session {session_id} --")
    logger.info(f"User prompt: '{question}'")

    for _ in range(len(api_key_manager.keys)):
        try:
            current_key = api_key_manager.get_current_key()
            genai.configure(api_key=current_key)

            # 1. Document Retrieval (Happens before streaming)
            logger.info(f"Step 1: Retrieving top {top_k} document chunks from knowledge base for streaming.")
            vectordb = _get_vectordb(session_id)
            docs = vectordb.similarity_search(question, k=top_k)
            doc_context = "\n\n".join([f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)])
            doc_sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]
            logger.info(f"Retrieved {len(docs)} chunks. Sources: {doc_sources}")

            # 2. Web Search (Happens before streaming)
            web_context, web_sources = "", []
            if web_search_manager.is_enabled():
                logger.info(f"Step 2: Performing web search for query: '{question}'")
                search_results = web_search_manager.search(question)
                web_context = "\n\n".join([f"[WEB-{i + 1}] {r.get('content')}" for i, r in enumerate(search_results)])
                web_sources = [f"[{r.get('title')}]({r.get('url')})" for r in search_results]
                logger.info(f"Web search found {len(search_results)} results.")
            else:
                logger.info("Step 2: Web search is disabled.")

            # 3. Build Prompt and Generate Stream
            logger.info("Step 3: Building prompt and calling Gemini model in streaming mode.")
            system_instruction = (
                "You are a helpful assistant..."  # Keeping this short for the log
            )

            model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_instruction)
            chat = model.start_chat(
                history=[{"role": "model" if m.get("role") == "assistant" else "user", "parts": [m.get("content", "")]}
                         for m in history or []])

            response_stream = chat.send_message(question, stream=True)

            # Yield each chunk of the main response as it arrives.
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

            # 4. Deterministically Yield Citations
            # After the main response, append the sources we know we used.
            logger.info("Step 4: Determining and yielding citations.")
            sources_to_return = []
            if doc_sources:
                sources_to_return.extend(list(set(doc_sources)))
            if web_sources:
                sources_to_return.extend(web_sources)

            if sources_to_return:
                unique_sources = sorted(list(set(sources_to_return)))
                source_header = "\n\n**Source:**\n"
                source_list = "- " + "\n- ".join(unique_sources)
                yield source_header + source_list

            logger.info("-- RAG Pipeline (Stream) Finished --\n")
            return  # Stop the generator

        except (ResourceExhausted, PermissionDenied, InvalidArgument) as e:
            logger.warning(
                f"API key at index {api_key_manager.current_index} failed during RAG. Reason: {type(e).__name__}")
            if not api_key_manager.switch_to_next_key():
                logger.error("All available API keys are invalid or have reached their quota.")
                raise e
            logger.info("Switching to the next API key for RAG answer.")

    raise ResourceExhausted("All available API keys have been tried and failed.")
