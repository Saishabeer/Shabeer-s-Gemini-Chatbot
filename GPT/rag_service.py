import os
from typing import Tuple, List
import shutil
from django.conf import settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai
from dotenv import load_dotenv

# --- Load .env and configure Gemini ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

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
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)

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

    embeddings = _get_embeddings()
    chroma_dir = _session_chroma_dir(session_id)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vectordb.persist()
    return chroma_dir, len(chunks)

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
def rag_answer(question: str, session_id: int, top_k: int = TOP_K_DEFAULT) -> Tuple[str, List[str]]:
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
    prompt = (
        "You are a helpful assistant. Your primary goal is to answer the user's question based on the provided CONTEXT.\n"
        "1. First, search the CONTEXT for the answer. If you find the answer, respond with it and cite the relevant sources like [1], [2], etc.\n"
        "2. If the answer is NOT found in the CONTEXT, you MUST state 'I don't have that information in the provided knowledge base, but I can answer using my general knowledge.' and then proceed to answer the question based on your own knowledge.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer:"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    answer = (getattr(resp, "text", "") or "").strip()
    return answer, sources
