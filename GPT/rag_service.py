import logging
import shutil
from pathlib import Path
from typing import List

from django.conf import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .utils import api_key_manager, with_api_key_rotation

logger = logging.getLogger(__name__)


def get_gemini_embeddings():
    """Initializes and returns the GoogleGenerativeAIEmbeddings instance using the current API key."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key_manager.get_current_key())


def has_vectorstore(session_id: int) -> bool:
    """Checks if a vector store directory exists for the given session."""
    vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
    return vectorstore_path.exists() and any(vectorstore_path.iterdir())


def delete_vectorstore_for_session(session_id: int):
    """Deletes the Chroma vector store directory for a given session."""
    if has_vectorstore(session_id):
        vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
        try:
            shutil.rmtree(vectorstore_path)
            logger.info(f"Successfully deleted vector store for session {session_id}.")
        except OSError as e:
            logger.error(f"Error deleting vector store for session {session_id}: {e}", exc_info=True)


@with_api_key_rotation
def ingest_document_for_session(session_id: int, file_path: str):
    """
    Loads a document, splits it into chunks, generates embeddings,
    and stores them in a persistent Chroma vector store for a specific session.
    """
    full_file_path = settings.MEDIA_ROOT / file_path
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")

    # Choose loader based on file type
    file_extension = Path(full_file_path).suffix.lower()
    if file_extension == '.pdf':
        loader = PyPDFLoader(str(full_file_path))
    elif file_extension == '.txt':
        loader = TextLoader(str(full_file_path))
    else:
        # Fallback for other types like .doc, .docx, etc.
        loader = UnstructuredFileLoader(str(full_file_path))

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embedding_function = get_gemini_embeddings()

    logger.info(f"Creating vector store for session {session_id} with {len(chunks)} chunks.")
    Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=vectorstore_path
    )
    logger.info(f"Vector store created successfully for session {session_id} at {vectorstore_path}")


@with_api_key_rotation
def get_rag_context(query: str, session_id: int, top_k: int = 4) -> List[str]:
    """
    Retrieves relevant document chunks from the vector store for a given session and query.

    Args:
        query: The user's question.
        session_id: The ID of the chat session.
        top_k: The number of relevant chunks to retrieve.

    Returns:
        A list of strings, where each string is the content of a relevant document chunk.
        Returns an empty list if no vector store exists or no documents are found.
    """
    if not has_vectorstore(session_id):
        logger.debug(f"No vectorstore found for session {session_id}. Skipping RAG context retrieval.")
        return []

    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")
    embedding_function = get_gemini_embeddings()

    vector_store = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(query)

    logger.info(f"Retrieved {len(relevant_docs)} document chunks for session {session_id}.")
    return [doc.page_content for doc in relevant_docs]