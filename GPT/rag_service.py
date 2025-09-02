import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from langchain.schema import Document

from django.conf import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .utils import api_key_manager, with_api_key_rotation

logger = logging.getLogger(__name__)


def get_gemini_embeddings():
    """Initializes and returns the GoogleGenerativeAIEmbeddings instance using the current API key."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key_manager.get_key())


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
def ingest_document_for_session(session_id: int, file_path: str = None):
    """
    Loads a document, adds source metadata, splits it into chunks, generates embeddings,
    and stores them in a persistent Chroma vector store for a specific session.
    If a vector store already exists, it adds the new document chunks to it.

    Args:
        session_id: The ID of the chat session
        file_path: Path to the file or temporary file containing the content
    """
    from .models import ChatSession

    # Get the chat session
    chat_session = ChatSession.objects.get(id=session_id)
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")

    temp_f = None  # Initialize temp file reference
    try:
        # If we have a file path (from API upload), use it.
        # Otherwise, create a temporary file from the database content.
        if file_path and os.path.exists(file_path):
            full_file_path = Path(file_path)
            document_name = full_file_path.name
        else:
            if not chat_session.document_content or not chat_session.document_name:
                raise ValueError("No document content found in the session to process.")
            file_extension = Path(chat_session.document_name).suffix.lower()
            document_name = chat_session.document_name
            # Create a temporary file to store the content for the loaders
            temp_f = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            temp_f.write(chat_session.document_content)
            temp_f.close()  # Close the file so loaders can open it
            full_file_path = Path(temp_f.name)

        # --- Document Loading ---
        if not full_file_path.exists() or os.path.getsize(full_file_path) == 0:
            raise ValueError("The uploaded file is empty or could not be read.")

        file_extension = full_file_path.suffix.lower()
        logger.info(f"Processing file '{document_name}' with extension: {file_extension}")

        if file_extension == '.pdf':
            loader = PyPDFLoader(str(full_file_path))
        elif file_extension == '.txt':
            # Use TextLoader with auto-detection for robustness
            loader = TextLoader(str(full_file_path), autodetect_encoding=True)
        else:
            # Fallback to UnstructuredFileLoader for other types (e.g., .docx, .md)
            loader = UnstructuredFileLoader(str(full_file_path))

        try:
            documents = loader.load()
        except Exception as e:
            logger.error(f"Failed to load document {full_file_path} with loader {type(loader).__name__}: {e}")
            raise ValueError(f"Could not process the file type '{file_extension}'. Please try a different format.")

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("The document was processed, but no text content could be extracted.")

        # --- Add Source Metadata ---
        # This is crucial for identifying the source of retrieved context.
        for doc in documents:
            doc.metadata['source'] = document_name

        # --- Splitting and Ingestion ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.warning(f"Document '{document_name}' resulted in 0 chunks after splitting.")
            return

        embedding_function = get_gemini_embeddings()

        # --- Append to existing vector store or create new one ---
        if has_vectorstore(session_id):
            logger.info(f"Adding {len(chunks)} chunks to existing vector store for session {session_id}.")
            vector_store = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function
            )
            vector_store.add_documents(documents=chunks)
            logger.info(f"Successfully added documents to vector store for session {session_id}.")
        else:
            logger.info(f"Creating new vector store for session {session_id} with {len(chunks)} chunks.")
            Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=vectorstore_path
            )
            logger.info(f"Vector store created successfully for session {session_id} at {vectorstore_path}")

    except Exception as e:
        logger.error(f"Error during document ingestion for session {session_id}: {str(e)}", exc_info=True)
        # Re-raise to be caught by the view and shown to the user
        raise
    finally:
        # Clean up the temporary file if it was created
        if temp_f:
            try:
                os.unlink(str(full_file_path))
            except Exception as e:
                logger.warning(f"Could not delete temporary file {full_file_path}: {e}")


@with_api_key_rotation
def get_rag_context(query: str, session_id: int, top_k: int = 4) -> List[str]:
    """
    Retrieves relevant document chunks from the vector store for a given session and query,
    including source metadata in the context.

    Args:
        query: The user's question.
        session_id: The ID of the chat session.
        top_k: The number of relevant chunks to retrieve.

    Returns:
        A list of formatted strings, where each string contains the content and source
        of a relevant document chunk. Returns an empty list if no context is found.
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

    # --- Format context with source metadata ---
    context_snippets = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown Document')
        # Format the snippet to be clearly understood by the LLM
        snippet = f"Source: {source}\nContent: {doc.page_content}"
        context_snippets.append(snippet)

    logger.info(f"Retrieved {len(context_snippets)} document chunks for session {session_id}.")
    return context_snippets