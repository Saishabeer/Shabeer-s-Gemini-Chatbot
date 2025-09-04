# --- Python Standard Library Imports ---
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List
# --- Django Core Imports ---
from django.conf import settings
# --- Third-Party Library Imports (LangChain) ---
# Document: A standard object to hold a piece of text and its metadata.
from langchain.schema import Document
# RecursiveCharacterTextSplitter: A smart text splitter that tries to keep related text together.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Document Loaders: Classes to load text from various file formats.
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
# Chroma: The vector database used to store and search document embeddings.
from langchain_community.vectorstores import Chroma
# GoogleGenerativeAIEmbeddings: The client for creating text embeddings with Google's models.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# --- Local Application Imports ---
# Import the API key manager and rotation decorator for resilient API calls.
from .utils import api_key_manager, with_api_key_rotation

# Get a logger instance for this file.
logger = logging.getLogger(__name__)


def get_gemini_embeddings():
    """Initializes and returns the GoogleGenerativeAIEmbeddings instance using the current API key."""
    # This is a factory function to consistently create the embeddings client.
    # It correctly fetches the current API key from our rotating key manager.
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key_manager.get_key())


def has_vectorstore(session_id: int) -> bool:
    """Checks if a vector store directory exists for the given session."""
    # Construct the expected path for the session's vector store.
    vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
    # Return True only if the directory exists AND it's not empty.
    return vectorstore_path.exists() and any(vectorstore_path.iterdir())


def delete_vectorstore_for_session(session_id: int):
    """Deletes the Chroma vector store directory for a given session."""
    # First, check if there's anything to delete.
    if has_vectorstore(session_id):
        vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
        try:
            # `shutil.rmtree` recursively deletes the directory and all its contents.
            shutil.rmtree(vectorstore_path)
            logger.info(f"Successfully deleted vector store for session {session_id}.")
        except OSError as e:
            logger.error(f"Error deleting vector store for session {session_id}: {e}", exc_info=True)

# This decorator ensures that if embedding creation fails due to an API key issue,
# it will automatically rotate the key and retry the function.
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
    # Import models locally to avoid circular import issues.
    from .models import ChatSession

    # 1. SETUP: Get the session object and define the path for its vector store.
    chat_session = ChatSession.objects.get(id=session_id)
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")

    temp_f = None  # Initialize temp file reference
    try:
        # 2. FILE HANDLING: Get the document content into a file on disk that loaders can read.
        # If we have a file path (from API upload), use it.
        # Otherwise, create a temporary file from the database content.
        if file_path and os.path.exists(file_path):
            full_file_path = Path(file_path)
            document_name = full_file_path.name
        else:
            # This branch handles uploads from the web UI where the file content is in the database.
            if not chat_session.document_content or not chat_session.document_name:
                raise ValueError("No document content found in the session to process.")
            file_extension = Path(chat_session.document_name).suffix.lower()
            document_name = chat_session.document_name
            # Create a temporary file to store the content for the loaders
            temp_f = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            temp_f.write(chat_session.document_content)
            temp_f.close()  # Close the file so loaders can open it
            full_file_path = Path(temp_f.name)

        # 3. DOCUMENT LOADING: Select the correct LangChain loader based on the file extension.
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

        # 4. ADD METADATA: Add the original filename as 'source' metadata to each document chunk.
        # This is crucial for identifying the source of retrieved context.
        for doc in documents:
            doc.metadata['source'] = document_name

        # 5. SPLITTING: Break the loaded document(s) into smaller, manageable chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.warning(f"Document '{document_name}' resulted in 0 chunks after splitting.")
            return

        # 6. EMBEDDING & STORAGE: Convert chunks to vectors and save them in ChromaDB.
        embedding_function = get_gemini_embeddings()

        # Check if a store already exists for this session.
        if has_vectorstore(session_id):
            # If it exists, add the new document chunks to it. This allows for multi-doc chats.
            logger.info(f"Adding {len(chunks)} chunks to existing vector store for session {session_id}.")
            vector_store = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function
            )
            vector_store.add_documents(documents=chunks)
            logger.info(f"Successfully added documents to vector store for session {session_id}.")
        else:
            # If it's the first document, create a new vector store.
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
        # 7. CLEANUP: This block always runs, ensuring the temporary file is deleted.
        if temp_f:
            try:
                os.unlink(str(full_file_path))
            except Exception as e:
                logger.warning(f"Could not delete temporary file {full_file_path}: {e}")

# This decorator handles API key rotation if embedding the *query* fails.
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
    # If no vector store exists for the session, there's nothing to search.
    if not has_vectorstore(session_id):
        logger.debug(f"No vectorstore found for session {session_id}. Skipping RAG context retrieval.")
        return []

    # 1. SETUP: Define the path and get the embedding function.
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")
    embedding_function = get_gemini_embeddings()

    # 2. LOAD: Connect to the persistent ChromaDB vector store on disk.
    vector_store = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )

    # 3. RETRIEVE: Create a retriever and find the most relevant documents.
    # `as_retriever` creates a standard LangChain interface for searching.
    # `search_kwargs={"k": top_k}` tells it to find the top 'k' most similar chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(query)

    # 4. FORMAT: Prepare the retrieved context for the LLM.
    context_snippets = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown Document')
        # Format the snippet to be clearly understood by the LLM
        snippet = f"Source: {source}\nContent: {doc.page_content}"
        context_snippets.append(snippet)

    logger.info(f"Retrieved {len(context_snippets)} document chunks for session {session_id}.")
    return context_snippets