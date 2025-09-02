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
def ingest_document_for_session(session_id: int, file_path: str):
    """
    Loads a document, splits it into chunks, generates embeddings,
    and stores them in a persistent Chroma vector store for a specific session.
    
    Args:
        session_id: The ID of the chat session
        file_path: Path to the file or temporary file containing the content
    """
    from .models import ChatSession
    
    # Get the chat session
    chat_session = ChatSession.objects.get(id=session_id)
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")
    
    # If we have a file path (temporary file), use it, otherwise use the content from database
    if file_path and os.path.exists(file_path):
        # This is a temporary file path
        full_file_path = Path(file_path)
        file_extension = full_file_path.suffix.lower()
    else:
        # Get file extension from the stored document name
        file_extension = Path(chat_session.document_name).suffix.lower() if chat_session.document_name else '.txt'
        # Create a temporary file to store the content
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(chat_session.document_content)
            full_file_path = Path(temp_file.name)
    
    try:
        # Ensure the file exists and has content
        if not full_file_path.exists() or os.path.getsize(full_file_path) == 0:
            raise ValueError("The uploaded file is empty or could not be read.")
            
        # Choose loader based on file type
        file_extension = file_extension.lower()
        logger.info(f"Processing file: {full_file_path} with extension: {file_extension}")
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(str(full_file_path))
            documents = loader.load()
        elif file_extension == '.txt':
            # For text files, ensure proper encoding
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(page_content=content)]
            except UnicodeDecodeError:
                # Fallback to different encoding if utf-8 fails
                with open(full_file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                documents = [Document(page_content=content)]
        else:
            # For other file types, try UnstructuredFileLoader
            try:
                loader = UnstructuredFileLoader(str(full_file_path))
                documents = loader.load()
            except Exception as e:
                raise ValueError(f"Unsupported file type: {file_extension}. Please upload a .txt or .pdf file.")
        
        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("The uploaded file appears to be empty or could not be processed.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embedding_function = get_gemini_embeddings()

        logger.info(f"Creating vector store for session {session_id} with {len(chunks)} chunks.")
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )
        logger.info(f"Vector store created successfully for session {session_id} at {vectorstore_path}")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary file if it was created from database content
        if not (file_path and os.path.exists(file_path)) and 'full_file_path' in locals():
            try:
                os.unlink(str(full_file_path))
            except Exception as e:
                logger.warning(f"Could not delete temporary file {full_file_path}: {e}")


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