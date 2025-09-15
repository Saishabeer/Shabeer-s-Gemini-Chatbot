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
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Local Application Imports ---
from .utils import api_key_manager, with_api_key_rotation

logger = logging.getLogger(__name__)


# ✅ Use Free Tier Gemini Embedding Model
def get_gemini_embeddings():
    """
    Initializes and returns the GoogleGenerativeAIEmbeddings instance
    using the free tier experimental embedding model.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07",  # Free experimental embedding model
        google_api_key=api_key_manager.get_key()
    )


def has_vectorstore(session_id: int) -> bool:
    """Check if a vector store directory exists for a given session."""
    vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
    return vectorstore_path.exists() and any(vectorstore_path.iterdir())


def delete_vectorstore_for_session(session_id: int):
    """Deletes the Chroma vector store directory for a session."""
    if has_vectorstore(session_id):
        vectorstore_path = settings.CHROMA_DIR / f"session_{session_id}"
        try:
            shutil.rmtree(vectorstore_path)
            logger.info(f"Deleted vector store for session {session_id}.")
        except OSError as e:
            logger.error(f"Error deleting vector store for session {session_id}: {e}", exc_info=True)


# ✅ Ingest documents using the free Gemini embedding model
@with_api_key_rotation
def ingest_document_for_session(session_id: int, file_path: str = None):
    """
    Loads a document, splits it, generates embeddings using the free Gemini model,
    and stores them in Chroma vector database.
    """
    from .models import ChatSession

    chat_session = ChatSession.objects.get(id=session_id)
    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")
    temp_f = None

    try:
        # Handle file input
        if file_path and os.path.exists(file_path):
            full_file_path = Path(file_path)
            document_name = full_file_path.name
        else:
            if not chat_session.document_content or not chat_session.document_name:
                raise ValueError("No document content found in the session to process.")

            file_extension = Path(chat_session.document_name).suffix.lower()
            document_name = chat_session.document_name

            temp_f = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            temp_f.write(chat_session.document_content)
            temp_f.close()
            full_file_path = Path(temp_f.name)

        # Validate file
        if not full_file_path.exists() or os.path.getsize(full_file_path) == 0:
            raise ValueError("The uploaded file is empty or unreadable.")

        logger.info(f"Processing file '{document_name}' with extension {full_file_path.suffix.lower()}")

        # Load document based on file type
        if full_file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(full_file_path))
        elif full_file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(full_file_path), autodetect_encoding=True)
        else:
            loader = UnstructuredFileLoader(str(full_file_path))

        documents = loader.load()
        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("No text content found after processing the file.")

        # Add metadata
        for doc in documents:
            doc.metadata['source'] = document_name

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            logger.warning(f"No chunks created for document '{document_name}'.")
            return

        # Use the free Gemini embedding model
        embedding_function = get_gemini_embeddings()

        # Store embeddings in Chroma
        if has_vectorstore(session_id):
            vector_store = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function
            )
            vector_store.add_documents(documents=chunks)
            logger.info(f"Added {len(chunks)} new chunks to existing vector store for session {session_id}.")
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=vectorstore_path
            )
            logger.info(f"Created new vector store for session {session_id}.")

    except Exception as e:
        logger.error(f"Error during document ingestion for session {session_id}: {str(e)}", exc_info=True)
        raise
    finally:
        if temp_f:
            try:
                os.unlink(str(full_file_path))
            except Exception as e:
                logger.warning(f"Could not delete temp file {full_file_path}: {e}")


# ✅ Retrieve relevant chunks using free Gemini embedding
@with_api_key_rotation
def get_rag_context(query: str, session_id: int, top_k: int = 4) -> List[str]:
    """
    Retrieves relevant chunks from the vector store for a query using free Gemini embeddings.
    """
    if not has_vectorstore(session_id):
        logger.debug(f"No vectorstore found for session {session_id}.")
        return []

    vectorstore_path = str(settings.CHROMA_DIR / f"session_{session_id}")
    embedding_function = get_gemini_embeddings()

    vector_store = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(query)

    context_snippets = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown Document')
        snippet = f"Source: {source}\nContent: {doc.page_content}"
        context_snippets.append(snippet)

    logger.info(f"Retrieved {len(context_snippets)} relevant chunks for session {session_id}.")
    return context_snippets
