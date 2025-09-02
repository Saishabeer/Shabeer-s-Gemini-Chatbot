import json
import logging
import os
import tempfile
from pathlib import Path

from django.http import HttpRequest
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .gemini_service import gemini_chat_stream
from .models import ChatMessage, ChatSession
from .rag_service import (
    delete_vectorstore_for_session,
    get_rag_context,
    has_vectorstore,
    ingest_document_for_session,
)
from .web_search_service import web_search_manager

logger = logging.getLogger(__name__)

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """Custom authentication class to disable CSRF for API endpoints."""
    def enforce_csrf(self, request):
        return  # Skip CSRF verification


@api_view(['POST'])
@authentication_classes([CsrfExemptSessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def upload_document(request: HttpRequest):
    """
    API endpoint to upload and process a document for a chat session.
    
    Returns:
        JSON response with processing results or error message
    """
    try:
        # Get or create chat session
        session_id = request.POST.get('session_id')
        if not session_id:
            return Response(
                {'error': 'session_id is required for document upload via API.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        else:
            chat_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Get uploaded file
        if 'file' not in request.FILES:
            return Response(
                {'error': 'No file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        uploaded_file = request.FILES['file']
        
        # Save the file to a temporary location to be processed by the RAG service
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_f:
            for chunk in uploaded_file.chunks():
                temp_f.write(chunk)
            temp_file_path = temp_f.name

        try:
            # Save document metadata to the session
            chat_session.save_document(uploaded_file)
            
            # Ingest the document into the vector store
            ingest_document_for_session(session_id=session_id, file_path=temp_file_path)
            
            return Response({
                'status': 'success',
                'session_id': session_id,
                'message': f"Document '{uploaded_file.name}' uploaded and processed successfully."
            })
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@authentication_classes([CsrfExemptSessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def query_chat(request: HttpRequest, session_id: int):
    """
    API endpoint to query the chat with a message.
    """
    try:
        chat_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        data = json.loads(request.body)
        prompt = data.get('prompt')
        
        if not prompt:
            return Response(
                {'error': 'No prompt provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # --- Replicate logic from chat_view ---
        history = list(chat_session.messages.filter(role__in=['user', 'assistant'])
                       .order_by("timestamp").values('role', 'content'))
        ChatMessage.objects.create(session=chat_session, role='user', content=prompt)

        GREETINGS = {"hi", "hello", "hlo", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"}
        is_simple_query = prompt.lower().strip() in GREETINGS
        search_query = prompt

        # --- Query Rewriting ---
        if not is_simple_query and history:
            # This is a simplified version. The full rewrite prompt is in views.py
            # For an API, you might want to refactor this into a shared utility.
            logger.info("Follow-up question detected, but query rewrite is simplified for API. Using original prompt for search.")

        # --- Information Retrieval ---
        doc_context, web_context = "", ""
        if not is_simple_query:
            if has_vectorstore(chat_session.id):
                doc_snippets = get_rag_context(search_query, chat_session.id)
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

        # --- LLM Call ---
        gemini_history = [
            {'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [{'text': msg['content']}]}
            for msg in history
        ]
        
        stream = gemini_chat_stream(final_prompt, history=gemini_history)
        full_response = "".join(list(stream))

        # --- Save and Respond ---
        ChatMessage.objects.create(
            session=chat_session,
            role='assistant',
            content=full_response
        )
        
        return Response({
            'status': 'success',
            'response': full_response,
        })
        
    except json.JSONDecodeError:
        return Response(
            {'error': 'Invalid JSON in request body'},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error in query_chat: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@authentication_classes([CsrfExemptSessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def get_chat_history_api(request: HttpRequest, session_id: int):
    """
    API endpoint to retrieve chat history for a session.
    """
    try:
        # Verify the session belongs to the user
        chat_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Get chat history from the database
        history = list(chat_session.messages.order_by('timestamp').values('role', 'content', 'timestamp'))
        
        return Response({
            'status': 'success',
            'session_id': session_id,
            'history': history
        })
    except Exception as e:
        logger.error(f"Error in get_chat_history_api: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
@authentication_classes([CsrfExemptSessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def delete_chat_session_api(request: HttpRequest, session_id: int):
    """
    API endpoint to delete a chat session and its vector store.
    """
    try:
        # Verify the session belongs to the user
        chat_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        # Delete the vector store
        delete_vectorstore_for_session(session_id)
        
        # Delete the chat session and associated messages
        chat_session.delete()
        
        return Response({
            'status': 'success',
            'message': f'Chat session {session_id} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in delete_chat_session_api: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
