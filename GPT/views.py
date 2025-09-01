import os
import re
import gc
import logging
import markdown2
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, get_user_model
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, get_object_or_404
from google.api_core.exceptions import (InvalidArgument, PermissionDenied, ResourceExhausted)
from langchain_google_genai._common import GoogleGenerativeAIError

from .gemini_service import gemini_chat
from .models import ChatMessage, ChatSession
from .rag_service import (delete_vectorstore_for_session, has_vectorstore, ingest_document_for_session, rag_answer)

# --- Basic Setup ---
logger = logging.getLogger(__name__)
User = get_user_model()


def _get_and_format_response(prompt: str, history: list, session_id: int = None) -> str:
    """Gets a response from the appropriate service and formats it with sources."""
    if session_id and has_vectorstore(session_id):
        logger.info(f"Routing to RAG service for session {session_id}.")
        answer, srcs = rag_answer(prompt, session_id, history=history)
    else:
        logger.info("Routing to General Chat service.")
        answer, srcs = gemini_chat(prompt, history=history)

    answer = re.sub(r'\s*\[(WEB-)?\d+(,\s*(WEB-)?\d+)*\]', '', answer).strip()

    if srcs:
        unique_srcs = sorted(list(set(s for s in srcs if s)))
        answer += f"\n\n**Source:**\n- {'\n- '.join(unique_srcs)}"
    return answer


# --- Auth Views (logging is minimal here) ---
def register(request):
    if request.method == "POST":
        # ... (registration logic) ...
        return redirect('home')
    return render(request, 'register.html')

def user_login(request):
    if request.method == "POST":
        # ... (login logic) ...
        return redirect('home')
    return render(request, "login.html")


# --- Main Application Views ---
@login_required
def delete_chat_session(request, session_id):
    logger.info(f"User {request.user} initiated DELETE for session {session_id}")
    if request.method == "POST":
        session_to_delete = get_object_or_404(ChatSession, id=session_id, user=request.user)

        if has_vectorstore(session_to_delete.id):
            delete_vectorstore_for_session(session_to_delete.id)
        if session_to_delete.document_path and os.path.exists(session_to_delete.document_path):
            try:
                os.remove(session_to_delete.document_path)
            except OSError as e:
                logger.error(f"Error deleting document file {session_to_delete.document_path}: {e}")

        session_to_delete.delete()
        messages.success(request, "Chat session deleted successfully.")
        logger.info(f"Successfully deleted session {session_id}")
    return redirect('home')


@login_required
def chat_view(request, session_id=None):
    logger.info(f"Request received for chat_view. Method: {request.method}, Session ID: {session_id}")
    active_session = get_object_or_404(ChatSession, id=session_id, user=request.user) if session_id else None
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")

    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")
        target_session = active_session

        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name}")
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=uploaded_file.name)
                logger.info(f"Created new session {target_session.id} for file upload.")
            else:
                target_session.title = uploaded_file.name
                logger.info(f"Updating existing session {target_session.id} with new file.")

            if target_session.document_path and os.path.exists(target_session.document_path):
                delete_vectorstore_for_session(target_session.id)

            fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'user_docs')
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            target_session.document_name = uploaded_file.name
            target_session.document_path = file_path
            target_session.save()

            try:
                ingest_document_for_session(target_session.id, file_path)
                if not prompt:
                    messages.success(request, f"✅ Ready to answer questions about '{uploaded_file.name}'.")
            except Exception as e:
                logger.error(f"Error processing document for session {target_session.id}: {e}", exc_info=True)
                messages.error(request, f"Error processing document: {e}")
                # ... (cleanup logic) ...
                return redirect(request.path)
            finally:
                gc.collect()

            if not prompt:
                return redirect('chat_session', session_id=target_session.id)

        if prompt:
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=prompt[:50])
                logger.info(f"Created new session {target_session.id} for prompt: '{prompt[:50]}...'")
            elif target_session.title == 'New Chat':
                target_session.title = prompt[:50]
                target_session.save()
                logger.info(f"Updated session {target_session.id} title to: '{prompt[:50]}...'")

            history = list(target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp").values('role', 'content'))
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            try:
                ai_response = _get_and_format_response(prompt, history, session_id=target_session.id)
                html_response = markdown2.markdown(ai_response, extras=["fenced-code-blocks", "code-friendly"])
                ChatMessage.objects.create(session=target_session, role='assistant', content=html_response)
                logger.info(f"Successfully generated and saved AI response for session {target_session.id}")
            except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
                logger.error(f"AI service error for session {target_session.id}: {e}", exc_info=True)
                messages.error(request, "The service is currently at its daily capacity. Please try again tomorrow.")
            finally:
                gc.collect()

            return redirect('chat_session', session_id=target_session.id)

        if not uploaded_file and not prompt:
            messages.error(request, "Please enter a prompt or upload a file.")
            return redirect(request.path)

    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': active_session.messages.all() if active_session else [],
    }
    return render(request, "home.html", context)
