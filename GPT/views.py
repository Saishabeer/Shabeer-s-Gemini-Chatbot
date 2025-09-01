import os
import gc
import logging
from django.conf import settings
from django.contrib import messages, auth
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from google.api_core.exceptions import (InvalidArgument, PermissionDenied, ResourceExhausted)
from langchain_google_genai._common import GoogleGenerativeAIError

from .gemini_service import gemini_chat_stream
from .models import ChatMessage, ChatSession
from .rag_service import (delete_vectorstore_for_session, has_vectorstore, ingest_document_for_session, rag_answer_stream)

# --- Basic Setup ---
logger = logging.getLogger(__name__)
User = get_user_model()


# --- Auth Views (logging is minimal here) ---
def register(request):
    """Handles user registration."""
    if request.method == "POST":
        # NOTE: Using Django Forms is highly recommended for robust validation and security.
        # This is a manual implementation as requested.
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')

        if not all([username, email, password, password2]):
            messages.error(request, "All fields are required.")
            return render(request, 'register.html')

        if password != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, 'register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, f"Username '{username}' is already taken.")
            return render(request, 'register.html')

        if User.objects.filter(email=email).exists():
            messages.error(request, "A user with that email already exists.")
            return render(request, 'register.html')

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        messages.success(request, "Registration successful!")
        return redirect('home')

    return render(request, 'register.html')


def user_login(request):
    """Handles user login."""
    if request.method == "POST":
        username_or_email = request.POST.get('username')
        password = request.POST.get('password')

        if not username_or_email or not password:
            messages.error(request, "Please enter a username/email and password.")
            return render(request, "login.html")

        user = authenticate(request, username=username_or_email, password=password)
        if user is None and '@' in username_or_email:
            try:
                user_by_email = User.objects.get(email=username_or_email)
                user = authenticate(request, username=user_by_email.username, password=password)
            except User.DoesNotExist:
                pass

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid username/email or password.")

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
    """
    Handles both rendering the chat interface (GET) and processing user input (POST).
    POST requests with a prompt will return a StreamingHttpResponse for real-time chat.
    """
    logger.info(f"Request for chat_view. Method: {request.method}, Session ID: {session_id}, User: {request.user}")
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
                messages.error(request, f"Sorry, there was an error processing your document: {e}")

                # Cleanup on failure
                if os.path.exists(file_path):
                    os.remove(file_path)
                delete_vectorstore_for_session(target_session.id)

                # If the session was newly created for this upload, delete it entirely
                if not active_session:  # active_session is None if we just created target_session
                    target_session.delete()
                    return redirect('home')  # Go home, the session is gone

                return redirect('chat_session', session_id=target_session.id)
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

            history = list(
                target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp").values('role',
                                                                                                            'content'))
            # Save the user's message before starting the stream.
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            def stream_response_generator():
                """A generator function that streams the AI response and saves the full text at the end."""
                full_response = []
                try:
                    # Route to the correct streaming service based on whether a vector store exists
                    if has_vectorstore(target_session.id):
                        logger.info(f"Streaming from RAG service for session {target_session.id}.")
                        stream = rag_answer_stream(prompt, target_session.id, history=history)
                    else:
                        logger.info("Streaming from General Chat service.")
                        stream = gemini_chat_stream(prompt, history=history)

                    for chunk in stream:
                        full_response.append(chunk)
                        yield chunk
                except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
                    logger.error(f"AI service error during stream for session {target_session.id}: {e}", exc_info=True)
                    yield "Error: The AI service is currently unavailable or at capacity. Please try again later."
                finally:
                    # After the stream is complete, save the full raw markdown response to the database.
                    # The frontend will be responsible for rendering it.
                    final_text = "".join(full_response).strip()
                    if final_text:
                        ChatMessage.objects.create(session=target_session, role='assistant', content=final_text)
                        logger.info(f"Successfully streamed and saved AI response for session {target_session.id}")
                    gc.collect()

            # Return the streaming response to the browser
            return StreamingHttpResponse(stream_response_generator(), content_type="text/plain")

        if not uploaded_file and not prompt:
            messages.error(request, "Please enter a prompt or upload a file.")
            # If there's an active session, redirect to it. Otherwise, go home.
            if active_session:
                return redirect('chat_session', session_id=active_session.id)
            return redirect('home')

    # This block handles GET requests
    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': active_session.messages.all() if active_session else [],
    }
    return render(request, "home.html", context)
