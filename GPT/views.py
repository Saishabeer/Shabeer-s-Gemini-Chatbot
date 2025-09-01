import os
import gc
import logging
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from google.api_core.exceptions import (InvalidArgument, PermissionDenied, ResourceExhausted)
from langchain_google_genai._common import GoogleGenerativeAIError

from .forms import UserRegistrationForm, UserLoginForm
from .gemini_service import gemini_chat_stream
from .models import ChatMessage, ChatSession
from .rag_service import (delete_vectorstore_for_session, has_vectorstore, ingest_document_for_session,
                          rag_answer_stream)

# --- Basic Setup ---
logger = logging.getLogger(__name__)
User = get_user_model()


# --- Auth Views (using Django Forms for security and validation) ---
def register(request):
    """Handles user registration."""
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            new_user = form.save(commit=False)
            new_user.set_password(form.cleaned_data['password'])
            new_user.save()
            login(request, new_user)
            messages.success(request, "Registration successful!")
            return redirect('home')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


def user_login(request):
    """Handles user login."""
    if request.method == "POST":
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            # Custom logic to allow login with email
            user = authenticate(request, username=username, password=password)
            if user is None and '@' in username:
                try:
                    user_by_email = User.objects.get(email=username)
                    user = authenticate(request, username=user_by_email.username, password=password)
                except User.DoesNotExist:
                    pass

            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, "Invalid username/email or password.")
        else:
            messages.error(request, "Invalid username/email or password.")
    else:
        form = UserLoginForm()
    return render(request, "login.html", {'form': form})


def user_logout(request):
    """Logs the user out."""
    logout(request)
    messages.info(request, "You have been successfully logged out.")
    return redirect('login')


# --- Main Application Views ---
@login_required
def chat_view(request, session_id=None):
    """
    Handles both rendering the chat interface (GET) and processing user input (POST).
    This is the main view for the chat application.
    """
    # --- Data Fetching for GET requests ---
    # Always fetch all sessions for the current user to display in the sidebar.
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")

    active_session = None
    chat_messages = []

    # If a session_id is provided, find the active session and its messages.
    if session_id:
        active_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        chat_messages = active_session.messages.order_by("timestamp").all()

    # --- Logic for POST requests (new messages or file uploads) ---
    if request.method == 'POST':
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")

        target_session = active_session

        # --- File Upload Handling ---
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name} by user {request.user.id}")
            # Create a new session if one isn't active, or update the existing one.
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=uploaded_file.name)
            else:
                target_session.title = uploaded_file.name

            # Clean up old vector store if a new file is uploaded to the same session
            if target_session.document_path and os.path.exists(target_session.document_path):
                delete_vectorstore_for_session(target_session.id)

            fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'user_docs')
            file_path = fs.save(f"user_{request.user.id}/{uploaded_file.name}", uploaded_file)
            target_session.document_path = file_path
            target_session.save()

            try:
                ingest_document_for_session(target_session.id, file_path)
                messages.success(request, f"✅ Ready to answer questions about '{uploaded_file.name}'.")
            except Exception as e:
                logger.error(f"Error processing document for session {target_session.id}: {e}", exc_info=True)
                messages.error(request, f"Sorry, there was an error processing your document: {e}")
                # Clean up failed session artifacts
                target_session.delete()
                return redirect('home')
            finally:
                gc.collect()

            # Redirect to the session view after file processing is complete.
            return redirect('chat_session', session_id=target_session.id)

        # --- Prompt Handling ---
        if prompt:
            is_new_session = False
            # If no session is active, create a new one.
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=prompt[:50])
                is_new_session = True
            # If the session is new, update its title with the first prompt.
            elif target_session.title == 'New Chat':
                target_session.title = prompt[:50]
                target_session.save()

            # Save the user's message before starting the stream.
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)
            history = list(
                target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp").values('role',
                                                                                                            'content'))

            def stream_response_generator():
                """A generator function that streams the AI response and saves the full text at the end."""
                full_response = []
                try:
                    # Route to the correct streaming service
                    if has_vectorstore(target_session.id):
                        stream = rag_answer_stream(prompt, target_session.id, history=history)
                    else:
                        stream = gemini_chat_stream(prompt, history=history)

                    for chunk in stream:
                        full_response.append(chunk)
                        yield chunk
                except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
                    logger.error(f"AI service error during stream for session {target_session.id}: {e}", exc_info=True)
                    yield "Error: The AI service is currently unavailable or at capacity. Please try again later."
                finally:
                    # After the stream, save the full raw markdown response to the database.
                    final_text = "".join(full_response).strip()
                    if final_text:
                        ChatMessage.objects.create(session=target_session, role='assistant', content=final_text)
                    gc.collect()

            # Return the streaming response to the browser
            response = StreamingHttpResponse(stream_response_generator(), content_type="text/plain")

            # If a new session was created, add a custom header with its ID
            # so the frontend can update the URL.
            if is_new_session:
                response['X-Chat-Session-Id'] = target_session.id
            return response

    # --- Context for GET requests ---
    # This context is used to render the page initially.
    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': chat_messages,
    }
    return render(request, 'home.html', context)


@login_required
def delete_chat_session(request, session_id):
    """Deletes a chat session and its associated vector store."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    if request.method == "POST":
        session_id_to_delete = session.id
        session.delete()
        delete_vectorstore_for_session(session_id_to_delete)
        messages.success(request, "Chat session deleted.")
    return redirect('home')