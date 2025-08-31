import os
import markdown2
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from langchain_google_genai._common import GoogleGenerativeAIError
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

from .models import ChatSession, ChatMessage
from .gemini_service import gemini_chat
from .rag_service import ingest_document_for_session, rag_answer, delete_vectorstore_for_session, has_vectorstore


def _process_and_format_response(prompt, session_id, history):
    """
    Gets a response from the appropriate service (RAG or general) and formats it.
    This now assumes rag_answer is smart enough to return empty sources for non-doc answers.
    """
    answer, srcs = rag_answer(prompt, session_id, history=history)
    if srcs:
        # It's a document-based answer. Append the sources.
        unique_srcs = sorted(list(set(s for s in srcs if s)))
        answer += "\n\n**Sources:**\n- " + "\n- ".join(unique_srcs)
    return answer


# --- Auth Views ---

def register(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password != password2:
            messages.error(request, 'Passwords do not match.')
            return redirect('register')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already in use.')
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already in use.')
            return redirect('register')

        new_user = User.objects.create_user(username=username, email=email, password=password)
        login(request, new_user)
        messages.success(request, f'✅ Welcome, {new_user.username}! Your registration was successful.')
        return redirect('home')

    return render(request, 'register.html')


def user_login(request):
    if request.method == "POST":
        username_or_email = request.POST.get("username")
        password = request.POST.get("password")

        if not username_or_email or not password:
            messages.error(request, "Please enter a username/email and password.")
            return redirect('login')

        # Try to authenticate with username first
        user = authenticate(request, username=username_or_email, password=password)

        # If that fails, try to see if it was an email
        if user is None:
            try:
                user_by_email = User.objects.get(email=username_or_email)
                user = authenticate(request, username=user_by_email.username, password=password)
            except User.DoesNotExist:
                pass  # No user with that email

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "❌ Invalid username/email or password")
            return redirect("login")

    return render(request, "login.html")


@login_required
def user_logout(request):
    """Logs the current user out."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


@login_required
def delete_chat_session(request, session_id):
    """Deletes a chat session and its related data."""
    if request.method == "POST":
        try:
            session_to_delete = ChatSession.objects.get(id=session_id, user=request.user)

            # Cleanup RAG data
            if has_vectorstore(session_to_delete.id):
                delete_vectorstore_for_session(session_to_delete.id)
            if session_to_delete.document_path and os.path.exists(session_to_delete.document_path):
                try:
                    os.remove(session_to_delete.document_path)
                except OSError as e:
                    print(f"Error deleting document file {session_to_delete.document_path}: {e}")

            session_to_delete.delete()
            messages.success(request, "Chat session deleted successfully.")
        except ChatSession.DoesNotExist:
            messages.error(request, "Chat session not found.")
    return redirect('home')


# --- Main Chat View ---

@login_required
def chat_view(request, session_id=None):
    active_session = None
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")

    if session_id:
        active_session = get_object_or_404(ChatSession, id=session_id, user=request.user)

    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")

        # --- ACTION 1: Handle File Upload ---
        if uploaded_file:
            target_session = active_session or ChatSession.objects.create(user=request.user)

            # If this session already had a document, clean up old data first.
            if target_session.document_path and os.path.exists(target_session.document_path):
                try:
                    os.remove(target_session.document_path)
                except OSError as e:
                    print(f"Error deleting old file {target_session.document_path}: {e}")
                delete_vectorstore_for_session(target_session.id)

            # Save the new file
            fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'user_docs')
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            # Update session model
            target_session.document_name = uploaded_file.name
            target_session.document_path = file_path
            target_session.save()

            # Ingest the new document
            try:
                ingest_document_for_session(target_session.id, file_path)
                if not prompt:
                    messages.success(request, f"✅ Ready to answer questions about '{uploaded_file.name}'.")
            except Exception as e:
                messages.error(request, f"Error processing document: {e}")
                # Clean up on failure
                target_session.document_name = None
                target_session.document_path = None
                target_session.save()
                if os.path.exists(file_path):
                    os.remove(file_path)
                if not active_session:
                    target_session.delete()
                return redirect(request.path)

            # If a prompt was submitted with the file, fall through to process it
            if not prompt:
                return redirect('chat_session', session_id=target_session.id)

        # --- ACTION 2: Handle Prompt Submission ---
        if prompt:
            target_session = active_session or ChatSession.objects.create(user=request.user)
            history = list(target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp").values('role', 'content'))
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            try:
                if target_session.document_path and has_vectorstore(target_session.id):
                    ai_response = _process_and_format_response(prompt, target_session.id, history)
                else:
                    ai_response = gemini_chat(prompt, history)

                # Convert Markdown to HTML for code formatting
                html_response = markdown2.markdown(
                    ai_response,
                    extras=["fenced-code-blocks", "code-friendly"]
                )
                ChatMessage.objects.create(session=target_session, role='assistant', content=html_response)

            except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError):
                messages.error(request, "The service is currently at its daily capacity. Please try again tomorrow.")

            return redirect('chat_session', session_id=target_session.id)

        # --- ACTION 3: POST with no file and no prompt ---
        else:
            messages.error(request, "Please enter a prompt.")
            return redirect(request.path)

    # GET request: Display the chat interface
    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': active_session.messages.all() if active_session else [],
    }
    return render(request, "home.html", context)