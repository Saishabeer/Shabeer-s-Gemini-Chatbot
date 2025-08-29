from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .models import ChatSession, ChatMessage
from .gemini_service import gemini_chat
from .rag_service import ingest_document_for_session, rag_answer


# --- Auth Views (Maintained from original, they are functional) ---

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
                pass # No user with that email

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "❌ Invalid username/email or password")
            return redirect("login")

    return render(request, "login.html")


# --- Consolidated and Corrected Chat View ---

@login_required
def chat_view(request, session_id=None):
    active_session = None
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")

    if session_id:
        try:
            active_session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            messages.error(request, "Chat session not found.")
            return redirect("home")

    # POST request: Handle new prompts and document uploads
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        uploaded_file = request.FILES.get("document")

        if not prompt:
            messages.error(request, "Prompt cannot be empty.")
            return redirect(request.path)

        # Scenario 1: A document is uploaded.
        if uploaded_file:
            # If we are in an existing session, use it. Otherwise, create a new one.
            if active_session:
                session_for_rag = active_session
                # If the session already had a document, we should clean up the old file.
                if session_for_rag.document_path and os.path.exists(session_for_rag.document_path):
                    try:
                        os.remove(session_for_rag.document_path)
                    except OSError as e:
                        # Log the error, but don't block the user.
                        print(f"Error deleting old file {session_for_rag.document_path}: {e}")
            else:
                session_for_rag = ChatSession.objects.create(user=request.user)

            # Save the file securely
            fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'user_docs')
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            # Update session with document info
            session_for_rag.document_name = uploaded_file.name
            session_for_rag.document_path = file_path
            session_for_rag.save()

            # Ingest the document into the vector store
            try:
                _, n_chunks = ingest_document_for_session(session_for_rag.id, file_path)
                messages.success(request, f"✅ Successfully uploaded and indexed '{uploaded_file.name}'.")
            except Exception as e:
                messages.error(request, f"Error processing document: {e}")
                session_for_rag.delete() # Clean up failed session
                return redirect('home')

            # Save user message and get RAG response
            ChatMessage.objects.create(session=session_for_rag, role='user', content=prompt)
            answer, srcs = rag_answer(prompt, session_for_rag.id)
            if srcs:
                # Append sources for clarity in the chat
                unique_srcs = sorted(list(set(s for s in srcs if s)))
                answer += "\n\n**Sources:**\n- " + "\n- ".join(unique_srcs)
            ChatMessage.objects.create(session=session_for_rag, role='assistant', content=answer)

            return redirect('chat_session', session_id=session_for_rag.id)

        # Scenario 2: No document, just a prompt. Continue existing or start new chat.
        else:
            target_session = active_session
            # If on the home page (no active session), create a new one
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user)

            # Save user message
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            # Check if this session is a RAG session (has a document)
            if target_session.document_path:
                answer, srcs = rag_answer(prompt, target_session.id)
                if srcs:
                    unique_srcs = sorted(list(set(s for s in srcs if s)))
                    answer += "\n\n**Sources:**\n- " + "\n- ".join(unique_srcs)
                ai_response = answer
            else:
                # Standard Gemini chat with history
                history = [
                    {"role": m.role, "content": m.content}
                    for m in target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp")
                ]
                ai_response = gemini_chat(prompt, history)

            ChatMessage.objects.create(session=target_session, role='assistant', content=ai_response)
            return redirect('chat_session', session_id=target_session.id)

    # GET request: Display the chat interface
    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': active_session.messages.all() if active_session else [],
    }
    # The original code used 'home.html' and 'chat.html'. We'll unify on 'home.html'
    # as it seems to be the main template.
    return render(request, "home.html", context)
