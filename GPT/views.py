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
from .rag_service import ingest_document_for_session, rag_answer, delete_vectorstore_for_session


def _get_rag_response_with_sources(prompt, session_id):
    """Helper to get RAG answer and conditionally append sources."""
    answer, srcs = rag_answer(prompt, session_id)

    # Only add sources if the answer doesn't contain the fallback phrase
    # and if sources were actually found.
    fallback_phrase = "I don't have that information in the provided knowledge base"
    if srcs and fallback_phrase not in answer:
        unique_srcs = sorted(list(set(s for s in srcs if s)))
        answer += "\n\n**Sources:**\n- " + "\n- ".join(unique_srcs)
    return answer

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
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")

        # --- ACTION 1: Handle File Upload ---
        # This block is triggered when a file is selected, thanks to the onchange event.
        # It can also handle a prompt being submitted at the same time.
        if uploaded_file:
            target_session = active_session or ChatSession.objects.create(user=request.user)

            # If this session already had a document, clean up old data first.
            if target_session.document_path:
                if os.path.exists(target_session.document_path):
                    try:
                        os.remove(target_session.document_path)
                    except OSError as e:
                        print(f"Error deleting old file {target_session.document_path}: {e}")
                delete_vectorstore_for_session(target_session.id)

            # Save the new file
            fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'user_docs')
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            # Update session model with new document info
            target_session.document_name = uploaded_file.name
            target_session.document_path = file_path
            target_session.save()

            # Ingest the new document
            try:
                _, n_chunks = ingest_document_for_session(target_session.id, file_path)
                # If there's no prompt, it's just a file upload. Show a success message.
                if not prompt:
                    messages.success(request, f"✅ Ready to answer questions about '{uploaded_file.name}'.")
            except Exception as e:
                messages.error(request, f"Error processing document: {e}")
                # Clean up what we just created
                target_session.document_name = None
                target_session.document_path = None
                target_session.save()
                if os.path.exists(file_path):
                    os.remove(file_path)
                # If it was a brand new session that failed, delete it.
                if not active_session:
                    target_session.delete()
                return redirect(request.path)

            # If a prompt was submitted along with the file, process it immediately.
            if prompt:
                ChatMessage.objects.create(session=target_session, role='user', content=prompt)
                ai_response = _get_rag_response_with_sources(prompt, target_session.id)
                ChatMessage.objects.create(session=target_session, role='assistant', content=ai_response)

            return redirect('chat_session', session_id=target_session.id)

        # --- ACTION 2: Handle Prompt Submission (No File Upload) ---
        elif prompt:
            target_session = active_session or ChatSession.objects.create(user=request.user)

            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            if target_session.document_path:
                ai_response = _get_rag_response_with_sources(prompt, target_session.id)
            else:
                history = [
                    {"role": m.role, "content": m.content}
                    for m in target_session.messages.filter(role__in=['user', 'assistant']).order_by("timestamp")
                ]
                ai_response = gemini_chat(prompt, history)

            ChatMessage.objects.create(session=target_session, role='assistant', content=ai_response)
            return redirect('chat_session', session_id=target_session.id)

        # --- ACTION 3: POST with no file and no prompt (e.g., empty form submission) ---
        else:
            messages.error(request, "Please enter a prompt.")
            return redirect(request.path)

    # GET request: Display the chat interface
    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': active_session.messages.all() if active_session else [],
    }
    # The original code used 'home.html' and 'chat.html'. We'll unify on 'home.html'
    # as it seems to be the main template.
    return render(request, "home.html", context)
