import os
import gc
import logging
from django.conf import settings
import re
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
from .rag_service import (
    delete_vectorstore_for_session,
    get_rag_context,
    has_vectorstore,
    ingest_document_for_session
)
from .web_search_service import web_search_manager

# --- Basic Setup ---
logger = logging.getLogger(__name__)
User = get_user_model()


# --- Auth Views ---
def register(request):
    """Handles user registration."""
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # The form's save method now handles password hashing and saving.
            user = form.save()

            # After successful creation, log the user in.
            # Since we didn't use `authenticate()`,
            # we must specify the authentication backend path.
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')

            messages.success(request, "Registration successful! Welcome.")
            return redirect('home')
        else:
            # If the form is invalid, inform the user so they look for field-specific errors
            # by iterating through the form's errors and adding them to the messages framework.
            # This provides explicit feedback even if the template doesn't render form errors.
            for field, error_list in form.errors.items():
                for error in error_list:
                    # Prepend the field label for clarity. '__all__' is for non-field errors.
                    if field == '__all__':
                        messages.error(request, error)
                    else:
                        label = form.fields[field].label or field.capitalize()
                        messages.error(request, f"{label}: {error}")
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


def user_login(request):
    """Handles user login (supports username or email)."""
    if request.method == "POST":
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            login_identifier = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            # The form's 'username' field can be either a username or an email.
            # We first try to authenticate assuming it's a username.
            user = authenticate(request, username=login_identifier, password=password)

            # If that fails and it looks like an email, we find the user by email
            # and then try to authenticate with their actual username.
            if user is None and '@' in login_identifier:
                try:
                    user_obj = User.objects.get(email__iexact=login_identifier)
                    user = authenticate(request, username=user_obj.username, password=password)
                except User.DoesNotExist:
                    user = None # User with this email does not exist.

            if user:
                login(request, user)
                return redirect('home')

            # If authentication fails after all attempts, show a clear error.
            messages.error(request, "Invalid credentials. Please check your username/email and password.")
        else:
            # If the form itself is invalid (e.g., empty fields), show a generic error.
            # The form will display field-specific errors in the template.
            messages.error(request, "There was an error with your submission.")
    else:
        form = UserLoginForm()
    return render(request, "login.html", {'form': form})


def user_logout(request):
    """Logs the user out."""
    logout(request)
    messages.info(request, "You have been successfully logged out.")
    return redirect('login')


# --- Chat View ---
@login_required
def chat_view(request, session_id=None):
    """
    Main chat view: renders chat UI (GET) and processes input (POST).
    Retrieval order = Documents → Web → LLM.
    """
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")
    active_session, chat_messages = None, []

    if session_id:
        active_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        chat_messages = active_session.messages.order_by("timestamp").all()

    if request.method == 'POST':
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")
        target_session = active_session

        # --- File Upload Handling ---
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name} by user {request.user.id}")

            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=uploaded_file.name)
            else:
                target_session.title = uploaded_file.name

            # Correctly delete the old vectorstore if a new file is uploaded for an existing session.
            # The previous logic was flawed as it checked for the file, not the vectorstore.
            if has_vectorstore(target_session.id):
                logger.info(f"Existing vectorstore found for session {target_session.id}. Deleting it before new ingestion.")
                delete_vectorstore_for_session(target_session.id)

            # Store files relative to MEDIA_ROOT for consistent and easy path reconstruction.
            # The previous logic made the path relative to a subdirectory, causing lookup failures.
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            file_path = fs.save(f"user_docs/user_{request.user.id}/{uploaded_file.name}", uploaded_file)
            target_session.document_path = file_path
            target_session.save()

            try:
                ingest_document_for_session(target_session.id, file_path)
                messages.success(request, f"✅ Ready to answer questions about '{uploaded_file.name}'.")
            except Exception as e:
                logger.error(f"Error processing document for session {target_session.id}: {e}", exc_info=True)
                messages.error(request, f"Sorry, there was an error processing your document: {e}")
                target_session.delete()
                return redirect('home')

            return redirect('chat_session', session_id=target_session.id)

        # --- Prompt Handling ---
        if prompt:
            is_new_session = False
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=prompt[:50])
                is_new_session = True
            elif target_session.title == 'New Chat':
                target_session.title = prompt[:50]
                target_session.save()

            # Fetch history BEFORE saving the new message to ensure it's in the correct state.
            history = list(target_session.messages.filter(role__in=['user', 'assistant'])
                           .order_by("timestamp").values('role', 'content'))
            # Now, save the new user message to the database.
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            def stream_response_generator():
                """Stream AI response with hierarchical retrieval."""
                full_response = []
                try:
                    # --- Query Analysis Step ---
                    # Decide if we need to perform external searches. Simple greetings and
                    # common pleasantries don't require the expensive rewrite/search pipeline.
                    # This is the most critical step for conserving API quota.
                    GREETINGS = {"hi", "hello", "hlo", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"}
                    is_simple_query = prompt.lower().strip() in GREETINGS

                    search_query = prompt  # Default to original prompt

                    # --- Query Rewriting (for follow-up questions) ---
                    # Only rewrite if it's not a simple greeting and there's history.
                    if not is_simple_query and history:
                        history_text = "\n".join(
                            [f"{item['role']}: {item['content']}" for item in history[-6:]]
                        )
                        rewrite_prompt = f"""You are an expert at rephrasing a follow-up question into a standalone question, using the provided chat history. Do not answer the question, just provide the rephrased, standalone question.

**Example 1:**
Chat History:
user: Who is the CEO of Tesla?
assistant: Elon Musk is the CEO of Tesla.
Follow-up Question: What other companies does he run?
Standalone Question: What other companies does Elon Musk run?

**Example 2:**
Chat History:
user: Can you tell me about the Golden Gate Bridge?
assistant: The Golden Gate Bridge is a suspension bridge spanning the Golden Gate.
Follow-up Question: how long is it
Standalone Question: How long is the Golden Gate Bridge?

**Your Task:**

Chat History:
{history_text}

Follow-up Question: {prompt}

Standalone Question:"""
                        try:
                            rewriter_stream = gemini_chat_stream(rewrite_prompt, history=[])
                            rewritten_query = "".join(list(rewriter_stream)).strip().replace('"', '')
                            if rewritten_query and '\n' not in rewritten_query:
                                search_query = rewritten_query
                            logger.info(f"Rewritten query: {search_query}")
                        except Exception as e:
                            logger.error(f"Query rewrite failed, using original prompt. Error: {e}")

                    # --- Information Retrieval ---
                    doc_context, web_context = "", ""

                    # Only perform search if it's not a greeting.
                    if not is_simple_query:
                        # 1. Uploaded Document Context
                        if has_vectorstore(target_session.id):
                            doc_snippets = get_rag_context(search_query, target_session.id)
                            if doc_snippets:
                                doc_context = "\n\n".join(doc_snippets)

                        # 2. Web Search Context
                        if web_search_manager.is_enabled():
                            web_results = web_search_manager.search(search_query)
                            web_context = "\n\n".join([r.get('content', '') for r in web_results if r.get('content')])

                    # 3. Build Final Prompt
                    if doc_context or web_context:
                        system_instruction = """You are Gemini Code Assist, a world-class software engineering coding assistant. Your primary goal is to provide clear, accurate, and well-formatted answers.

**Instructions:**
- Use the provided context to answer the user's question. Prioritize 'UPLOADED DOCUMENT CONTEXT', then 'WEB SEARCH RESULTS'.
- Format your answers clearly. Use lists, bold text, and paragraphs to make the response easy to read.
- **MANDATORY Code Formatting:** When you are asked to write code, you MUST enclose the entire code block in triple backticks, specifying the language. This is not optional.

    **Correct Example:**
    ```python
    def hello_world():
        print("Hello, World!")
    ```

    **Incorrect Example:**
    def hello_world():
        print("Hello, World!")

- Provide explanations before or after the code block, but the code itself must be inside the formatted block."""
                        context_parts = [system_instruction]
                        if doc_context:
                            context_parts.append(f"--- UPLOADED DOCUMENT CONTEXT ---\n{doc_context}")
                        if web_context:
                            context_parts.append(f"--- WEB SEARCH RESULTS ---\n{web_context}")
                        final_prompt = "\n\n".join(context_parts) + f"\n\n--- USER QUESTION ---\n{prompt}"
                    else:
                        # This path is taken for greetings or if search returns no results.
                        final_prompt = prompt

                    # --- LLM Streaming ---
                    # Transform the history into the format expected by the Gemini API.
                    # The API expects 'model' instead of 'assistant' for the role,
                    # and the content must be nested under 'parts'.
                    gemini_history = [
                        {'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [{'text': msg['content']}]}
                        for msg in history
                    ]

                    stream = gemini_chat_stream(final_prompt, history=gemini_history)

                    for chunk in stream:
                        full_response.append(chunk)
                        yield chunk

                except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
                    logger.error(f"AI service error in session {target_session.id}: {e}", exc_info=True)
                    yield "Error: AI service unavailable or at capacity. Please try again later."
                finally:
                    response_text = "".join(full_response).strip()
                    if response_text:
                        ChatMessage.objects.create(session=target_session, role='assistant', content=response_text)

            response = StreamingHttpResponse(stream_response_generator(), content_type="text/plain")
            if is_new_session:
                response['X-Chat-Session-Id'] = target_session.id
                response['X-Chat-Session-Title'] = target_session.title
            return response

    # --- Render for GET ---
    return render(request, 'home.html', {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': chat_messages,
    })


@login_required
def delete_chat_session(request, session_id):
    """Deletes chat session + vectorstore."""
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    if request.method == "POST":
        sid = session.id
        session.delete()
        delete_vectorstore_for_session(sid)
        messages.success(request, "Chat session deleted.")
    return redirect('home')
