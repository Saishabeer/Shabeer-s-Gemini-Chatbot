# --- Python Standard Library Imports ---
import gc
import logging

# --- Django Core Imports ---
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404

# --- Third-Party Library Imports ---
# Specific exceptions from the Google API client to handle API-related errors gracefully.
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted
# The custom forms defined in forms.py for user registration and login.
from .forms import UserRegistrationForm, UserLoginForm
# The core function that communicates with the Gemini API.
from .gemini_service import gemini_chat_stream
# The database models that represent our application's data structure.
from .models import ChatMessage, ChatSession
# Helper functions for the Retrieval-Augmented Generation (RAG) service.
from .rag_service import (
    delete_vectorstore_for_session,
    get_rag_context,
    has_vectorstore,
    ingest_document_for_session
)
# A specific error from the LangChain Google GenAI library.
from langchain_google_genai._common import GoogleGenerativeAIError
# The service for performing web searches.
from .web_search_service import web_search_manager

# --- Basic Setup ---
# Get a logger instance for this file to record events and errors.
logger = logging.getLogger(__name__)
# Get the currently active User model for the project (our custom User model).
User = get_user_model()


# --- Auth Views ---
def register(request):
    """
    Handles the user registration process.
    - Renders the registration form on GET request.
    - Processes the submitted form data on POST request.
    """
    # If the form has been submitted...
    if request.method == "POST":
        # Create a form instance and populate it with data from the request.
        form = UserRegistrationForm(request.POST)
        # Check if the form data is valid (e.g., passwords match, email is unique).
        if form.is_valid():
            try:
                # The form's save() method creates the new user in the database.
                user = form.save()
                # Log the new user in automatically.
                login(request, user)
                # Add a success message to be displayed on the next page.
                messages.success(request, 'Registration successful! Welcome to TechJays GPT.')
                # Redirect the user to the main chat page.
                return redirect('home')
            except Exception as e:
                # If something goes wrong during user creation, show an error.
                messages.error(request, f'An error occurred during registration: {str(e)}')
    # If it's a GET request (user is just visiting the page)...
    else:
        # Create a blank instance of the registration form.
        form = UserRegistrationForm()
    
    # Render the registration page template with the form.
    return render(request, 'register.html', {'form': form})


def user_login(request):
    """
    Handles the user login process.
    - Renders the login form on GET request.
    - Authenticates the user on POST request.
    """
    # If the login form has been submitted...
    if request.method == 'POST':
        # Create an instance of Django's AuthenticationForm with the submitted data.
        form = UserLoginForm(request, data=request.POST)
        # Check if the form is valid.
        if form.is_valid():
            # Get the cleaned email and password from the form.
            email = form.cleaned_data.get('username').lower().strip()
            password = form.cleaned_data.get('password')
            
            # Use Django's authenticate() function to check credentials against the database.
            # We pass the email as the 'username' because we configured it in our custom User model.
            user = authenticate(request, username=email, password=password)
            
            # If authenticate() returns a user object, credentials are correct.
            if user is not None:
                # Log the user in, creating a session.
                login(request, user)
                # Redirect to the 'next' page if it exists, otherwise to the home page.
                next_url = request.GET.get('next', 'home')
                return redirect(next_url)
            else:
                # If authentication fails, show a generic error. This is a security best practice.
                messages.error(request, 'Invalid email or password. Please try again.')
        # If the form itself is invalid (e.g., empty fields), show an error.
        messages.error(request, 'Invalid email or password. Please try again.')
    # If it's a GET request...
    else:
        # Create a blank instance of the login form.
        form = UserLoginForm()
    # Render the login page template with the form.
    return render(request, 'login.html', {'form': form})


def user_logout(request):
    """Logs the current user out and redirects them to the login page."""
    # Clear the user's session data from the server.
    logout(request)
    # Add a message to inform the user they've been logged out.
    messages.info(request, "You have been successfully logged out.")
    # Redirect to the login page.
    return redirect('login')


# --- Chat View ---
# This decorator ensures that only logged-in users can access this view.
# If a non-authenticated user tries to access it, they are redirected to the LOGIN_URL.
@login_required
def chat_view(request, session_id=None):
    """
    Main chat view: renders chat UI (GET) and processes input (POST).
    Retrieval order = Documents → Web → LLM.
    """
    # --- GET Request Logic (Displaying the page) ---
    # Fetch all chat sessions for the current user to display in the sidebar.
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by("-created_at")
    # Initialize variables for the active session and its messages.
    active_session, chat_messages = None, []

    # If a session_id is provided in the URL, we are loading an existing chat.
    if session_id:
        # Fetch the specific session, ensuring it belongs to the current user for security.
        # get_object_or_404 will raise a 404 Not Found error if the session doesn't exist or isn't owned by the user.
        active_session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        # Get all messages for the active session, ordered by timestamp.
        chat_messages = active_session.messages.order_by("timestamp").all()

    # --- POST Request Logic (Processing user input) ---
    if request.method == 'POST':
        # Get the prompt and any uploaded file from the form submission.
        prompt = request.POST.get("prompt", "").strip()
        uploaded_file = request.FILES.get("document")
        # The session we'll be working with is the currently active one.
        target_session = active_session

        # --- File Upload Handling ---
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name} by user {request.user.id}")

            # If there's no active session, create a new one and set its title to the filename.
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=uploaded_file.name)
            # If there is an active session but its title is the default, update it.
            elif target_session.title == 'New Chat':
                target_session.title = uploaded_file.name
                target_session.save()

            try:
                # Use the model's helper method to save the document's content and metadata to the database.
                target_session.save_document(uploaded_file)
                
                # Call the RAG service to process the document and create/update the vector store.
                ingest_document_for_session(target_session.id)
                
                # Create a "system" message to inform the user the file is ready.
                ChatMessage.objects.create(
                    session=target_session,
                    role='system',
                    content=f"Document '{uploaded_file.name}' was uploaded and is ready for questions."
                )
                
            except Exception as e:
                logger.error(f"Error processing document for session {target_session.id}: {e}", exc_info=True)
                messages.error(request, f"Sorry, there was an error processing your document: {e}")
                target_session.delete()
                return redirect('home')

            return redirect('chat_session', session_id=target_session.id)

        # --- Prompt Handling ---
        if prompt:
            is_new_session = False
            # If there's no active session, create a new one and use the first 50 chars of the prompt as the title.
            if not target_session:
                target_session = ChatSession.objects.create(user=request.user, title=prompt[:50])
                is_new_session = True
            elif target_session.title == 'New Chat':
                target_session.title = prompt[:50]
                target_session.save()

            # Fetch the conversation history for context BEFORE adding the new user message.
            history = list(target_session.messages.filter(role__in=['user', 'assistant'])
                           .order_by("timestamp").values('role', 'content'))
            # Save the user's new message to the database immediately.
            ChatMessage.objects.create(session=target_session, role='user', content=prompt)

            # This nested function is a "generator". It will be executed piece by piece
            # by StreamingHttpResponse, allowing us to send the AI's response in chunks.
            def stream_response_generator():
                """Stream AI response with hierarchical retrieval."""
                full_response = []
                try:
                    # --- 1. Query Analysis ---
                    # Check if the prompt is a simple greeting. This is a crucial optimization
                    # to avoid wasting expensive API calls on simple interactions.
                    GREETINGS = {"hi", "hello", "hlo", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"}
                    is_simple_query = prompt.lower().strip() in GREETINGS

                    search_query = prompt  # Default to original prompt

                    # --- Query Rewriting (for follow-up questions) ---
                    # Only rewrite if it's not a simple greeting and there's history.
                    # This turns a question like "how long is it" into a standalone question
                    # like "How long is the Golden Gate Bridge?", which is much better for searching.
                    if not is_simple_query and history:
                        history_text = "\n".join(
                            [f"{item['role']}: {item['content']}" for item in history[-6:]]
                        )
                        # Construct a detailed prompt for the LLM to perform the rewrite task.
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
                            # Call the LLM to get the rewritten query.
                            rewriter_stream = gemini_chat_stream(rewrite_prompt, history=[])
                            rewritten_query = "".join(list(rewriter_stream)).strip().replace('"', '')
                            if rewritten_query and '\n' not in rewritten_query:
                                search_query = rewritten_query
                            logger.info(f"Rewritten query: {search_query}")
                        except Exception as e:
                            logger.error(f"Query rewrite failed, using original prompt. Error: {e}")

                    # --- 2. Information Retrieval (RAG) ---
                    doc_context, web_context = "", ""

                    # Only perform searches if it's not a simple greeting.
                    if not is_simple_query:
                        # First, search the uploaded document's vector store.
                        if has_vectorstore(target_session.id):
                            doc_snippets = get_rag_context(search_query, target_session.id)
                            if doc_snippets:
                                doc_context = "\n\n".join(doc_snippets)

                        # Second, perform a web search for up-to-date information.
                        if web_search_manager.is_enabled():
                            web_results = web_search_manager.search(search_query)
                            web_context = "\n\n".join([r.get('content', '') for r in web_results if r.get('content')])

                    # --- 3. Final Prompt Construction ---
                    # If we found any context from our searches, build a detailed final prompt.
                    if doc_context or web_context:
                        # This is the main "system prompt" that defines the AI's persona and rules.
                        system_instruction = """**📖 Role & Personality**
You are a friendly, helpful, and conversational AI assistant 🤖✨.
You always respond in a clear, approachable, and warm tone.
Use emojis to make conversations engaging 🎉🔥, but don’t overdo it.
Highlight important parts with quotes or markdown headings.

**📚 Knowledge Sources & Context**
Your knowledge comes from three places. You must use the provided context when it's available.
1.  **RAG (Uploaded Documents):** When I provide `--- UPLOADED DOCUMENT CONTEXT ---`, you MUST prioritize this information. Mention that your answer is based on the document.
2.  **Web Search:** When I provide `--- WEB SEARCH RESULTS ---`, use it for up-to-date information.
3.  **Internal Knowledge:** Use your pre-trained knowledge for general questions or when no other context is provided.

**🧠 Conversational Memory**
Keep track of the current conversation to keep answers relevant. Remember user preferences and past topics across sessions to create a personalized experience. For example, if the user mentioned learning Java, you can refer to it in future chats.

**💬 Response Style**
- Begin with a friendly greeting (e.g., "Hey there! 👋").
- Structure your answers with bold headings. **Do not use** hash characters like `#`, `##`, or `###`.
- Use bullets and short paragraphs for clarity.
- End with a positive and helpful closing remark (e.g., "Hope this clears things up! 🚀").

**💻 Flawless Code Snippets**
- This is **MANDATORY**: All code blocks MUST be enclosed in triple backticks with the language specified (e.g., ```python).
- Provide clear explanations before or after the code, but never mix explanations inside the code block itself.
"""
                        context_parts = [system_instruction]
                        # Add the retrieved context, clearly labeled for the AI.
                        if doc_context:
                            context_parts.append(f"--- UPLOADED DOCUMENT CONTEXT ---\n{doc_context}")
                        if web_context:
                            context_parts.append(f"--- WEB SEARCH RESULTS ---\n{web_context}")
                        # Combine everything into the final prompt.
                        final_prompt = "\n\n".join(context_parts) + f"\n\n--- USER QUESTION ---\n{prompt}"
                    else:
                        # If no context was found (or for simple greetings), just use the user's prompt.
                        final_prompt = prompt

                    # --- 4. LLM Streaming ---
                    # Transform the history into the format expected by the Gemini API.
                    # The API expects 'model' for the AI's role, not 'assistant'.
                    # and the content must be nested under 'parts'.
                    gemini_history = [
                        {'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': [{'text': msg['content']}]}
                        for msg in history
                    ]

                    stream = gemini_chat_stream(final_prompt, history=gemini_history)

                    # Loop through the stream from the AI.
                    for chunk in stream:
                        # Append each chunk to our full response list.
                        full_response.append(chunk)
                        # `yield` sends this chunk immediately to the browser, creating the typing effect.
                        yield chunk

                except (ResourceExhausted, PermissionDenied, InvalidArgument, GoogleGenerativeAIError) as e:
                    # Catch specific API errors and yield a user-friendly message.
                    logger.error(f"AI service error in session {target_session.id}: {e}", exc_info=True)
                    yield "Error: AI service unavailable or at capacity. Please try again later."
                finally:
                    # This block runs after the stream is complete, whether it succeeded or failed.
                    # Join all the received chunks into a single string.
                    response_text = "".join(full_response).strip()
                    # If we got a response, save it to the database as an 'assistant' message.
                    if response_text:
                        ChatMessage.objects.create(session=target_session, role='assistant', content=response_text)
                    # Manually trigger Python's garbage collector to free up memory.
                    gc.collect()

            # Create the streaming response object, pointing to our generator function.
            response = StreamingHttpResponse(stream_response_generator(), content_type="text/plain")
            # If this was a new chat, send the new session ID and title back to the frontend
            # in the response headers so the URL can be updated without a page reload.
            if is_new_session:
                response['X-Chat-Session-Id'] = target_session.id
                response['X-Chat-Session-Title'] = target_session.title
            return response

    # --- Final Render for GET requests ---
    # This is the default action if the request is not a POST.
    # It renders the main chat page with the session list and any active chat messages.
    return render(request, 'home.html', {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'chat_messages': chat_messages,
    })


@login_required
def delete_chat_session(request, session_id):
    """Deletes a chat session and its associated vector store."""
    # Get the session, ensuring it belongs to the current user.
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    # This action should only be triggered by a POST request for security.
    if request.method == "POST":
        sid = session.id
        # Deleting the session object will also delete all its messages
        # because of the `on_delete=models.CASCADE` setting in the ChatMessage model.
        session.delete()
        # Crucially, also delete the associated vector store from the filesystem.
        delete_vectorstore_for_session(sid)
        messages.success(request, "Chat session deleted.")
    # Redirect back to the home page after deletion.
    return redirect('home')
