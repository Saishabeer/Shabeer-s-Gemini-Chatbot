from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import ChatSession, ChatMessage


# Register User
def register(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        password2 = request.POST.get('password2')

        # Your template sends 'password1' and 'password2', so we need to check them.
        if password != password2:
            messages.error(request, 'Passwords do not match.')
            return redirect('register')

        # The template uses the messages framework, so we should use it for all errors.
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already in use.')
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already in use.')
            return redirect('register')

        # create new user
        new_user = User.objects.create_user(username=username, email=email, password=password)

        # Log the new user in automatically
        login(request, new_user)

        # Add a welcome message and redirect directly to the home page.
        messages.success(request, f'✅ Welcome, {new_user.username}! Your registration was successful.')
        return redirect('home')

    return render(request, 'register.html')

# Login User
def user_login(request):
    if request.method == "POST":
        username_or_email = request.POST.get("username")
        password = request.POST.get("password")

        if not username_or_email or not password:
            messages.error(request, "Please enter a username/email and password.")
            return redirect('login')

        # First, try to authenticate with the input as a username
        user = authenticate(request, username=username_or_email, password=password)

        # If that fails, try to see if the input was an email
        if user is None:
            try:
                user_by_email = User.objects.get(email=username_or_email)
                user = authenticate(request, username=user_by_email.username, password=password)
            except User.DoesNotExist:
                pass  # Let the final check handle the error message

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("home")
            else:
                messages.error(request, "⚠️ Your account is inactive.")
                return redirect("login")
        else:
            messages.error(request, "❌ Invalid username/email or password")
            return redirect("login")

    return render(request, "login.html")

@login_required
def home(request, session_id=None):
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
    active_session = None
    messages = []

    if session_id:
        try:
            active_session = ChatSession.objects.get(id=session_id, user=request.user)
            messages = active_session.messages.all().order_by('timestamp')
        except ChatSession.DoesNotExist:
            messages.error(request, "Chat session not found.")
            return redirect('home')

    if request.method == "POST":
        prompt = request.POST.get("prompt")
        if not prompt:
            messages.error(request, "Prompt cannot be empty.")
            # Redirect back to the same page to show the error
            return redirect(request.path or 'home')

        # If there's no active session, create a new one
        if not active_session:
            active_session = ChatSession.objects.create(user=request.user)

        # Save the user's message
        ChatMessage.objects.create(session=active_session, role='user', content=prompt)

        # --- AI LOGIC GOES HERE ---
        # This is where you would call the actual GPT API.
        # For now, we'll just create a placeholder response.
        ai_response = f"This is a placeholder AI response to: '{prompt}'"
        # --------------------------

        # Save the AI's response
        ChatMessage.objects.create(session=active_session, role='assistant', content=ai_response)

        # Redirect to the same chat session to display the new messages
        return redirect('chat_session', session_id=active_session.id)

    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'messages': messages,
    }
    return render(request, "home.html", context)