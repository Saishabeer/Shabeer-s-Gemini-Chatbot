from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from .models import ChatSession, ChatMessage
from .gemini_service import gemini_chat  # ✅ Using Gemini

# ------------------ REGISTER ------------------
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

# ------------------ LOGIN ------------------
def user_login(request):
    if request.method == "POST":
        username_or_email = request.POST.get("username")
        password = request.POST.get("password")

        if not username_or_email or not password:
            messages.error(request, "Please enter a username/email and password.")
            return redirect('login')

        user = authenticate(request, username=username_or_email, password=password)

        if user is None:
            try:
                user_by_email = User.objects.get(email=username_or_email)
                user = authenticate(request, username=user_by_email.username, password=password)
            except User.DoesNotExist:
                pass

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("home")
            messages.error(request, "⚠️ Your account is inactive.")
        else:
            messages.error(request, "❌ Invalid username/email or password")

        return redirect("login")

    return render(request, "login.html")

# ------------------ CHAT HOME ------------------
@login_required
def home(request, session_id=None):
    chat_sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
    active_session = None
    chat_messages = []

    # Load an existing chat if session_id provided
    if session_id:
        try:
            active_session = ChatSession.objects.get(id=session_id, user=request.user)
            chat_messages = active_session.messages.all().order_by('timestamp')
        except ChatSession.DoesNotExist:
            messages.error(request, "Chat session not found.")
            return redirect('home')

    if request.method == "POST":
        prompt = request.POST.get("prompt")
        if not prompt:
            messages.error(request, "Prompt cannot be empty.")
            return redirect(request.path or 'home')

        # Create a new session if none yet
        if not active_session:
            active_session = ChatSession.objects.create(user=request.user)

        # Save user's message
        ChatMessage.objects.create(session=active_session, role='user', content=prompt)

        # Build history in our schema (role + content)
        history = [
            {"role": m.role, "content": m.content}
            for m in active_session.messages.all().order_by("timestamp")
        ]

        # Get Gemini's reply
        ai_response = gemini_chat(prompt, history)

        # Save assistant message
        ChatMessage.objects.create(session=active_session, role='assistant', content=ai_response)

        return redirect('chat_session', session_id=active_session.id)

    context = {
        'chat_sessions': chat_sessions,
        'active_session': active_session,
        'messages': chat_messages,
    }
    return render(request, "home.html", context)
