from django.db import models
from django.contrib.auth.models import User


class ChatSession(models.Model):
    """Represents a single chat conversation."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    document_name = models.CharField(max_length=255, blank=True, null=True)
    document_path = models.CharField(max_length=512, blank=True, null=True)
    title = models.CharField(max_length=100, default='New Chat')

    def __str__(self):
        return self.title


class ChatMessage(models.Model):
    """Represents a single message within a chat session."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:50]}"
