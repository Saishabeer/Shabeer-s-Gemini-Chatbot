from django.db import models
from django.contrib.auth.models import User
import os


class ChatSession(models.Model):
    """Represents a single chat conversation."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    document_name = models.CharField(max_length=255, blank=True, null=True)
    document_path = models.CharField(max_length=512, blank=True, null=True)

    def __str__(self):
        # Prioritize document name for the title
        if self.document_name:
            # Return just the filename, not the whole path
            return os.path.basename(self.document_name)

        # Fallback to existing logic
        first_message = self.messages.filter(role='user').first()
        return first_message.content[:50] if first_message else f"Chat {self.id}"


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
