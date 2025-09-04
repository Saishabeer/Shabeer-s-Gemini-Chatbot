from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _

class UserManager(BaseUserManager):
    """Custom user model manager where email is the unique identifier."""
    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        """
                A private helper method to create and save a user with the given email and password.
                It's used by both create_user and create_superuser.
                """
        # 1. Check if an email was provided. If not, raise an error.
        if not email:
            raise ValueError('The Email must be set')

        # 2. Normalize the email (e.g., converts the domain part to lowercase).
        email = self.normalize_email(email)
        # 3. Create a new user instance.
        user = self.model(email=email, **extra_fields)
        # 4. Set the password. This correctly hashes the password for security.
        user.set_password(password)
        # 5. Save the user object to the database.
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        """
        Public method to create a regular user.
        """
        # Sets default values for a regular user. They are not staff or superusers.
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        # Calls the internal _create_user method to do the actual work.
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        """
        Public method to create a superuser (admin).
        """
        # Sets default values for a superuser. They must be staff and a superuser.
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        # Adds extra validation to ensure a superuser has the correct permissions.
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        # Calls the internal _create_user method to create the superuser.
        return self._create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Custom user model that uses email as the unique identifier."""
    username = None
    email = models.EmailField(_('email address'), unique=True)

    # Add custom related names to avoid clash with auth.User
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name=_('groups'),
        blank=True,
        help_text=(
            'The groups this user belongs to. A user will get all permissions '
            'granted to each of their groups.'
        ),
        related_name='custom_user_set',
        related_query_name='custom_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name=_('user permissions'),
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='custom_user_set',
        related_query_name='custom_user',
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = UserManager()

    def __str__(self):
        return self.email

class ChatSession(models.Model):
    """Represents a single chat conversation."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    document_name = models.CharField(max_length=255, blank=True, null=True)
    document_content = models.BinaryField(blank=True, null=True)  # Store file content
    content_type = models.CharField(max_length=100, blank=True, null=True)  # Store file MIME type
    title = models.CharField(max_length=100, default='New Chat')

    def __str__(self):
        return self.title

    def save_document(self, uploaded_file):
        """Save file content to database."""
        self.document_name = uploaded_file.name
        self.content_type = uploaded_file.content_type
        self.document_content = uploaded_file.read()
        self.save()
        return self

class ChatMessage(models.Model):
    """Represents a single message within a chat session."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:50]}"
