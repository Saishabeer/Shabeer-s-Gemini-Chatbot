from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

User = get_user_model()


class UserRegistrationForm(forms.ModelForm):
    """Custom registration form that uses email as the primary identifier."""
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'autocomplete': 'new-password',
            'required': True
        }),
        error_messages={
            'required': 'Please enter a password'
        }
    )
    
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Type your password again',
            'autocomplete': 'new-password',
            'required': True
        }),
        error_messages={
            'required': 'Please confirm your password'
        }
    )
    email = forms.EmailField(
        label='Email',
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email',
            'autocomplete': 'email',
            'required': True
        }),
        error_messages={
            'required': 'Please enter your email',
            'invalid': 'Please enter a valid email address',
            'unique': 'This email is already in use. Please use a different email or log in.'
        },
        help_text='Required. Enter a valid email address.'
    )
    
    first_name = forms.CharField(
        label='Display Name (optional)',
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Choose a display name (optional)',
            'autocomplete': 'name',
        }),
        help_text='Optional. This is how your name will appear in the system.'
    )
    
    class Meta:
        model = User
        fields = ('email', 'first_name')
    
    def clean_email(self):
        email = self.cleaned_data.get('email', '').lower().strip()
        if not email:
            raise forms.ValidationError('Email is required.')
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('A user with this email already exists.')
        return email
    
    def clean_first_name(self):
        first_name = self.cleaned_data.get('first_name', '').strip()
        if first_name and len(first_name) < 2:
            raise forms.ValidationError('Display Name must be at least 2 characters long.')
        return first_name
    
    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')
        
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        
        return cleaned_data

    def save(self, commit=True):
        """Create and return the new user."""
        user = User.objects.create_user(
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password1']
        )
        # If display name was provided, update it after user creation
        first_name = self.cleaned_data.get('first_name', '').strip()
        if first_name:
            user.first_name = first_name
            user.save()
        return user


class UserLoginForm(AuthenticationForm):
    """Custom login form that uses email for authentication."""
    username = forms.EmailField(
        label='Email',
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email',
            'autocomplete': 'email',
            'required': True
        }),
        error_messages={
            'required': 'Please enter your email',
            'invalid': 'Please enter a valid email address'
        }
    )
    password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password',
            'required': True
        }),
        error_messages={
            'required': 'Please enter your password'
        }
    )
    
    error_messages = {
        'invalid_login': (
            "Please enter a correct email and password. Note that both "
            "fields may be case-sensitive."
        ),
        'inactive': "This account is inactive.",
    }