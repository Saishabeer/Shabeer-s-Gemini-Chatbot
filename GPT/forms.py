from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm

User = get_user_model()


class UserRegistrationForm(forms.Form):
    """
    A more robust registration form using forms.Form for explicit field control,
    avoiding potential ModelForm conflicts with the User model's password field.
    """
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'placeholder': 'Username'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'placeholder': 'Email'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Password'})
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password'})
    )

    def clean_username(self):
        """Ensure the username is unique (case-insensitive)."""
        username = self.cleaned_data.get('username')
        if username and User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError('A user with this username already exists.')
        return username

    def clean_email(self):
        """Ensure the email is unique (case-insensitive)."""
        email = self.cleaned_data.get('email')
        if email and User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('User already exist.')
        return email

    def clean(self):
        """Check if passwords match."""
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password2 = cleaned_data.get("password2")

        if password and password2 and password != password2:
            self.add_error('password2', "Passwords do not match.")

        return cleaned_data

    def save(self, commit=True):
        """Creates the user using the cleaned data."""
        # Using create_user is a robust way to create a new user as it handles password hashing.
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password']
        )
        return user


class UserLoginForm(AuthenticationForm):
    """Custom login form to add placeholders and allow email login."""
    username = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Username or Email', 'id': 'username'}))
    password = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password', 'id': 'password'}))