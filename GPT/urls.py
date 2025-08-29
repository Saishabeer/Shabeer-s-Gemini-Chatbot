from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),  # Main page, new chat
    path('chat/<int:session_id>/', views.home, name='chat_session'), # Existing chat
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    # Use Django's built-in view for logging out.
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]