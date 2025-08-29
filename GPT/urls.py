from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # A single view handles the home page (new chat) and existing chat sessions.
    path('', views.chat_view, name='home'),
    path("chat/<int:session_id>/", views.chat_view, name="chat_session"),

    # Auth URLs
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path("chat/<int:session_id>/delete/", views.delete_chat_session, name="delete_chat_session"),
]