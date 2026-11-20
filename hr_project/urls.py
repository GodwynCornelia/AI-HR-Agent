from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path
from assistant import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('', views.index, name='index'),
    path('conversations/new/', views.new_conversation, name='new_conversation'),
    path('conversations/<uuid:conversation_id>/', views.index, name='chat'),
    path('conversations/<uuid:conversation_id>/send/', views.send, name='send'),
    path('documents/', views.documents, name='documents'),
    path('documents/<uuid:document_id>/delete/', views.remove_document, name='delete_document'),
    path('sources/<uuid:document_id>/', views.source, name='source'),
]
