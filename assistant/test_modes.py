from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from .models import Conversation

class ConversationModeTests(TestCase):
    def setUp(self):self.user=get_user_model().objects.create_user('mode-owner');self.client.force_login(self.user)
    def test_mode_selected_at_creation_and_not_changeable(self):
        response=self.client.post('/conversations/new/',{'mode':'agent'})
        self.assertEqual(response.status_code,302)
        conversation=Conversation.objects.get(owner=self.user);self.assertEqual(conversation.mode,'agent')
        conversation.mode='rag'
        with self.assertRaises(ValidationError):conversation.save()
    def test_unknown_mode_not_created(self):
        self.client.post('/conversations/new/',{'mode':'shell'})
        self.assertFalse(Conversation.objects.exists())
    def test_default_mode_remains_rag(self):
        self.client.post('/conversations/new/')
        self.assertEqual(Conversation.objects.get(owner=self.user).mode,'rag')
