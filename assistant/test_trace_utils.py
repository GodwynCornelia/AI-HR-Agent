import os
from unittest.mock import patch
from django.test import SimpleTestCase
from .trace_utils import redact

class TracePrivacyTests(SimpleTestCase):
    def test_known_credentials_removed_without_changing_input(self):
        original={'query':'пароль: demo-value','nested':['test-server-secret-not-real'],'count':3}
        with patch.dict(os.environ,{'SERVICE_SECRET':'test-server-secret-not-real'}):clean=redact(original)
        self.assertNotIn('demo-value',clean['query']);self.assertEqual(clean['nested'],['[REDACTED]'])
        self.assertEqual(original['nested'],['test-server-secret-not-real']);self.assertEqual(clean['count'],3)
