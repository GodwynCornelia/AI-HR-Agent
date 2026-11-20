from datetime import date
from types import SimpleNamespace
from unittest.mock import patch
from django.test import SimpleTestCase
from fastapi.testclient import TestClient
from assistant.api import app,current_user


class UploadApiTests(SimpleTestCase):
    def tearDown(self):
        app.dependency_overrides.clear()

    def test_admin_upload_keeps_bytes_and_typed_metadata_in_processing_api(self):
        admin=SimpleNamespace(is_active=True,is_staff=True)
        app.dependency_overrides[current_user]=lambda:admin
        observed={}
        def accept(user,uploaded,title,**metadata):
            observed.update(user=user,filename=uploaded.name,body=b''.join(uploaded.chunks()),title=title,metadata=metadata)
            return SimpleNamespace(id='qa-document',state='queued')
        with patch('assistant.api.upload_document',side_effect=accept),TestClient(app) as client:
            response=client.post('/documents/upload',data={'title':'Учебный регламент','scope':'Офисные сотрудники',
                'effective_from':'2026-11-17','provenance':'Авторский QA-текст','synthetic':'true'},
                files={'file':('регламент.txt','Проверочный текст'.encode('utf-8'),'text/plain')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(observed['body'],'Проверочный текст'.encode('utf-8'))
        self.assertEqual(observed['filename'],'регламент.txt')
        self.assertEqual(observed['metadata']['effective_from'],date(2026,11,17))
        self.assertTrue(observed['metadata']['synthetic'])

    def test_employee_cannot_upload_by_spoofing_form_role(self):
        app.dependency_overrides[current_user]=lambda:SimpleNamespace(is_active=True,is_staff=False)
        with patch('assistant.api.upload_document') as processing,TestClient(app) as client:
            response=client.post('/documents/upload',data={'title':'QA','provenance':'QA','is_staff':'true'},
                                 files={'file':('qa.txt',b'qa','text/plain')})
        self.assertEqual(response.status_code,403)
        processing.assert_not_called()

    def test_api_size_limit_rejects_before_creating_a_document(self):
        app.dependency_overrides[current_user]=lambda:SimpleNamespace(is_active=True,is_staff=True)
        with patch('assistant.api.MAX_FILE',8),patch('assistant.api.upload_document') as processing,TestClient(app) as client:
            response=client.post('/documents/upload',data={'title':'QA','provenance':'QA'},
                                 files={'file':('qa.txt',b'123456789','text/plain')})
        self.assertEqual(response.status_code,413)
        processing.assert_not_called()
