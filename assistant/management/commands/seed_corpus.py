import json
import secrets
from datetime import date
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError

from assistant.models import Document
from assistant.services import upload_document
from assistant.dataset import dataset_directory


class Command(BaseCommand):
    help = 'Загрузить подготовленный учебный корпус в очередь, без копирования бенчмарка'

    def add_arguments(self, parser):
        parser.add_argument('--admin', default='local_admin')
        parser.add_argument('--create-demo', action='store_true')

    def handle(self, *args, **options):
        root = settings.BASE_DIR
        user_model = get_user_model()
        if options['create_demo']:
            credentials = {}
            for username, staff in [(options['admin'], True), ('employee', False)]:
                if not user_model.objects.filter(username=username).exists():
                    password = secrets.token_urlsafe(18)
                    user_model.objects.create_user(username, password=password, is_staff=staff, is_superuser=staff)
                    credentials[username] = password
            if credentials:
                directory = root / 'runtime'
                directory.mkdir(exist_ok=True)
                (directory / 'local_accounts.json').write_text(json.dumps(credentials, indent=2), encoding='utf-8')
                self.stdout.write('Учётные записи созданы; пароли сохранены только в runtime/local_accounts.json (исключён из Git).')
        try:
            admin = user_model.objects.get(username=options['admin'], is_staff=True)
        except user_model.DoesNotExist:
            raise CommandError('Создайте администратора либо используйте --create-demo')
        data=dataset_directory(root)
        manifest = json.loads((data / 'corpus_manifest.json').read_text(encoding='utf-8'))
        added = {}
        for item in manifest:
            doc = Document.objects.filter(sha256=item['sha256']).exclude(state='deleted').first()
            if not doc:
                file = SimpleUploadedFile(item['file'], (data / 'corpus' / item['file']).read_bytes())
                doc = upload_document(admin, file, item['title'], scope=item['scope'],
                                      effective_from=date.fromisoformat(item['effective_from']) if item['effective_from'] else None,
                                      provenance=item['provenance'], synthetic=item['synthetic'])
            added[item.get('id',item['file'])] = doc
        for item in manifest:
            if item['supersedes']:
                doc = added[item.get('id',item['file'])]
                doc.supersedes = added[item['supersedes']]
                doc.save(update_fields=['supersedes'])
        self.stdout.write(f'Документов в манифесте: {len(added)}. Запустите воркер для индексирования.')
