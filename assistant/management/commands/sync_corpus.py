"""Загрузка принятого корпуса; вывод из поиска только известного прототипа."""
import hashlib
import json
import os
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from assistant.dataset import dataset_directory
from assistant.models import Document, Job
from assistant.services import delete_document


class Command(BaseCommand):
    help='Синхронизировать активный учебный корпус, сохраняя незнакомые пользовательские документы'
    def add_arguments(self,parser):
        parser.add_argument('--admin',default='local_admin')
        parser.add_argument('--retire-known-prototype',action='store_true')
        parser.add_argument('--index',action='store_true')
    def handle(self,*args,**options):
        data=dataset_directory(settings.BASE_DIR)
        manifest=json.loads((data/'corpus_manifest.json').read_text(encoding='utf-8'))
        expected={item['sha256'] for item in manifest}
        admin=get_user_model().objects.get(username=options['admin'],is_staff=True,is_active=True)
        legacy_path=settings.BASE_DIR/'data/corpus_manifest.json'
        legacy={item['sha256'] for item in json.loads(legacy_path.read_text(encoding='utf-8'))}
        retired=[]
        if options['retire_known_prototype']:
            for document in Document.objects.exclude(state__in=['deleted','deleting']).filter(sha256__in=legacy-expected):
                delete_document(admin,document.id)
                retired.append(str(document.id))
        unknown=list(Document.objects.filter(state='ready').exclude(sha256__in=expected|legacy).values_list('title',flat=True))
        if unknown:
            raise CommandError('Обнаружены документы вне принятого корпуса; они сохранены: '+str(unknown))
        call_command('seed_corpus',admin=options['admin'])
        if options['index']:
            # Только локальный процесс индексирования; постоянные настройки ОС не меняются.
            os.environ['EMBED_KEEP_ALIVE_SECONDS']='120'
            call_command('index_all')
        ready=set(Document.objects.filter(state='ready').values_list('sha256',flat=True))
        report={'expected_documents':len(expected),'ready_expected':len(ready&expected),'missing_hashes':sorted(expected-ready),
                'extra_ready_hashes':sorted(ready-expected),'retired_prototype_ids':retired,
                'job_errors':list(Job.objects.filter(state='error').values('id','kind','error'))}
        output=settings.BASE_DIR/'reports/verification'
        output.mkdir(parents=True,exist_ok=True)
        (output/'corpus_sync.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.stdout.write(json.dumps(report,ensure_ascii=False))
        if options['index'] and (expected!=ready):
            raise CommandError('Индекс пока не соответствует принятому манифесту; см. corpus_sync.json')
