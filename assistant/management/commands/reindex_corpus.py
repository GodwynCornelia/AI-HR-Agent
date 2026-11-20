from django.core.management.base import BaseCommand
from django.db import transaction
from assistant.models import Document, Job


class Command(BaseCommand):
    help = 'Повторно поставить активные документы в очередь после изменения конфигурации embeddings'
    def handle(self, *args, **kwargs):
        with transaction.atomic():
            documents = Document.objects.select_for_update().exclude(state__in=['deleted', 'deleting'])
            count = 0
            for document in documents:
                if Job.objects.filter(document=document, kind='index', state__in=['queued', 'running']).exists():
                    continue
                document.state = 'queued'
                document.save(update_fields=['state', 'updated_at'])
                Job.objects.create(document=document, kind='index')
                count += 1
        self.stdout.write(f'Новых заданий переиндексации: {count}')
