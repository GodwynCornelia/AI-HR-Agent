import time
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import close_old_connections, transaction
from django.utils import timezone

from assistant.limits import heavy_operation
from assistant.models import Document, Job
from assistant.services import process_job


class Command(BaseCommand):
    help = 'Один воркер индексации и удаления с повторяемыми заданиями PostgreSQL'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true')

    def handle(self, *args, **options):
        while True:
            close_old_connections()
            try:
                with heavy_operation():
                    # Глобальная блокировка гарантирует, что активного старого воркера нет.
                    Job.objects.filter(state='running').update(state='queued')
                    Job.objects.filter(state='queued', attempts__gte=3).update(state='error', error='Превышено число попыток после сбоя воркера')
                    with transaction.atomic():
                        job = Job.objects.select_for_update(skip_locked=True).filter(state='queued', attempts__lt=3).order_by('created_at').first()
                        if job:
                            job.state, job.started_at = 'running', timezone.now()
                            job.attempts += 1
                            job.save()
                    if job:
                        try:
                            process_job(job)
                            job.state, job.error = 'done', ''
                        except Exception as exc:
                            job.state = 'queued' if job.attempts < 3 else 'error'
                            job.error = f'{type(exc).__name__}: {exc}'[:1000]
                            Document.objects.filter(pk=job.document_id).exclude(state__in=['deleting', 'deleted']).update(state='error', error=job.error)
                        job.save()
            except RuntimeError:
                pass  # Занято генерацией; задание остаётся в очереди.
            if options['once']:
                return
            time.sleep(2)
