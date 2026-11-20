import time
from django.core.management import call_command
from django.core.management.base import BaseCommand
from assistant.models import Job


class Command(BaseCommand):
    help = 'Обработать текущую очередь без постоянно работающего воркера'
    def handle(self, *args, **kwargs):
        while Job.objects.filter(state='queued', attempts__lt=3).exists():
            call_command('worker', once=True)
            remaining = Job.objects.filter(state='queued', attempts__lt=3).count()
            self.stdout.write(f'Осталось заданий: {remaining}')
            if remaining:
                time.sleep(2)
