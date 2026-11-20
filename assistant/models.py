import uuid

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError


class Document(models.Model):
    def __str__(self):
        return self.title

    class State(models.TextChoices):
        QUEUED = 'queued', 'В очереди'
        PROCESSING = 'processing', 'Обрабатывается'
        READY = 'ready', 'Готов'
        ERROR = 'error', 'Ошибка'
        DELETING = 'deleting', 'Удаляется'
        DELETED = 'deleted', 'Удалён'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=240)
    file = models.FileField(upload_to='documents/%Y/%m/')
    sha256 = models.CharField(max_length=64)
    state = models.CharField(max_length=20, choices=State.choices, default=State.QUEUED)
    scope = models.CharField(max_length=240, default='Все сотрудники')
    effective_from = models.DateField(null=True, blank=True)
    supersedes = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL)
    provenance = models.TextField(blank=True)
    synthetic = models.BooleanField(default=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Chunk(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    ordinal = models.PositiveIntegerField()
    text = models.TextField()
    location = models.CharField(max_length=240)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['document', 'ordinal'], name='unique_document_chunk')]


class Job(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    kind = models.CharField(max_length=10, choices=[('index', 'Индексировать'), ('delete', 'Удалить')])
    state = models.CharField(max_length=12, default='queued')
    attempts = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Conversation(models.Model):
    class Mode(models.TextChoices):
        RAG='rag','Обычный RAG'
        AGENT='agent','LangGraph Agent'
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=120, default='Новый диалог')
    mode = models.CharField(max_length=16,choices=Mode.choices,default=Mode.RAG)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self,*args,**kwargs):
        if not self._state.adding:
            previous=type(self).objects.filter(pk=self.pk).values_list('mode',flat=True).first()
            if previous is not None and previous!=self.mode:raise ValidationError('Режим существующего диалога менять нельзя; создайте новый диалог.')
        return super().save(*args,**kwargs)


class Turn(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='turns')
    original = models.TextField()
    rewritten = models.TextField(blank=True)
    answer = models.TextField(blank=True)
    status = models.CharField(max_length=20, default='pending')
    diagnostics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)


class ExperimentRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    configuration = models.JSONField()
    results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
