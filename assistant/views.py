import os
import httpx

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import UploadForm
from .models import Conversation, Document
from .security import issue_context


def api_call(user, path, body=None, files=None):
    token = issue_context(user, 'POST', path)
    with httpx.Client(timeout=900) as client:
        payload={'json':body or {}} if files is None else {'data':body or {},'files':files}
        response = client.post(os.getenv('API_URL', 'http://127.0.0.1:8001') + path,
                               headers={'X-Service-Context': token}, **payload)
        if response.status_code in {401, 403, 404, 409, 413, 422}:
            raise ValueError(response.json().get('detail', 'Запрос отклонён'))
        response.raise_for_status()
        return response.json()


@login_required
def index(request, conversation_id=None):
    conversations = Conversation.objects.filter(owner=request.user).order_by('-created_at')
    selected = get_object_or_404(Conversation, id=conversation_id, owner=request.user) if conversation_id else None
    turns = selected.turns.order_by('created_at', 'id') if selected else []
    return render(request, 'assistant/chat.html', {'conversations': conversations, 'selected': selected, 'turns': turns})


@login_required
@require_POST
def new_conversation(request):
    conversation = Conversation.objects.create(owner=request.user)
    return redirect('chat', conversation_id=conversation.id)


@login_required
@require_POST
def send(request, conversation_id):
    get_object_or_404(Conversation, id=conversation_id, owner=request.user)
    try:
        api_call(request.user, f'/conversations/{conversation_id}/ask', {'message': request.POST.get('message', '')})
    except ValueError as exc:
        messages.error(request, str(exc))
    except httpx.HTTPError:
        messages.error(request, 'Сервис обработки недоступен. Сообщение не отправлено; повторите позже.')
    return redirect('chat', conversation_id=conversation_id)


@login_required
@user_passes_test(lambda user: user.is_staff)
def documents(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data.copy()
        file = data.pop('file')
        title = data.pop('title')
        try:
            metadata={'title':title}
            for key,value in data.items():
                if value is None:
                    continue
                metadata[key]=(str(value.pk) if key=='supersedes' else value.isoformat() if hasattr(value,'isoformat') else
                               'true' if value is True else 'false' if value is False else str(value))
            file.seek(0)
            api_call(request.user,'/documents/upload',metadata,
                     files={'file':(file.name,file.file,file.content_type or 'application/octet-stream')})
            messages.success(request, 'Документ добавлен в очередь обработки.')
            return redirect('documents')
        except ValueError as exc:
            form.add_error('file', str(exc))
        except httpx.HTTPError:
            form.add_error(None,'Сервис обработки документов недоступен. Файл не поставлен в очередь.')
    return render(request, 'assistant/documents.html', {
        'form': form, 'documents': Document.objects.exclude(state='deleted').order_by('-created_at')})


@login_required
@require_POST
def remove_document(request, document_id):
    if not request.user.is_staff:
        raise Http404
    try:
        api_call(request.user, f'/documents/{document_id}/delete')
        messages.success(request, 'Документ исключён из поиска. Очистка индекса поставлена в очередь.')
    except ValueError as exc:
        messages.error(request, str(exc))
    except httpx.HTTPError:
        messages.error(request, 'Сервис обработки недоступен. Удаление не подтверждено.')
    return redirect('documents')


@login_required
def source(request, document_id):
    document = get_object_or_404(Document, id=document_id, state='ready')
    try:
        return FileResponse(document.file.open('rb'), as_attachment=True, filename=document.title + os.path.splitext(document.file.name)[1])
    except FileNotFoundError:
        raise Http404
