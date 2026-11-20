import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hr_project.settings')
import django
django.setup()

from django.contrib.auth import get_user_model
from django.core import signing
from django.core.exceptions import ObjectDoesNotExist
from django.db import close_old_connections
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import date
from uuid import UUID
from pathlib import Path
from django.core.files.uploadedfile import UploadedFile as DjangoUploadedFile

from .models import Conversation, Document
from .security import verify_context, require_admin
from .services import ask, delete_document, upload_document, MAX_FILE

app = FastAPI(title='Локальный HR RAG API', docs_url=None, redoc_url=None)


def current_user(request: Request, x_service_context: str = Header(default='')):
    close_old_connections()
    try:
        return verify_context(x_service_context, request.method, request.url.path)
    except (signing.BadSignature, ObjectDoesNotExist, KeyError, ValueError):
        raise HTTPException(401, 'Недействительный контекст доступа')


class Question(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get('/health')
def health():
    return {'status': 'alive'}


@app.post('/conversations/{conversation_id}/ask')
def answer(conversation_id: str, body: Question, user=Depends(current_user)):
    try:
        turn = ask(user, conversation_id, body.message)
        return {'turn_id': str(turn.id), 'status': turn.status}
    except (ObjectDoesNotExist, django.core.exceptions.ValidationError):
        raise HTTPException(404, 'Диалог не найден')
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post('/documents/{document_id}/delete')
def remove(document_id: str, user=Depends(current_user)):
    try:
        document = delete_document(user, document_id)
        return {'state': document.state}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    except (ObjectDoesNotExist, django.core.exceptions.ValidationError):
        raise HTTPException(404, 'Документ не найден')


@app.post('/documents/upload')
def add_document(file:UploadFile=File(...), title:str=Form(...,min_length=1,max_length=240),
                 scope:str=Form('Все сотрудники',max_length=240), effective_from:date|None=Form(None),
                 supersedes:UUID|None=Form(None), provenance:str=Form(...,min_length=1),
                 synthetic:bool=Form(True), user=Depends(current_user)):
    try:
        require_admin(user)
        if file.size is not None and file.size>MAX_FILE:
            raise HTTPException(413,'Размер файла превышает 20 MiB')
        filename=file.filename or ''
        if not filename or filename!=Path(filename).name or '\\' in filename:
            raise HTTPException(422,'Требуется имя файла без пути')
        previous=Document.objects.exclude(state='deleted').get(pk=supersedes) if supersedes else None
        wrapped=DjangoUploadedFile(file=file.file,name=filename,content_type=file.content_type,size=file.size)
        document=upload_document(user,wrapped,title,scope=scope,effective_from=effective_from,
                                 supersedes=previous,provenance=provenance,synthetic=synthetic)
        return {'document_id':str(document.id),'state':document.state}
    except PermissionError as error:
        raise HTTPException(403,str(error))
    except ObjectDoesNotExist:
        raise HTTPException(404,'Заменяемый документ не найден')
    except ValueError as error:
        raise HTTPException(422,str(error))
    finally:
        file.file.close()
        close_old_connections()
