from django.conf import settings
from django.core import signing
from django.contrib.auth import get_user_model


def issue_context(user, method, path):
    return signing.dumps({'uid': user.pk, 'aud': 'hr-rag-api', 'method': method, 'path': path},
                         key=settings.SERVICE_SECRET, salt='hr-service-v1')


def verify_context(token, method, path):
    claims = signing.loads(token, key=settings.SERVICE_SECRET, salt='hr-service-v1', max_age=60)
    if (claims.get('aud'), claims.get('method'), claims.get('path')) != ('hr-rag-api', method, path):
        raise signing.BadSignature('Неверное назначение контекста')
    return get_user_model().objects.get(pk=claims['uid'], is_active=True)


def owned_conversation(user, conversation_id):
    from .models import Conversation
    return Conversation.objects.get(pk=conversation_id, owner=user)


def require_admin(user):
    if not user.is_active or not user.is_staff:
        raise PermissionError('Операция доступна администратору')
