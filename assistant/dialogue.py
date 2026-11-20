"""Общие проверки недоопределённых продолжений и пакет контекста без RAG."""
import json
import re


def explicit_topic_reset(message):
    # Явная граница новой темы; «а для офисных?» продолжает прежний вопрос.
    return bool(re.match(r'^\s*(?:а\s+)?теперь\s+(?:другая\s+тема\s*:|(?:о|про)\s+[^:?.!]{5,}:)',message,re.I))


def inferential_followup(message, previous_query):
    # Отсылка к следствию известного события отличается от неназванного варианта.
    if not previous_query or not re.match(r'^\s*(?:это\s+(?:означает|значит)|из\s+этого\s+следует|такой\s+(?:результат|исход)\s+(?:означает|значит))\b',message,re.I):
        return None
    query=message+' Контекст предыдущего вопроса: '+previous_query
    return query if len(query)<=900 else None


def unspecified_alternative(message):
    # Только отсылка без названного действия/условия. Не запрещает «А для офисных?».
    words = re.findall(r'[а-яёa-z]+', message.lower())
    filler = {'а', 'и', 'так', 'также', 'же', 'это', 'тоже', 'можно', 'ли', 'нельзя',
              'тогда', 'иначе', 'по', 'другому', 'нет', 'да', 'сделать', 'поступить'}
    return bool(words) and set(words) <= filler and bool(set(words) & {'так', 'также', 'иначе', 'другому'})


def baseline_messages(history, current):
    """Доступна та же собственная история; последнее сообщение отделено явно."""
    return [{'role': 'user', 'content': json.dumps({'history': history, 'current': current}, ensure_ascii=False)}]
