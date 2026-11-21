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
    filler = {'а', 'и', 'если', 'другой', 'иной', 'вариант', 'так', 'также', 'же', 'это', 'тоже', 'можно', 'ли', 'нельзя',
              'тогда', 'иначе', 'по', 'другому', 'нет', 'да', 'сделать', 'поступить'}
    return bool(words) and set(words) <= filler and bool(set(words) & {'так', 'также', 'иначе', 'другому','другой','иной'})


def needs_reference_repair(current,query,status,history,*,check_current=True):
    if not history or unspecified_alternative(current):return False
    unchanged=query.strip().rstrip('?.!').lower()==current.strip().rstrip('?.!').lower()
    if re.match(r'^\s*(?:я\s+имею\s+в\s+виду|речь\s+о)\b',current,re.I) and (status=='clarify' or unchanged):return True
    reference_text=(current+' '+query) if check_current else query
    deictic=bool(re.search(r'\b(?:этой|эта|этого|эту|такой|этот)\s+(?:сумм|вариант|результат|услов|срок)',reference_text,re.I))
    named_words=[w for w in re.findall(r'[а-яё]{4,}',current.lower()) if w not in {'если','этого','этой','этот','тогда','такой','него','неё','вариант','другой','другие','условия'}]
    if status=='clarify':return deictic or bool(re.match(r'^\s*а\s+',current,re.I) and named_words)
    elliptical=unchanged and bool(re.match(r'^\s*а\s+',query,re.I)) and len(query.split())<=14
    return deictic or elliptical


def baseline_messages(history, current):
    """Доступна та же собственная история; последнее сообщение отделено явно."""
    return [{'role': 'user', 'content': json.dumps({'history': history, 'current': current}, ensure_ascii=False)}]
