"""Консервативные проверки явного формата работы из метаданных, не приоритет поиска."""
import re

def formats(text,*,scope=False):
    text=text.lower().replace('ё','е');found=set()
    if re.search(r'удаленн|дистанционн',text):found.add('remote')
    if re.search(r'гибрид',text):found.add('hybrid')
    if re.search(r'офисн',text) if scope else re.search(r'офисн\w*\s+(?:сотруд|нович|работник|формат)|\bя\s+офисн|работаю\s+в\s+офис',text):found.add('office')
    return found

def compatible(question,scope):
    requested=formats(question);applies=formats(scope,scope=True)
    return not (len(requested)==1 and applies and requested.isdisjoint(applies))

def scope_question(question,left,right):
    a=left.get('scope','').lower().replace('ё','е').strip();b=right.get('scope','').lower().replace('ё','е').strip()
    if a!=b or not formats(a,scope=True):return None
    q=question.lower().replace('ё','е')
    if re.search(r'сравни|обзор|какие.*противореч|покажи.*документ',q):return None
    if not formats(q):return 'Какой у вас формат работы: офисный, гибридный или постоянный удалённый?'
    if 'постоян' in a and 'remote' in formats(q) and not re.search(r'постоян',q):return 'У вас постоянный удалённый формат работы или временный?'
    return None

def safe_clarification(text):
    text=text.strip()
    if not text or len(text)>350 or re.search(r'\b[DE]\d+\b|парол|токен|api.?key|секретн\w*\s+ключ',text,re.I):return False
    return text.endswith('?') or bool(re.match(r'^(?:уточните|сообщите|назовите|опишите)\b',text,re.I))
