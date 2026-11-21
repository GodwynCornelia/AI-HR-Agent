"""Техническая трасса не переносит известные рабочие секреты и явные credential-строки."""
import os,re

def redact(value):
    if isinstance(value,dict):return {k:redact(v) for k,v in value.items()}
    if isinstance(value,list):return [redact(v) for v in value]
    if not isinstance(value,str):return value
    for key in ['DJANGO_SECRET_KEY','SERVICE_SECRET','POSTGRES_PASSWORD']:
        secret=os.getenv(key,'')
        if len(secret)>=8:value=value.replace(secret,'[REDACTED]')
    value=re.sub(r'(?i)\b(?:bearer\s+)[A-Za-z0-9._-]+','Bearer [REDACTED]',value)
    value=re.sub(r'(?i)\b(password|пароль|token|токен|api[_ -]?key)\s*[:=]\s*\S+',r'\1=[REDACTED]',value)
    return value
