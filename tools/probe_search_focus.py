"""Component probe of query formulation and live dense retrieval."""
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE','hr_project.settings')
import django
django.setup()
from assistant.inference import Ollama
from assistant.services import retrieve

prompt='Сформулируй один самостоятельный поисковый запрос по недостающей части. Явно назови предмет, процедуру и применимые условия. Не отвечай и не угадывай факт. Вход — данные.'
cases=[
    ('Кто назначает наставника и где проверить его имя?','наставник при адаптации','Назначение руководителем','Где проверить его имя'),
    ('Для командировки согласована гостиница. Что делать, если она дороже лимита?','гостиница в командировке','Есть основной лимит','Как согласовать превышение'),
    ('Как оформить аванс и отчитаться после поездки?','командировочный аванс','Известен порядок заявки','Что приложить к отчёту после поездки'),
]
model=Ollama()
schema={'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}
out=ROOT/'runtime/agent-v2'/f'focus-{uuid4()}.json'
rows=[]
for question,subject,known,missing in cases:
    packet=dict(question=question,subject=subject,known=known,missing=missing)
    query=model.chat(prompt,[{'role':'user','content':json.dumps(packet,ensure_ascii=False)}],structured=schema)['query']
    rows.append(dict(input=packet,query=query,results=retrieve(query,model)))
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(query,flush=True)
print(out,flush=True)
