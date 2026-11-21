"""Сравнение режимов при одинаковой пустой первой выдаче."""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hr_project.settings')
import django
django.setup()
from django.contrib.auth import get_user_model
from assistant import services
from assistant.models import Conversation

case = json.loads((ROOT / 'experiments/agent_v2/protocol.json').read_text(encoding='utf-8'))['cases'][0]
user, _ = get_user_model().objects.get_or_create(username='local_agent_probes')
retrieve = services.retrieve
output = ROOT / 'runtime/agent-v2' / str(uuid4())
output.mkdir(parents=True)
for mode in ('rag', 'agent'):
    calls = []
    def first_empty(query, model):
        results = retrieve(query, model)
        returned = results if calls else []
        calls.append({'query': query, 'returned': returned, 'original_count': len(results)})
        return returned
    conversation = Conversation.objects.create(owner=user, mode=mode, title='Empty search probe')
    start = time.perf_counter()
    with patch.object(services, 'retrieve', side_effect=first_empty):
        for attempt in range(10):
            try:
                turn = services.ask(user, conversation.id, case['turns'][0], as_of='2026-11-21')
                break
            except RuntimeError as error:
                if 'занята' not in str(error) or attempt == 9:
                    raise
                time.sleep(1)
    result = dict(mode=mode, question=turn.original, status=turn.status, answer=turn.answer,
                  seconds=time.perf_counter()-start, calls=calls, diagnostics=turn.diagnostics)
    (output / (mode+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output.name, mode, turn.status, len(calls), flush=True)
