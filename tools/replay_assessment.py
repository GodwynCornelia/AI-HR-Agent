"""Replay the shared evidence selector on previously supplied dev context."""
import argparse
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
from assistant.grounded import assess
from assistant.inference import Ollama

parser=argparse.ArgumentParser()
parser.add_argument('answers',type=Path)
opts=parser.parse_args()
rows=[json.loads(line) for line in opts.answers.read_text(encoding='utf-8').splitlines()]
out=ROOT/'runtime/agent-v2'/f'replay-{uuid4()}.json'
results=[]
for row in rows:
    if row['mode']!='rag' or row['question_id'] not in {'v4.onboarding_plan.mentor','cross.learning_trip'}:
        continue
    before=row['diagnostics']
    sources=before['provided_sources']
    diagnostics={}
    value,_,proofs,_=assess(Ollama(),row['rewritten'],sources,diagnostics)
    results.append({'id':row['question_id'],'question':row['question'],'before':before['assessment'],
                    'after':value,'quotes':[proofs[i]['quote'] for i in value['proof_ids']],
                    'fallback_attempted':'assessment_initial_empty' in diagnostics})
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(row['question_id'],value['sufficient'],len(value['proof_ids']),flush=True)
print(out)
