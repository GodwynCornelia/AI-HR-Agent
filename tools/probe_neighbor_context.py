"""Controlled context probe on real corpus chunks, separate from natural search."""
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
from assistant.agent_graph import read_policy_fragment
from assistant.grounded import assess
from assistant.inference import Ollama
from assistant.services import retrieve

question='Кто согласует командировку и что делать при срочной поездке?'
model=Ollama()
ranked=retrieve(question,model)
anchor=ranked[0]
neighbors=read_policy_fragment(anchor['id'],{anchor['id']}) or []
rows=[]
for phase,scope in [('anchor',[anchor]),('neighbors',[anchor,*neighbors])]:
    diagnostic={}
    verdict,provided,proofs,_=assess(model,question,scope,diagnostic)
    rows.append(dict(phase=phase,verdict=verdict,provided=provided,
                     citations=[proofs[i] for i in verdict['proof_ids']]))
output=ROOT/'runtime/agent-v2'/f'neighbor-{uuid4()}.json'
output.write_text(json.dumps(dict(question=question,anchor=anchor,neighbors=neighbors,phases=rows),ensure_ascii=False,indent=2),encoding='utf-8')
print(output)
