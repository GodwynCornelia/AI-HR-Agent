"""Build the portfolio protocol from reviewed question/evidence specifications."""
import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('source_commit')
parser.add_argument('output',type=Path)
args=parser.parse_args()
catalog=json.loads((ROOT/'experiments/agent_dev/chunk_catalog.json').read_text(encoding='utf-8'))
specs=json.loads((ROOT/'experiments/portfolio/specifications.json').read_text(encoding='utf-8'))
past=json.loads((ROOT/'data/benchmark/questions.json').read_text(encoding='utf-8'))
normalize=lambda text:' '.join(re.findall(r'\w+',text.casefold()))
def extract(doc,clue):
    texts=[c['text'] for c in catalog.values() if c['document']==doc and clue in c['text']]
    sentences=[s.strip() for t in texts for s in re.split(r'(?<=[.!?])\s+|\n+',t) if clue in s]
    assert sentences,(doc,clue)
    return dict(document=doc,quote=min(sentences,key=len))

questions=[]
duplicates=[]
for j,scenario in enumerate(specs,1):
    group=f'portfolio.{j:02}'
    for k,(question,action,facts,refs) in enumerate(scenario['turns'],1):
        identifier=f'{group}.{k}'
        evidence=[extract(*ref) for ref in refs]
        assert action in {'answer','clarify','refuse','conflict'}
        assert facts and (evidence or action in {'clarify','refuse'})
        questions.append(dict(id=identifier,group_id=group,dialogue_id=group if len(scenario['turns'])>1 else None,
                              turn=k,question=question,category=scenario['category'],expected_status=action,
                              acceptable_statuses=[action],required_facts=facts,expected_answer='; '.join(facts),
                              evidence=evidence,evaluation_as_of='2026-11-21',split='portfolio_final'))
        nearby=sorted(((SequenceMatcher(None,normalize(question),normalize(q['question'])).ratio(),q['id'],q['question'])
                       for q in past),reverse=True)[:3]
        assert nearby[0][0]<1,(identifier,nearby[0])
        duplicates.append(dict(id=identifier,nearest=nearby))
assert 30<=len(specs)<=40
config_hash=hashlib.sha256((ROOT/'config/experiment.env').read_bytes()).hexdigest()
protocol=dict(source_commit=args.source_commit,hardware={'cpu':'i7-3770','ram_gib':16,'gpu':'RX570 4 GiB'},
              profile_sha256=config_hash,scenarios=len(specs),questions=questions,
              duplicate_review=duplicates,
              criteria={'paired_modes':['rag','agent'],'order':'alternate by turn','date':'2026-11-21',
                        'recall_at_8':'mean literal evidence recall over applicable questions',
                        'full_context':'all reference evidence in assessment context',
                        'success':'all required facts supported or expected action correctly performed',
                        'recovery':'new relevant evidence plus answer covering the previously missing part',
                        'judge':'preliminary local grades; human evaluation separate'})
args.output.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(len(specs),len(questions),hashlib.sha256(args.output.read_bytes()).hexdigest())
