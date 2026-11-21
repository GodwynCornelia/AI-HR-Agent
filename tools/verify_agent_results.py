"""Пересчёт публичной dev-сводки без БД, модели и закрытых диагностик."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from assistant.assessment import percentile
from assistant.evaluation import retrieval_metrics,chunk_relevant
BASE=ROOT/'experiments/agent_dev'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-8),(a,b)
summary=read(BASE/'summary.json');catalog=read(BASE/'chunk_catalog.json')
manifest=read(ROOT/'data/rgr_20261118/corpus_manifest.json')
hashes={d['file']:d['sha256'] for d in manifest}
for d in manifest:assert hashlib.sha256((ROOT/'data/rgr_20261118/corpus'/d['file']).read_bytes()).hexdigest()==d['sha256']
sources={k:{**v,'document_id':v['document']} for k,v in catalog.items()}
for k,v in catalog.items():assert hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]==k
questions={q['id']:q for q in read(ROOT/'data/benchmark/questions.json')}
for run in summary['runs']:
 records=read(BASE/('records_'+run['run']+'.json'))
 assert len(records)==run['records']==2*len(run['configuration']['ids'])
 assert len({(r['question_id'],r['mode']) for r in records})==len(records)
 for r in records:
  assert questions[r['question_id']]['split']=='dev'
  assert r['source_commit']==run['configuration']['source_commit']
  if r['mode']=='agent':assert r['searches']<=3 and r['actions']<=8
  refs=[{**e,'sha256':hashes[e['document']]} for e in questions[r['question_id']].get('evidence',[])]
  relevant=sum(any(chunk_relevant(s,e,hashes) for e in refs) for s in sources.values())
  for step in r['search_steps']:
   ss=[sources[k] for k in step['chunks']]
   metrics=retrieval_metrics(ss,refs,hashes,len(ss),relevant)
   for key in ['recall_at_k','span_recall_at_k','ndcg_at_k']:close(metrics[key],step['coverage'][key])
 for mode,s in run['modes'].items():
  rows=[r for r in records if r['mode']==mode];assert len(rows)==s['n']
  for key,p in [('p50',.5),('p95',.95)]:close(percentile([r['seconds'] for r in rows],p),s[key])
  for key,field in [('mean_searches','searches'),('mean_llm_calls','llm_calls')]:close(sum(r[field] for r in rows)/len(rows),s[key])
  for kind in ['provided','cited']:
   values=[r[kind+'_metrics']['span_recall_at_k'] for r in rows if r[kind+'_metrics']['span_recall_at_k'] is not None]
   assert len(values)==s[kind+'_coverage']['n'];close(sum(values)/len(values) if values else None,s[kind+'_coverage']['mean'])
  for key,field in [('known_prompt_tokens','known_prompt_tokens'),('known_output_tokens','known_output_tokens'),('action_label_matches','action_label_proxy'),('strict_span_action_proxy','strict_span_action_proxy'),('blocked_duplicates','blocked_duplicate_calls'),('partial_answers','partial')]:assert sum(r[field] for r in rows)==s[key]
  assert sum(r['searches']>1 for r in rows)==s['extra_search_requests']
  assert sum((r['additional_search_reference_gain'] or 0)>0 for r in rows)==s['positive_reference_gain_requests']
  groups={r['group_id'] for r in rows};assert len(groups)==s['author_groups']
  assert sum(all(r['strict_span_action_proxy'] for r in rows if r['group_id']==g) for g in groups)==s['whole_group_strict_proxy']
 print(run['run'],len(records),'records: OK')
judges=read(BASE/'preliminary_llm.json')
assert not judges['human_review']
for run in judges['runs']:
 for mode,s in run['summary'].items():
  rows=[r for r in run['grades'] if r['mode']==mode];assert len(rows)==s['n']
  for key in ['coverage','critical_error','action_correct']:close(sum(r[key] for r in rows)/len(rows),s[key])
print('41 corpus hashes, 286 catalog chunks, retrieval steps, summary tables and 34 preliminary LLM grades: OK')
