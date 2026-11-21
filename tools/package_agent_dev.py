"""Компактная публикация dev-результатов без сырых промптов и рабочих диалогов."""
import argparse,json,hashlib,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.environ.setdefault('DJANGO_SETTINGS_MODULE','hr_project.settings')
import django;django.setup()
from assistant.models import Chunk,Document
from assistant.evaluation import retrieval_metrics,chunk_relevant
from assistant.assessment import percentile
parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path);parser.add_argument('--final',required=True);args=parser.parse_args()
output=ROOT/'experiments/agent_dev';output.mkdir(exist_ok=True)
questions={q['id']:q for q in json.loads((ROOT/'data/benchmark/questions.json').read_text(encoding='utf-8'))}
corpus=json.loads((ROOT/'data/rgr_20261118/corpus_manifest.json').read_text(encoding='utf-8'));hashes={d['file']:d['sha256'] for d in corpus}
docs={str(d.id):d.sha256 for d in Document.objects.filter(state='ready')};files={v:k for k,v in hashes.items()}
catalog={};by_id={}
for c in Chunk.objects.select_related('document').filter(document__state='ready',document__sha256__in=hashes.values()):
 item={'document':files[c.document.sha256],'location':c.location,'ordinal':c.ordinal,'text':c.text}
 key=hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]
 catalog[key]=item;by_id[str(c.id)]=(key,{'id':str(c.id),'document_id':str(c.document_id),'location':c.location,'ordinal':c.ordinal,'text':c.text})
summaries=[];published=[];trace_example=None;final_rows=[]
for directory in args.runs:
 if not directory.is_absolute():directory=ROOT/directory
 manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
 raw=[json.loads(l) for l in (directory/'answers.jsonl').read_text(encoding='utf-8').splitlines()];assert len(raw)==2*len(manifest['ids'])
 compact=[]
 for r in raw:
  q=questions[r['question_id']];agent=r['diagnostics'].get('agent',{});usage=r['diagnostics'].get('model_usage',{})
  events=agent.get('events',[]);searches=[e for e in events if e.get('tool')=='search_policies' and 'result_ids' in e]
  if not agent and r['diagnostics'].get('retrieved') is not None:searches=[{'result_ids':[s['id'] for s in r['diagnostics']['retrieved']]}]
  refs=[{**ref,'sha256':hashes[ref['document']]} for ref in q.get('evidence',[])]
  relevant=sum(any(chunk_relevant(source,ref,docs) for ref in refs) for _,source in by_id.values())
  steps=[]
  for e in searches:
   sources=[by_id[i][1] for i in e['result_ids'] if i in by_id]
   steps.append({'query':e.get('arguments',{}).get('query',r.get('rewritten')),'chunks':[by_id[i][0] for i in e['result_ids'] if i in by_id],
                 'coverage':retrieval_metrics(sources,refs,docs,len(sources),relevant)})
  first=steps[0]['coverage']['span_recall_at_k'] if steps else None
  union_ids=list(dict.fromkeys(i for e in searches for i in e['result_ids'] if i in by_id))
  union= retrieval_metrics([by_id[i][1] for i in union_ids],refs,docs,len(union_ids),relevant)['span_recall_at_k']
  extra_gain=union-first if first is not None and union is not None else None
  partial=r.get('result',{}).get('partial',r['answer'].startswith('Подтверждённая часть'))
  technical=agent.get('stop_reason') in {'timeout','model_unavailable','tool_error','assessment_unavailable','graph_execution_error','invalid_or_failed_decision'}
  row={'run':directory.name,'source_commit':manifest['source_commit'],'question_id':r['question_id'],'group_id':r['group_id'],'mode':r['mode'],
   'status':r['status'],'answer':r['answer'],'query':r.get('rewritten'),'seconds':r['seconds'],'partial':partial,
   'retrieved_metrics':r['retrieved_metrics'],'provided_metrics':r['provided_metrics'],'cited_metrics':r['cited_metrics'],
   'action_label_proxy':r['action_label_proxy'],'is_semantic_accuracy':False,'searches':agent.get('searches',len(searches)),
   'actions':agent.get('actions'),'llm_calls':usage.get('llm_calls'),
   'known_prompt_tokens':sum(e.get('prompt_eval_count',0) for e in usage.get('events',[])),
   'known_output_tokens':sum(e.get('eval_count',0) for e in usage.get('events',[])),
   'token_metadata_calls':len(usage.get('events',[])),'blocked_duplicate_calls':sum(e.get('blocked')=='duplicate' for e in events),
   'stop_reason':agent.get('stop_reason','linear_completion'),'fallback':r['diagnostics'].get('fallback'),
   'search_steps':steps,'additional_search_reference_gain':extra_gain,
   'necessary_success_proxy':bool(r['action_label_proxy'] and not partial and not technical and (r['cited_metrics']['all_required_evidence_covered'] is not False)),
   'sources':[{'title':s['title'],'location':s['location'],'quote':s['quote']} for s in r.get('result',{}).get('sources',[])]}
  compact.append(row)
  if directory.name==args.final:
   final_rows.append(row)
   if trace_example is None and r['mode']=='agent' and r['status']=='answer':
    trace_example={'scope':'synthetic_dev_example_not_working_user_dialogue','question':r['question'],'answer':r['answer'],
      'graph_version':agent.get('graph_version'),'source_commit':manifest['source_commit'],'configuration':agent.get('configuration'),
      'events':[{k:v for k,v in e.items() if k in {'node','seconds','action','tool','arguments','new_evidence','blocked','stop_reason'}} for e in events],
      'searches':agent.get('searches'),'llm_calls':usage.get('llm_calls'),'stop_reason':agent.get('stop_reason')}
 run_summary={'run':directory.name,'configuration':manifest,'records':len(compact),'modes':{}}
 for mode in ['rag','agent']:
  part=[r for r in compact if r['mode']==mode]
  def mean(key):values=[r[key] for r in part if r.get(key) is not None];return sum(values)/len(values) if values else None
  def coverage(kind):values=[r[kind]['span_recall_at_k'] for r in part if r[kind]['span_recall_at_k'] is not None];return {'n':len(values),'mean':sum(values)/len(values) if values else None}
  groups={g:[r for r in part if r['group_id']==g] for g in {r['group_id'] for r in part}}
  run_summary['modes'][mode]={'n':len(part),'p50':percentile([r['seconds'] for r in part],.5),'p95':percentile([r['seconds'] for r in part],.95),
    'provided_coverage':coverage('provided_metrics'),'cited_coverage':coverage('cited_metrics'),'mean_searches':mean('searches'),'mean_llm_calls':mean('llm_calls'),
    'known_prompt_tokens':sum(r['known_prompt_tokens'] for r in part),'known_output_tokens':sum(r['known_output_tokens'] for r in part),
    'action_label_matches':sum(r['action_label_proxy'] for r in part),'necessary_success_proxy':sum(r['necessary_success_proxy'] for r in part),
    'whole_group_proxy_success':sum(all(r['necessary_success_proxy'] for r in rows) for rows in groups.values()),'author_groups':len(groups),
    'extra_search_requests':sum(r['searches']>1 for r in part),'positive_reference_gain_requests':sum((r['additional_search_reference_gain'] or 0)>0 for r in part),
    'blocked_duplicates':sum(r['blocked_duplicate_calls'] for r in part),'partial_answers':sum(r['partial'] for r in part)}
 summaries.append(run_summary)
 (output/('records_'+directory.name+'.json')).write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
 published.extend(compact)
(output/'summary.json').write_text(json.dumps({'purpose':'targeted_dev_diagnostics_not_final_test','final_run':args.final,'runs':summaries,
 'limits':['Different source versions are separate runs','Proxies are not semantic correctness','Additional queries are useful only when reference coverage increases','Tokens cover returned metadata, not unknown failed requests']},ensure_ascii=False,indent=2),encoding='utf-8')
(output/'trace_example.json').write_text(json.dumps(trace_example,ensure_ascii=False,indent=2),encoding='utf-8')
(output/'chunk_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'runs':len(summaries),'final_records':len(final_rows),'catalog_chunks':len(catalog),'trace_example':trace_example is not None}))
