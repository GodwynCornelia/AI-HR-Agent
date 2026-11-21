"""Последовательная dev-диагностика двух режимов, без передачи эталонов приложению."""
import argparse,hashlib,json,os,sys,time,uuid,subprocess,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.environ.setdefault('DJANGO_SETTINGS_MODULE','hr_project.settings')
import django;django.setup()
import httpx
from django.contrib.auth import get_user_model
from assistant.models import Conversation,Document,Chunk
from assistant.services import ask
from assistant.assessment import ResourceSampler
from assistant.evaluation import retrieval_metrics,chunk_relevant
from assistant.inference import Ollama
parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--ids',nargs='*');args=parser.parse_args()
protocol=json.loads((ROOT/'experiments/agent_dev/protocol.json').read_text(encoding='utf-8'))
questions=json.loads((ROOT/'data/benchmark/questions.json').read_text(encoding='utf-8'))
ids=set(args.ids or (protocol['question_ids'][:1] if args.smoke else protocol['question_ids']))
selected=[q for q in questions if q['id'] in ids];assert len(selected)==len(ids) and all(q['split']=='dev' for q in selected)
dialogues={q.get('dialogue_id') for q in selected if q.get('dialogue_id')}
selected=[q for q in questions if q['id'] in ids or q.get('dialogue_id') in dialogues]
assert all(q['split']=='dev' for q in selected)
corpus=json.loads((ROOT/'data/rgr_20261118/corpus_manifest.json').read_text(encoding='utf-8'));hashes={d['file']:d['sha256'] for d in corpus}
documents={str(d.id):d.sha256 for d in Document.objects.filter(state='ready')}
assert len(documents)==len(corpus) and set(documents.values())==set(hashes.values())
user,_=get_user_model().objects.get_or_create(username='local_dev_comparison',defaults={'is_active':True})
directory=ROOT/'runtime/agent-dev'/str(uuid.uuid4());directory.mkdir(parents=True)
commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
config={'purpose':protocol['purpose'],'source_commit':commit,'langgraph':importlib.metadata.version('langgraph'),
 'ids':[q['id'] for q in selected],'same_retriever':True,'new_final_test':False,'model':os.getenv('LLM_MODEL'),'embedding':os.getenv('EMBED_MODEL'),
 'model_context':os.getenv('MODEL_CONTEXT'),'model_output':os.getenv('MODEL_OUTPUT'),'top_k':os.getenv('TOP_K'),
 'corpus_manifest_sha256':hashlib.sha256((ROOT/'data/rgr_20261118/corpus_manifest.json').read_bytes()).hexdigest(),
 'protocol_sha256':hashlib.sha256((ROOT/'experiments/agent_dev/protocol.json').read_bytes()).hexdigest()}
server=Ollama();tags=httpx.get(server.base+'/api/tags',timeout=15).json()
config.update(ollama_version=httpx.get(server.base+'/api/version',timeout=15).json(),model_digests={m['name']:m['digest'] for m in tags.get('models',[]) if m['name'] in {server.model,server.embed_model,server.embed_model+':latest'}},requirements_lock_sha256=hashlib.sha256((ROOT/'requirements.lock.txt').read_bytes()).hexdigest(),initial_query_shared_with_rag=True)
(directory/'manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
warm=Ollama();warm.chat('Ответь одним словом: готов.',[{'role':'user','content':'Начало локальной технической проверки.'}],structured=False)
(directory/'warmup.json').write_text(json.dumps(warm.last_chat_metadata,ensure_ascii=False),encoding='utf-8')
conversations={};results=[]
with (directory/'answers.jsonl').open('w',encoding='utf-8') as f:
 for index,q in enumerate(selected):
  for mode in (['rag','agent'] if index%2==0 else ['agent','rag']):
   key=(mode,q.get('dialogue_id') or q['id'])
   if key not in conversations:conversations[key]=Conversation.objects.create(owner=user,mode=mode,title='Dev comparison')
   sampler=ResourceSampler();sampler.start();start=time.perf_counter()
   turn=ask(user,conversations[key].id,q['question'],as_of=q['evaluation_as_of'])
   row={'run_id':directory.name,'question_id':q['id'],'group_id':q['group_id'],'mode':mode,'question':q['question'],
        'status':turn.status,'answer':turn.answer,'rewritten':turn.rewritten,'history':turn.diagnostics.get('history',[]),
        'result':turn.diagnostics.get('result',{}),'diagnostics':turn.diagnostics,'seconds':time.perf_counter()-start,
        'phase':'dev_diagnostic','independent_evaluation':False,**sampler.stop()}
   refs=[{**ref,'sha256':hashes[ref['document']]} for ref in q.get('evidence',[])]
   candidates=Chunk.objects.select_related('document').filter(document__state='ready',document__sha256__in=[r['sha256'] for r in refs])
   relevant=sum(any(chunk_relevant({'document_id':str(c.document_id),'text':c.text},ref,documents) for ref in refs) for c in candidates)
   for name,sources in [('retrieved',turn.diagnostics.get('retrieved',[])),('provided',turn.diagnostics.get('provided_sources',[])),('cited',[{**s,'text':s['quote']} for s in row['result'].get('sources',[])])]:
    row[name+'_metrics']=retrieval_metrics(sources,refs,documents,len(sources),relevant)
   row['action_label_proxy']=turn.status in q.get('acceptable_statuses',[q['expected_status']])
   row['is_semantic_accuracy']=False
   results.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   print(q['id']+' / '+mode+': '+turn.status+f" ({row['seconds']:.1f}s)",flush=True)
(directory/'completion.json').write_text(json.dumps({'records':len(results),'expected':2*len(selected),'errors':sum(r['status']=='error' for r in results)}),encoding='utf-8')
print(directory,flush=True)
