"""Ограниченный живой replay двух структурированных решений на сохранённом dev-контексте."""
import argparse,json,os,sys,time,uuid,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.environ.setdefault('DJANGO_SETTINGS_MODULE','hr_project.settings')
import django;django.setup()
from assistant.inference import Ollama
from assistant.services import REFERENCE_REPAIR_PROMPT
from assistant.contracts import StandaloneQuery
from assistant.agent_graph import PLAN_PROMPT,Decision
parser=argparse.ArgumentParser();parser.add_argument('answers',type=Path);parser.add_argument('--tokenizer',type=Path,required=True);parser.add_argument('--models',nargs='+',default=['qwen3:4b','qwen3:1.7b']);args=parser.parse_args()
record=next(r for r in (json.loads(l) for l in args.answers.read_text(encoding='utf-8').splitlines()) if r['question_id']=='v4.dialogue.cowork_to_hotel.3' and r['mode']=='rag')
history=[{'question':h['question'],'search_query':h.get('search_query','')} for h in record['history']]
repair={'current':record['question'],'previous_question':next((h['search_query'] for h in reversed(history) if h['search_query']),history[-1]['question']),'proposed_query':''}
planning={'current':record['question'],'resolved_question':'','topic':'','conditions_from_user':[h['question'] for h in history]+[record['question']],
 'suggested_clarification':record['answer'],'queries':[],'results':[],'assessment':{},'remaining_searches':3,'remaining_actions':7}
schema=Decision.model_json_schema();schema['properties']['action']['enum']=['search_policies','clarify','refuse']
out=ROOT/'runtime/model-probes'/str(uuid.uuid4());out.mkdir(parents=True)
results=[]
for name in args.models:
 os.environ['TOKENIZER_PATH']=str(args.tokenizer);model=Ollama();model.model=name
 for component,prompt,payload,contract in [('reference_repair',REFERENCE_REPAIR_PROMPT,repair,StandaloneQuery.model_json_schema()),('action_choice',PLAN_PROMPT,planning,schema)]:
  start=time.perf_counter()
  try:value=model.chat(prompt,[{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],structured=contract);error=None
  except Exception as exc:value=None;error=str(exc)
  result={'model':name,'component':component,'response':value,'error':error,'seconds':time.perf_counter()-start,
   'request':getattr(model,'last_request',{}),'metadata':getattr(model,'last_chat_metadata',{}),
   'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),'scope':'component_replay_no_tool_execution_not_end_to_end'}
  results.append(result);(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
  print(name+' '+component+': '+json.dumps(value,ensure_ascii=False),flush=True)
print(out,flush=True)
