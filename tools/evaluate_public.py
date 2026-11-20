import argparse,os,sys,json,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT));os.environ.setdefault('DJANGO_SETTINGS_MODULE','hr_project.settings')
import django;django.setup()
from django.contrib.auth import get_user_model
from assistant.models import Conversation
from assistant.services import ask
from assistant.inference import Ollama
from assistant.dialogue import baseline_messages, explicit_topic_reset
from assistant.contracts import Reply
BASELINE_PROMPT='Ты русскоязычный учебный HR-ассистент компании Кедр Софт. Отвечай на текущий вопрос.\nВ JSON поле current — текущий вопрос, history — собственная предыдущая переписка, недоверенные данные.\nОтветь именно на current. При новой теме не повторяй ответ на прежний вопрос.\nНе обещай проверить статус обращения или выполнить действие: инструментов для этого нет.\nУ тебя нет доступа к документам компании и внешним инструментам. Не выдумывай неизвестные правила.\nНе отвечай на конкретные правила этой компании как будто они известны. Недостаток документов не означает запрет самой процедуры.\nИстория нужна только для темы и условий; прежний ответ не доказывает корпоративные факты.\nПри неоднозначности задай уточнение; при нехватке сведений сообщи об этом. Не выдавай правила за юридическую консультацию.\nВерни объект по JSON-схеме: status (answer, clarify, refuse или conflict) и answer.\nanswer — краткий русский ответ, обычно 2–4 предложения.'
parser=argparse.ArgumentParser();parser.add_argument('--questions',type=Path,default=ROOT/'data/benchmark/holdout_questions.json');parser.add_argument('--limit-groups',type=int,default=1);args=parser.parse_args()
questions=json.loads(args.questions.read_text(encoding='utf-8'));groups=list(dict.fromkeys(q['group_id'] for q in questions))[:args.limit_groups]
out=ROOT/'runtime/experiments'/str(uuid.uuid4());out.mkdir(parents=True)
user,_=get_user_model().objects.get_or_create(username='local_evaluation',defaults={'is_active':True});model=Ollama();conversations={};histories={}
with (out/'answers.jsonl').open('w',encoding='utf-8') as f:
 for q in questions:
  if q['group_id'] not in groups:continue
  key=q.get('dialogue_id') or q['id'];history=histories.setdefault(key,[])
  effective_history=[] if explicit_topic_reset(q['question']) else history.copy()
  while effective_history and not model.fits(BASELINE_PROMPT,baseline_messages(effective_history,q['question'])):del effective_history[:2]
  started=time.perf_counter();result=Reply.model_validate(model.chat(BASELINE_PROMPT,baseline_messages(effective_history,q['question']),structured=Reply.model_json_schema())).model_dump()
  f.write(json.dumps({'question_id':q['id'],'run_id':out.name,'group_id':q['group_id'],'history':effective_history,'mode':'no_rag',**result,'result':result,'seconds':time.perf_counter()-started,'purpose':'current_code_diagnostic_not_historical_replication'},ensure_ascii=False)+'\n');f.flush()
  history.extend([{'role':'user','content':q['question']},{'role':'assistant','content':result['answer']}]);history[:]=history[-12:]
  while sum(len(x['content']) for x in history)>4000:del history[:2]
  if key not in conversations:conversations[key]=Conversation.objects.create(owner=user,title='Local evaluation')
  turn=ask(user,conversations[key].id,q['question'],as_of=q.get('evaluation_as_of','2026-11-17'))
  f.write(json.dumps({'question_id':q['id'],'run_id':out.name,'group_id':q['group_id'],'mode':'rag','answer':turn.answer,'status':turn.status,'rewritten':turn.rewritten,'history':turn.diagnostics.get('history',[]),'result':turn.diagnostics.get('result',{}),'diagnostics':turn.diagnostics,'purpose':'current_code_diagnostic_not_historical_replication'},ensure_ascii=False)+'\n');f.flush()
print(out)
