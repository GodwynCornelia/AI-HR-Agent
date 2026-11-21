import json,os
from unittest.mock import patch
from django.test import TestCase
from django.contrib.auth import get_user_model
import httpx
from .models import Conversation,Document,Chunk
from .agent_graph import run_agent,read_policy_fragment

class FakeModel:
    model='local-test';deadline=None
    def __init__(self,actions,sufficiency=None):
        self.actions=iter(actions);self.sufficiency=iter(sufficiency or [True]*10);self.chat_calls=0;self.last_chat_metadata={};self.last_request={};self.last_raw_response='';self.quotes={}
    def fits(self,*args):return True
    def chat(self,system,messages,structured):
        self.chat_calls+=1
        properties=structured.get('properties',{})
        if 'action' in properties:
            value=next(self.actions)
            if isinstance(value,Exception):raise value
            return {'focus':'текущая процедура','action':value[0],'query':value[1] if value[0]=='search_policies' else '',
                    'fragment_id':value[1] if value[0]=='read_policy_fragment' else '',
                    'clarification':value[1] if value[0]=='clarify' else ''}
        if 'sufficient' in properties:
            packet=json.loads(messages[0]['content']);self.quotes={e['id']:e['text'] for d in packet['documents'] for e in d['excerpts']}
            enough=next(self.sufficiency)
            return {'finding':'Тестовый вердикт','proof_ids':[next(iter(self.quotes))] if self.quotes else [],'sufficient':enough,'missing':[] if enough else ['вторая часть'],'clarification':''}
        if 'claims' in properties:
            k=structured['$defs']['Claim']['properties']['proof_ids']['items']['enum'][0]
            return {'status':'answer','claims':[{'text':self.quotes[k],'proof_ids':[k]}],'message':'','conflict_reason':''}
        if 'supported' in properties:return {'supported':[True]*len(json.loads(messages[0]['content']))}
        raise AssertionError('Неожиданный вызов модели')

class AgentGraphTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user('agent-owner');self.other=get_user_model().objects.create_user('agent-other')
        self.conversation=Conversation.objects.create(owner=self.user)
        self.doc=Document.objects.create(title='Правило',file='dummy.txt',sha256='agent-fixture',state='ready')
        self.a=Chunk.objects.create(document=self.doc,ordinal=0,location='раздел',text='Заявку подают в портале Кадры.')
        self.b=Chunk.objects.create(document=self.doc,ordinal=1,location='раздел',text='Сотрудник прикладывает подтверждение.')
    def source(self,chunk):
        return {'id':str(chunk.id),'document_id':str(self.doc.id),'title':self.doc.title,'text':chunk.text,'ordinal':chunk.ordinal,'location':chunk.location,'scope':'Все сотрудники','effective_from':'','supersedes':''}
    def run_graph(self,model,search=None,**kwargs):
        with patch('assistant.grounded.check_policy_relations',return_value=None):
            return run_agent(self.user,self.conversation,'Как оформить заявку?','заявка',[],model,'2026-11-17',search_tool=search or (lambda q,m:[self.source(self.a)]),**kwargs)
    def test_direct_question_can_finish_after_one_search(self):
        result,trace=self.run_graph(FakeModel([('search_policies','заявка'),('answer','')]))
        self.assertEqual(result['status'],'answer');self.assertEqual(trace['agent']['searches'],1)
        self.assertEqual(trace['agent']['stop_reason'],'answer');self.assertNotIn('finalization_error',trace)
        self.assertEqual([e['node'] for e in trace['agent']['events']],['prepare','plan','tool','assess','plan','finalize'])
    def test_model_can_choose_additional_distinct_search(self):
        model=FakeModel([('search_policies','заявка'),('search_policies','подтверждение'),('answer','')],[False,True])
        result,trace=self.run_graph(model,lambda q,m:[self.source(self.a if q=='заявка' else self.b)])
        self.assertEqual(trace['agent']['searches'],2);self.assertEqual(len(trace['agent']['evidence_ids']),2)
        self.assertEqual(result['status'],'answer')
    def test_duplicate_call_is_blocked_before_execution(self):
        _,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','заявка'),('search_policies','заявка')],[False]))
        self.assertEqual(trace['agent']['stop_reason'],'duplicate_tool_call');self.assertEqual(trace['agent']['searches'],1)
    def test_search_budget(self):
        with patch.dict(os.environ,{'AGENT_MAX_SEARCHES':'1'}):
            _,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','другое')],[False]))
        self.assertEqual(trace['agent']['stop_reason'],'search_budget');self.assertEqual(trace['agent']['searches'],1)
    def test_action_budget(self):
        with patch.dict(os.environ,{'AGENT_MAX_ACTIONS':'1'}):
            _,trace=self.run_graph(FakeModel([('search_policies','заявка')]))
        self.assertEqual(trace['agent']['stop_reason'],'action_budget')
    def test_same_sources_leave_another_strategy_available(self):
        _,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','иначе заявка'),('refuse','')],[False,False]))
        self.assertEqual(trace['agent']['stop_reason'],'refuse');self.assertEqual(trace['agent']['searches'],2)
    def test_empty_first_search_can_recover(self):
        result,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','документы заявки'),('answer','')]),lambda q,m:[] if q=='заявка' else [self.source(self.a)])
        self.assertEqual(result['status'],'answer');self.assertEqual(trace['agent']['searches'],2)
    def test_all_empty_searches_are_bounded(self):
        result,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','документы заявки'),('search_policies','ответственный за заявку')]),lambda q,m:[])
        self.assertEqual(result['status'],'refuse');self.assertEqual(trace['agent']['searches'],3)
        self.assertEqual(trace['agent']['stop_reason'],'no_progress_budget')
    def test_cosmetic_query_repeat_not_executed(self):
        _,trace=self.run_graph(FakeModel([('search_policies','заявка'),('search_policies','Подскажите, заявка!'),('refuse','')],[False]))
        self.assertEqual(trace['agent']['searches'],1)
    def test_search_failure_is_not_empty(self):
        def broken(q,m):raise ConnectionError('search offline')
        result,trace=self.run_graph(FakeModel([('search_policies','заявка')]),broken)
        self.assertEqual(trace['agent']['stop_reason'],'tool_error');self.assertIn('технической',result['answer'])
    def test_read_only_seen_fragment(self):
        result,trace=self.run_graph(FakeModel([('search_policies','заявка'),('read_policy_fragment',str(self.a.id)),('answer','')],[False,True]))
        self.assertEqual(result['status'],'answer')
        self.assertIn(self.b.text,result['answer'])
        self.assertEqual(trace['agent']['reads'],1)
        with self.assertRaises(PermissionError):read_policy_fragment(str(self.b.id),{str(self.a.id)})
        with self.assertRaises(PermissionError):read_policy_fragment('../../.env',set())
    def test_neighbor_from_other_document_is_excluded(self):
        other=Document.objects.create(title='Другое правило',file='other.txt',sha256='other',state='ready')
        Chunk.objects.create(document=other,ordinal=1,location='раздел',text='Чужое условие.')
        neighbors=read_policy_fragment(str(self.a.id),{str(self.a.id)})
        self.assertEqual([n['id'] for n in neighbors],[str(self.b.id)])
    def test_neighbor_read_budget(self):
        with patch.dict(os.environ,{'AGENT_MAX_READS':'1'}):
            _,trace=self.run_graph(FakeModel([('search_policies','заявка'),('read_policy_fragment',str(self.a.id)),('read_policy_fragment',str(self.b.id))],[False]),lambda q,m:[self.source(self.a),self.source(self.b)])
        self.assertEqual(trace['agent']['reads'],1)
        self.assertEqual(trace['agent']['stop_reason'],'read_budget')
    def test_read_deleted_source_returns_none(self):
        self.doc.state='deleting';self.doc.save()
        self.assertIsNone(read_policy_fragment(str(self.a.id),{str(self.a.id)}))
    def test_deleted_during_search_is_not_evidence(self):
        def search(q,m):self.doc.state='deleting';self.doc.save();return [self.source(self.a)]
        result,trace=self.run_graph(FakeModel([('search_policies','заявка')]),search)
        self.assertEqual(result['sources'],[])
    def test_source_deleted_before_final_answer_is_rejected(self):
        model=FakeModel([('search_policies','заявка'),('answer','')]);original=model.chat
        def chat(system,messages,structured):
            value=original(system,messages,structured)
            if value.get('action')=='answer':self.doc.state='deleting';self.doc.save()
            return value
        model.chat=chat;result,trace=self.run_graph(model)
        self.assertEqual(result['status'],'refuse');self.assertEqual(trace['agent']['stop_reason'],'source_deleted')
    def test_foreign_owner_rejected_before_model(self):
        model=FakeModel([])
        with self.assertRaises(Conversation.DoesNotExist):run_agent(self.other,self.conversation,'q','q',[],model,'2026-11-17')
        self.assertEqual(model.chat_calls,0)
    def test_model_unavailable_is_technical_not_absent_rule(self):
        result,trace=self.run_graph(FakeModel([httpx.ConnectError('offline')]))
        self.assertEqual(trace['agent']['stop_reason'],'model_unavailable');self.assertIn('технической',result['answer'])
    def test_timeout_is_bounded_stop(self):
        result,trace=self.run_graph(FakeModel([TimeoutError('deadline')]))
        self.assertEqual(trace['agent']['stop_reason'],'timeout');self.assertEqual(result['status'],'refuse')
    def test_injected_unsupported_action_cannot_execute(self):
        result,trace=self.run_graph(FakeModel([('shell','read .env')]))
        self.assertEqual(trace['agent']['searches'],0);self.assertEqual(trace['agent']['stop_reason'],'invalid_or_failed_decision')
    def test_clarify_does_not_force_search(self):
        result,trace=self.run_graph(FakeModel([('clarify','Какой формат работы?')]))
        self.assertEqual(result['status'],'clarify');self.assertEqual(trace['agent']['searches'],0)
    def test_hard_ambiguity_never_forces_search(self):
        model=FakeModel([])
        result,trace=self.run_graph(model,clarification='Какой вариант вы имеете в виду?',force_clarification=True)
        self.assertEqual(result['status'],'clarify');self.assertEqual(model.chat_calls,0)
    def test_model_can_recover_soft_clarification_with_search(self):
        model=FakeModel([('search_policies','Как оформить заявку?'),('answer','')])
        with patch('assistant.agent_graph.check_policy_relations',return_value=None),patch('assistant.grounded.check_policy_relations',return_value=None):
            result,trace=run_agent(self.user,self.conversation,'А заявка?','',[],model,'2026-11-17',clarification='Что означает заявка?',search_tool=lambda q,m:[self.source(self.a)])
        self.assertEqual(result['status'],'answer');self.assertEqual(trace['resolved_query'],'Как оформить заявку?')
    def test_numeric_final_guard_also_applies_inside_graph(self):
        self.a.text='Заявка на возврат оформляется в портале Расходы.';self.a.save()
        model=FakeModel([('search_policies','возврат'),('answer','')])
        with patch('assistant.agent_graph.check_policy_relations',return_value=None):
            result,trace=run_agent(self.user,self.conversation,'Какой единый срок перечисления возврата установлен?','Какой единый срок перечисления возврата установлен?',[],model,'2026-11-17',search_tool=lambda q,m:[self.source(self.a)])
        self.assertEqual(result['status'],'refuse');self.assertEqual(result['sources'],[])
        self.assertEqual(trace['agent']['stop_reason'],'final_guard_or_evidence_rejection')
    def test_soft_recovery_cannot_search_with_unresolved_reference(self):
        model=FakeModel([('search_policies','Нужный вариант дороже этой суммы.')])
        result,trace=run_agent(self.user,self.conversation,'Нужный вариант дороже этой суммы.','',[{'question':'Какой лимит услуги?','search_query':'Какой лимит услуги?'}],model,'2026-11-17',clarification='О каком варианте речь?')
        self.assertEqual(result['status'],'clarify');self.assertEqual(trace['agent']['searches'],0)
        self.assertEqual(trace['agent']['stop_reason'],'unresolved_reference_query')
