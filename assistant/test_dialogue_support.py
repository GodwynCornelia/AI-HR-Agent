import json
from unittest.mock import patch
from django.test import SimpleTestCase, TestCase
from django.contrib.auth import get_user_model
from .models import Conversation, Turn
from .services import ask
from .dialogue import baseline_messages, unspecified_alternative, explicit_topic_reset, inferential_followup
from .answer_support import AnswerSupport, apply_answer_support, requested_value_check
from .evidence import excerpts_from_sources, generate_grounded_answer


class DialogueSupportTests(SimpleTestCase):
    def test_payment_party_condition_requires_more_than_general_expense_instruction(self):
        question='Услуги оплачивает третья сторона. Что записать об оплате?'
        original={'status':'answer','answer':'Проект','sources':[{'quote':'Документ об оплате сохраняют в карточке.'}]}
        self.assertEqual(requested_value_check(question,original)[0]['status'],'refuse')
        original['sources']=[{'quote':'При оплате третьей стороной указывают оплаченные услуги.'}]
        self.assertEqual(requested_value_check(question,original)[0]['status'],'answer')
        self.assertEqual(requested_value_check('Что записать об оплате?',original)[0]['status'],'answer')
    def test_explicit_topic_reset_does_not_treat_condition_change_as_new_topic(self):
        self.assertTrue(explicit_topic_reset('Теперь другая тема: как получить справку?'))
        self.assertTrue(explicit_topic_reset('Теперь о питании: как исправить чек?'))
        self.assertFalse(explicit_topic_reset('А для офисных сотрудников?'))
        self.assertFalse(explicit_topic_reset('Теперь о втором варианте расскажите подробнее.'))

    def test_inferential_reference_restores_topic_but_not_an_unknown_alternative(self):
        previous='Как отразить неудачную пробу приспособления?'
        query=inferential_followup('Это означает отказ от дальнейшей помощи?',previous)
        self.assertIn(previous,query)
        self.assertIsNone(inferential_followup('А так же можно?',previous))
        self.assertIsNone(inferential_followup('Это означает отказ?', ''))

    def test_requested_phone_requires_an_actual_number_not_related_instructions(self):
        question='Назовите конкретный телефон поддержки.'
        candidate={'status':'answer','sources':[{'quote':'Поддержка отвечает через портал.'}],'answer':'Соседняя инструкция.'}
        result,audit=requested_value_check(question,candidate)
        self.assertEqual(result['status'],'refuse')
        self.assertFalse(audit['value_present_in_selected_quotes'])
        candidate['sources']=[{'quote':'Телефон поддержки: +7 (800) 555-01-01.'}]
        self.assertEqual(requested_value_check(question,candidate)[0]['status'],'answer')
        self.assertFalse(requested_value_check('Как поменять телефон в карточке?',candidate)[1]['applicable'])

    def test_company_role_is_not_company_name_but_explicit_name_is_accepted(self):
        question='Как называется курьерская компания?'
        candidate={'status':'answer','sources':[{'quote':'Обращайтесь к курьеру IT-службы.'}],'answer':'Курьер IT-службы.'}
        self.assertEqual(requested_value_check(question,candidate)[0]['status'],'refuse')
        candidate['sources']=[{'quote':'Доставку выполняет компания «Пример доставки».'}]
        self.assertEqual(requested_value_check(question,candidate)[0]['status'],'answer')
    def test_bare_alternative_is_different_from_explicit_condition(self):
        for message in ['А так же можно?', 'А иначе?', 'А можно по-другому?']:
            self.assertTrue(unspecified_alternative(message))
        for message in ['А для офисных сотрудников?', 'Это означает отказ от адаптации?', 'А можно подать заявку завтра?']:
            self.assertFalse(unspecified_alternative(message))

    def test_baseline_keeps_own_history_but_identifies_current_question(self):
        history=[{'role':'user','content':'Тема обучения'}, {'role':'assistant','content':'Старый ответ'}]
        current='Теперь о питании: как исправить ошибку в чеке?'
        packet=baseline_messages(history,current)
        self.assertEqual(len(packet),1)
        self.assertEqual(json.loads(packet[0]['content']),{'history':history,'current':current})
        self.assertEqual(history[-1]['content'],'Старый ответ')

    def test_missing_detail_does_not_fall_back_to_neighbouring_quotes(self):
        original={'answer':'Подайте заявку в IT.','sources':[{'quote':'Подайте заявку в IT.'}]}
        result=apply_answer_support(original,AnswerSupport(status='refuse',proof_ids=[],clarification='').model_dump(),{})
        self.assertEqual(result['status'],'refuse')
        self.assertEqual(result['sources'],[])
        self.assertNotIn('Подайте заявку в IT.',result['answer'])
        self.assertEqual(result['model_answer_before_guard'],original['answer'])

    def test_support_recovers_sufficient_proof_even_after_primary_refusal(self):
        source={'id':'s','document_id':'d','title':'Правило','location':'раздел','text':'Для пробы оформляют акт выдачи.'}
        _,proofs=excerpts_from_sources([source])
        result=apply_answer_support({'answer':'Нет ответа'}, {'status':'answer','proof_ids':['E1'],'clarification':''}, proofs)
        self.assertEqual(result['status'],'answer')
        self.assertEqual(result['sources'][0]['quote'],source['text'])
        with self.assertRaises(KeyError):
            apply_answer_support({'answer':'Проект'}, {'status':'answer','proof_ids':['E99'],'clarification':''},proofs)

    def test_true_explicit_limit_remains_answer_with_positive_proof(self):
        source={'id':'s','document_id':'d','title':'Регламент','location':'раздел','text':'Выдача устройства за один день не гарантируется.'}
        _,proofs=excerpts_from_sources([source])
        result=apply_answer_support({'answer':'Проект'},{'status':'answer','proof_ids':['E1'],'clarification':''},proofs)
        self.assertEqual(result['status'],'answer')
        self.assertIn('не гарантируется',result['answer'])

    def test_evidence_selected_before_writer_without_discarded_primary_generation(self):
        class Stub:
            calls=0;last_request={};last_raw_response='';last_chat_metadata={}
            def fits(self,*args):return True
            def chat(self,system,messages,structured):
                self.calls+=1
                if self.calls==1:
                    self.packet=json.loads(messages[0]['content'])
                    return {'finding':'Прямое правило о питании. '*20,'proof_ids':['E2'],'sufficient':True,'missing':[],'clarification':''}
                return {'status':'answer','claims':[{'text':'Для питания прикладывают чек.','proof_ids':['E2']}],'message':'','conflict_reason':''}
        source={'id':'s','document_id':'d','title':'Регламент','location':'раздел',
                'text':'Для обучения создают заявку. Для питания прикладывают чек.'}
        model=Stub()
        with patch('assistant.evidence.check_policy_relations',return_value=None):
            result,audit=generate_grounded_answer(model,'Что приложить для питания?',[source],'2026-11-20')
        self.assertIn('Для питания прикладывают чек.',result['answer'])
        self.assertNotIn('Для обучения',result['answer'])
        self.assertEqual(len(model.packet['documents'][0]['excerpts']),2)
        self.assertEqual(model.calls,2)


class TopicContextTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user('topic-test')
        self.conversation=Conversation.objects.create(owner=self.user)
        Turn.objects.create(conversation=self.conversation,original='Как оформить учебный отпуск?',
                            rewritten='Как оформить учебный отпуск?',answer='Старый ответ не доказательство.',status='answer')

    def test_explicit_new_topic_uses_empty_context_even_if_model_overclarifies(self):
        question='Теперь другая тема: как получить кадровую справку?'
        with patch('assistant.services.heavy_operation'),patch('assistant.services.Ollama') as model,patch('assistant.services.retrieve',return_value=[]) as search:
            model.return_value.last_request={};model.return_value.last_chat_metadata={}
            model.return_value.last_embed_request={};model.return_value.last_embed_metadata={};model.return_value.last_retrieval_diagnostics={}
            model.return_value.chat.return_value={'status':'clarify','query':'','clarification':'Что значит справка?'}
            result=ask(self.user,self.conversation.id,question)
        packet=json.loads(model.return_value.chat.call_args.args[1][0]['content'])
        self.assertEqual(packet['history'],[])
        self.assertEqual(search.call_args.args[0],question)
        self.assertTrue(result.diagnostics['topic_reset'])
        self.assertEqual(len(result.diagnostics['history_before_topic_reset']),1)

    def test_rewrite_has_user_topics_but_no_prior_model_facts(self):
        with patch('assistant.services.heavy_operation'),patch('assistant.services.Ollama') as model:
            model.return_value.last_request={};model.return_value.last_chat_metadata={}
            model.return_value.chat.return_value={'status':'clarify','query':'','clarification':'Какое условие меняется?'}
            ask(self.user,self.conversation.id,'А иначе?')
        packet=json.loads(model.return_value.chat.call_args.args[1][0]['content'])
        self.assertEqual(packet['history'][0]['question'],'Как оформить учебный отпуск?')
        self.assertNotIn('Старый ответ не доказательство.',json.dumps(packet,ensure_ascii=False))
