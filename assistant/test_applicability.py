from unittest.mock import patch
from django.test import SimpleTestCase
from .applicability import compatible,safe_clarification
from .answer_checks import check_policy_relations
from .final_checks import enforce
from .dialogue import needs_reference_repair,unspecified_alternative

class ApplicabilityTests(SimpleTestCase):
    def relation(self,question,**fields):
        class Model:
            last_request={};last_chat_metadata={};last_raw_response=''
            def fits(self,*args):return True
            def chat(self,*args,**kwargs):return {'reason':'Явные условия сравниваются','same_service':True,'same_conditions':True,'question_matches_service':True,'relation':'unresolved_conflict','clarification':'',**fields}
        common={'scope':'Постоянные удалённые сотрудники','effective_from':'2026-01-01','supersedes':'','location':'раздел','title':'Программа'}
        proofs={'E1':{**common,'id':'a','document_id':'a','quote':'Расход возмещается.','text':'Расход возмещается.'},
                'E2':{**common,'id':'b','document_id':'b','quote':'Расход не возмещается.','text':'Расход не возмещается.'}}
        with patch('assistant.answer_checks.policy_pairs',return_value=[('E1','E2')]):return check_policy_relations(Model(),question,proofs,'2026-11-17',{})
    def test_unknown_applicability_is_clarified_before_conflict(self):
        self.assertEqual(self.relation('Компенсируют мне этот расход?')['status'],'clarify')
    def test_explicit_scope_still_reports_real_conflict(self):
        self.assertEqual(self.relation('Я постоянный удалённый сотрудник. Компенсируют расход?')['status'],'conflict')
    def test_other_service_or_conditions_do_not_become_conflict(self):
        self.assertIsNone(self.relation('Какой лимит другой услуги?',question_matches_service=False))
        self.assertIsNone(self.relation('Я постоянный удалённый. Какой лимит?',same_service=False))
        self.assertIsNone(self.relation('Я постоянный удалённый. Какой лимит?',same_conditions=False))
    def test_inapplicable_metadata_not_used_as_user_rule(self):
        self.assertFalse(compatible('Я офисный сотрудник. Как получить компенсацию?','Постоянные удалённые сотрудники'))
        self.assertTrue(compatible('Как получить компенсацию?','Постоянные удалённые сотрудники'))
    def test_policy_assertion_cannot_be_rendered_as_clarification(self):
        text='Постоянным удалённым ничего не предоставляется (D1, E2).'
        self.assertFalse(safe_clarification(text))
        result,audit=enforce('Как получить компенсацию?',{'status':'clarify','answer':text,'sources':[]})
        self.assertNotIn('не предоставляется',result['answer']);self.assertTrue(audit['unsafe_or_nonquestion_clarification_replaced'])
    def test_elliptical_repair_is_bounded_and_not_guessing_unnamed_variant(self):
        history=[{'question':'Как оформить услугу?'}]
        self.assertTrue(needs_reference_repair('А другая услуга?','А другая услуга?','search',history))
        self.assertTrue(needs_reference_repair('Дороже этой суммы','Дороже этой суммы','search',history))
        self.assertTrue(unspecified_alternative('А если другой вариант?'))
        self.assertFalse(needs_reference_repair('А если другой вариант?','','clarify',history))
        self.assertFalse(needs_reference_repair('Я имею в виду ремонт','Разрешён ли ремонт устройства внешним мастером?','search',history))
