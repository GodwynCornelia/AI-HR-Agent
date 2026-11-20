from django.test import SimpleTestCase
from unittest.mock import patch
from .evidence import generate_grounded_answer

class FinalChecksTests(SimpleTestCase):
    def run_pipeline(self,text):
        class Model:
            calls=0;last_request={};last_raw_response='';last_chat_metadata={}
            def fits(self,*args):return True
            def chat(self,*args,**kwargs):
                self.calls+=1
                return ({'finding':'Выбран фрагмент','proof_ids':['E1'],'sufficient':True,'missing':[],'clarification':''}
                        if self.calls==1 else {'status':'answer','claims':[{'text':text,'proof_ids':['E1']}],'message':'','conflict_reason':''})
        source={'id':'s','document_id':'d','title':'Расходы','location':'раздел','text':text}
        with patch('assistant.evidence.check_policy_relations',return_value=None):
            return generate_grounded_answer(Model(),'Какой единый срок перечисления возврата установлен?',[source],'2026-11-17')
    def test_support_cannot_undo_numeric_rejection(self):
        result,trace=self.run_pipeline('Заявка на возврат оформляется в портале Расходы.')
        self.assertTrue(trace['assessment']['sufficient'])
        self.assertEqual(result['status'],'refuse')
        self.assertFalse(trace['final_mandatory_checks']['numeric']['explicit_value_or_boundary_in_selected_proof'])
    def test_actual_numeric_value_or_explicit_boundary_passes(self):
        for text in ['Возврат перечисляют в течение пяти рабочих дней.','Единый срок перечисления не установлен.']:
            self.assertEqual(self.run_pipeline(text)[0]['status'],'answer')
