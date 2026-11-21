"""Обязательные ограничения применяются к окончательному выбранному ответу."""
from .answer_checks import numeric_requirement_check
from .answer_support import requested_value_check
from .applicability import safe_clarification

def enforce(question,result):
    audit={}
    if result['status']=='clarify' and not safe_clarification(result['answer']):
        result={**result,'answer':'Какое существенное условие или вариант вы хотите уточнить?'}
        audit['unsafe_or_nonquestion_clarification_replaced']=True
    sources=result.get('sources',[])
    if result['status'] in {'answer','conflict'} and (not sources or any(not s.get('quote') or s['quote'] not in s.get('text','') for s in sources)):
        result={**result,'status':'refuse','sources':[],'answer':'Не удалось подтвердить ответ точными доступными источниками.','validation_actions':['invalid_final_citations']}
    result,audit['numeric']=numeric_requirement_check(question,result)
    result,audit['requested_value']=requested_value_check(question,result)
    return result,audit
