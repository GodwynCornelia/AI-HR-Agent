"""Сводка фактических прогонов с отдельной ручной оценкой; не LLM-судья."""
import csv
import json
import math
import random
import threading
import psutil
from collections import defaultdict


class ResourceSampler:
    """Общесистемные наблюдения; не приписываются только модели."""
    def __init__(self):
        self.peak=0; self.cpu=[]; self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _sample(self):
        self.peak=max(self.peak,psutil.virtual_memory().used)
        self.cpu.append(psutil.cpu_percent(interval=None))
    def _run(self):
        self._sample()
        while not self.stop_event.wait(.2):
            self._sample()
    def start(self):
        self._sample(); self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=2); self._sample()
        return {'system_peak_ram_bytes':self.peak,'system_cpu_percent_samples':len(self.cpu),
                'system_cpu_percent_mean':sum(self.cpu[1:-1])/len(self.cpu[1:-1]) if len(self.cpu)>2 else None}


def percentile(values, probability):
    if not values:
        return None
    values=sorted(values)
    position=(len(values)-1)*probability
    lower=int(position); upper=math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def family_bootstrap(records, metric, repeats=1000, seed=116):
    groups=defaultdict(list)
    for row in records:
        value=row.get(metric)
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            groups[row['group_id']].append(float(value))
    values=[sum(group)/len(group) for group in groups.values()]
    if len(groups)<5:
        return {'mean_over_families':sum(values)/len(values) if values else None,
                'observations':sum(map(len,groups.values())),'interval':None,'families':len(groups),
                'reason':'Меньше пяти групп разметки; интервал не заявляется. Смысловая независимость групп не доказана.'}
    rng=random.Random(seed)
    samples=[sum(rng.choices(values,k=len(values)))/len(values) for _ in range(repeats)]
    return {'mean_over_families':sum(values)/len(values),'interval':[percentile(samples,.025),percentile(samples,.975)],
            'families':len(groups),'method':'percentile bootstrap по группам разметки, не по отдельным ходам',
            'assumption_limit':'Группировка не доказывает независимость; межгрупповая смысловая зависимость может сужать интервал.', 'resamples':repeats,'seed':seed}


def summarize(records, assessments):
    # Ключ включает режим: эталоны не смешиваются с фактической оценкой ответа.
    marked=defaultdict(list)
    for grade in assessments:
        marked[(grade['question_id'],grade['mode'],grade.get('run_id'))].append(grade)
    combined=[]
    for record in records:
        row=record.copy()
        row['assessments']=[]
        exact=marked.get((row['question_id'],row['mode'],row.get('run_id')),[])
        legacy=marked.get((row['question_id'],row['mode'],None),[]) if row.get('run_id') else []
        for grade in exact+legacy:
            parsed={'evaluator':grade.get('evaluator',''),'evaluation_kind':grade.get('evaluation_kind','')}
            for metric in ['fact_coverage_0_1','critical_error_present','grounding_0_2','citation_validity_0_2',
                           'action_semantic_correct','rewrite_semantic_correct','rewrite_conditions_correct','rewrite_clarification_correct']:
                value=grade.get(metric,'')
                if value is not None and str(value).strip()!='':
                    number=float(value)
                    maximum=2 if metric.endswith('_0_2') else 1
                    if not 0<=number<=maximum:
                        raise ValueError(f'Оценка {metric} вне шкалы: {value}')
                    parsed[metric]=number
            row['assessments'].append(parsed)
        combined.append(row)
    buckets=defaultdict(list)
    for row in combined:
        categories=['all',row.get('category','unknown')]
        if row.get('knowledge_case'):
            categories.append('knowledge:'+row['knowledge_case'])
        for category in categories:
            buckets[(row['mode'],category,row.get('cache_regime','unspecified'))].append(row)
    output=[]
    for (mode,category,cache),part in sorted(buckets.items()):
        entry={'mode':mode,'category':category,'cache_regime':cache,'examples':len(part),
               'families':len({row['group_id'] for row in part}), 'errors':sum(row.get('status')=='error' for row in part),
               'latency_seconds':{'n':len(part),'p50':percentile([row['total_seconds'] for row in part],.5),'p95':percentile([row['total_seconds'] for row in part],.95)},
               'human_assessed':sum(any(grade['evaluation_kind']=='human' and grade['evaluator'] for grade in row['assessments']) for row in part),
               'llm_assessed':sum(any(grade['evaluation_kind']=='llm' and grade['evaluator'] for grade in row['assessments']) for row in part)}
        for metric in ['fact_coverage_0_1','critical_error_present','grounding_0_2','citation_validity_0_2','action_semantic_correct',
                       'rewrite_semantic_correct','rewrite_conditions_correct','rewrite_clarification_correct']:
            # Человеческая и LLM-оценка агрегируются раздельно.
            entry[metric]={kind:family_bootstrap([{'group_id':row['group_id'],metric:grade[metric]} for row in part for grade in row['assessments']
                                                if grade['evaluation_kind']==kind and grade['evaluator'] and metric in grade],metric) for kind in ['human','llm']}
        for metric in ['recall_at_k','ndcg_at_k','span_recall_at_k','all_required_evidence_covered']:
            values=[row.get('retrieval_metrics',{}).get(metric) for row in part]
            values=[float(value) for value in values if value is not None]
            family_values=[{'group_id':row['group_id'],metric:float(row['retrieval_metrics'][metric])} for row in part
                           if row.get('retrieval_metrics',{}).get(metric) is not None]
            entry[metric]={'n':len(values),'mean':sum(values)/len(values) if values else None,
                           'family_estimate':family_bootstrap(family_values,metric)}
        stages=defaultdict(list)
        for row in part:
            for name,value in row.get('timings',{}).items():
                stages[name].append(value)
        entry['stages']={name:{'n':len(values),'p50':percentile(values,.5),'p95':percentile(values,.95)} for name,values in stages.items()}
        entry['system_peak_ram_bytes']=max((row.get('system_peak_ram_bytes',row.get('system_memory_after_bytes',0)) for row in part),default=0)
        output.append(entry)
    dialogues=defaultdict(list)
    for row in combined:
        if row.get('dialogue_id'):
            dialogues[(row['mode'],row['dialogue_id'])].append(row)
    dialog_results=[]
    for (mode,name),part in sorted(dialogues.items()):
        for kind in ['human','llm']:
            evaluators={grade['evaluator'] for row in part for grade in row['assessments'] if grade['evaluation_kind']==kind and grade['evaluator']}
            for evaluator in evaluators or {''}:
                grades=[next((grade for grade in row['assessments'] if grade['evaluation_kind']==kind and grade['evaluator']==evaluator),{}) for row in part]
                expected=next((row.get('dialogue_total_turns') for row in part if row.get('dialogue_total_turns')),None)
                whole=bool(expected) and len(part)==expected and {row.get('turn') for row in part}==set(range(1,expected+1))
                assessed=bool(evaluator) and whole and all(all(metric in grade for metric in ['action_semantic_correct','fact_coverage_0_1','critical_error_present']) for grade in grades)
                if mode=='rag':
                    assessed=assessed and all(all(metric in grade for metric in ['rewrite_semantic_correct','rewrite_conditions_correct','rewrite_clarification_correct'])
                                             for row,grade in zip(part,grades) if row.get('turn',1)>1)
                success=all(grade['action_semantic_correct']==1 and grade['fact_coverage_0_1']==1 and grade['critical_error_present']==0 and
                            (mode!='rag' or row.get('turn',1)<=1 or all(grade[metric]==1 for metric in ['rewrite_semantic_correct','rewrite_conditions_correct','rewrite_clarification_correct']))
                            for row,grade in zip(part,grades)) if assessed else None
                dialog_results.append({'mode':mode,'dialogue_id':name,'evaluation_kind':kind,'evaluator':evaluator,'turns':len(part),
                                       'expected_turns':expected,'whole_dialogue':whole,'success':success})
    return {'by_category':output,'dialogue_scenarios':dialog_results,
            'limits':['Нет результата качества без фактического прогона и соответствующей оценки.',
                      'Совпадение метки действия не является смысловой корректностью.',
                      'Системная RAM включает другие процессы; это не точная память модели.',
                      'p95 на малом числе наблюдений носит описательный характер.']}
