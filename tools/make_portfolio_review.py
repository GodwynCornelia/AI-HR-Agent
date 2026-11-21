"""Prepare blinded paired answers from a completed final run."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('frozen_protocol',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
assert (args.run/'completion.json').is_file()
protocol=json.loads(args.frozen_protocol.read_text(encoding='utf-8'))
questions={q['id']:q for q in protocol['questions']}
pairs=defaultdict(dict)
for line in (args.run/'answers.jsonl').read_text(encoding='utf-8').splitlines():
    row=json.loads(line)
    pairs[row['question_id']][row['mode']]=row
assert pairs.keys()==questions.keys() and all(set(p)=={'rag','agent'} for p in pairs.values())
rng=random.Random(116)
ids=sorted(pairs)
random_ids=rng.sample(ids,4)
remaining=[i for i in ids if i not in random_ids]
remaining.sort(key=lambda i: (pairs[i]['rag']['status']==pairs[i]['agent']['status'],
                              pairs[i]['rag']['answer']==pairs[i]['agent']['answer']))
selected=random_ids+remaining[:4]
args.output.mkdir(parents=True,exist_ok=False)
lines=['# Парная оценка ответов','',
       'Для каждого варианта отметьте обязательные факты, существенные ложные утверждения и верность действия. Для покрытия используйте 0, 0,5 или 1. Для действия используйте 0 или 1. Если факт искать не нужно, оцените действие отдельно.','']
key={}
for slot,identifier in enumerate(selected,1):
    q=questions[identifier]
    key[slot]={'question_id':identifier,'selection':'random' if slot<=4 else 'targeted'}
    lines += [f"## Пара {slot}",'',q['question'],'',
              'Ожидаемый результат — '+q['expected_answer'],'']
    modes=['rag','agent'];rng.shuffle(modes)
    for label,mode in zip(('X','Y'),modes):
        row=pairs[identifier][mode]
        key[slot][label]=mode
        lines += [f'### Вариант {label}','']
        for step in row.get('history',[]):
            lines += ['Предыдущий вопрос — '+step['question'],'Предыдущий ответ — '+step.get('answer',''),'']
        lines += [row['answer'],'']
        for source in row.get('result',{}).get('sources',[]):
            lines += [f"- {source['title']}, {source['location']} — {source['quote']}"]
        lines += ['','Покрытие ___ · Критическая ошибка ___ · Действие ___ · Основание ___','']
(args.output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
(args.output/'mode_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
