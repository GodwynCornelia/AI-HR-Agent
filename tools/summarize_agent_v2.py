"""Export synthetic diagnostic runs and recompute summaries from compact records."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/agent_v2'
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='append', default=[], help='kind:directory under runtime')
args = parser.parse_args()
outfile = BASE / 'recorded_runs.json'

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

if args.run:
    records = []
    expected = {q['id']:q for q in load(ROOT / 'data/benchmark/questions.json')}
    cases = {q['turns'][0]:q for q in load(BASE / 'protocol.json')['cases']}
    for entry in args.run:
        kind, name = entry.split(':', 1)
        folder = ROOT / 'runtime' / ('agent-dev' if kind == 'natural' else 'agent-v2') / name
        if kind == 'natural':
            configuration = load(folder / 'manifest.json')
            assert (folder / 'completion.json').exists()
            raw = [json.loads(line) for line in (folder / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
        else:
            configuration = {'first_search': 'empty', 'same_intervention': True,
                             'profile': 'config/experiment.env',
                             'source_commit': 'dd16477' if name.startswith('41a5b693-') else None,
                             'source_note': 'Other empty probes ran during development before the fast path commit.'}
            raw = [load(folder / f'{mode}.json') for mode in ('rag', 'agent')]
        for row in raw:
            d = row['diagnostics']
            agent = d.get('agent', {})
            usage = d.get('model_usage', {})
            records.append({
                'run': name, 'kind': kind, 'configuration': configuration, 'mode': row['mode'],
                'question': row['question'], 'question_id': row.get('question_id'),
                'expected': expected.get(row.get('question_id'),{}).get('expected_answer') or cases.get(row['question'],{}).get('expected'),
                'history': [{k: h.get(k) for k in ('question','search_query','answer')} for h in d.get('history', [])],
                'query': row.get('rewritten') or d.get('resolved_query'),
                'answer': row['answer'], 'status': row['status'], 'seconds': row['seconds'],
                'llm_calls': usage.get('llm_calls',0), 'searches': agent.get('searches',len(row.get('calls',[])) or int('retrieved' in d)),
                'reads': agent.get('reads',0), 'stop_reason':agent.get('stop_reason'), 'fallback':d.get('fallback'),
                'span_recall': row.get('cited_metrics',{}).get('span_recall_at_k'),
                'sources': [{k:s.get(k) for k in ('title','location','quote')} for s in d.get('result',{}).get('sources',[])],
                'dialogue_state':d.get('dialogue_state'), 'assessment':d.get('assessment'),
                'decisions': [{k: v.get(k) for k in ('action','query','clarification')} for v in d.get('decisions',[])],
                'graph_configuration': agent.get('configuration'), 'graph_code_sha256':agent.get('code_version_sha256'),
                'events': [{k:e.get(k) for k in ('node','seconds','action','tool','arguments','blocked','new_chunks','new_proofs','retrieval_state')} for e in agent.get('events',[])],
            })
    outfile.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
else:
    records = load(outfile)

summary = []
for run, mode in sorted({(r['run'],r['mode']) for r in records}):
    rows = [r for r in records if r['run']==run and r['mode']==mode]
    summary.append({'run':run,'kind':rows[0]['kind'],'mode':mode,'n':len(rows),
                    'median_seconds':statistics.median(r['seconds'] for r in rows),
                    'mean_llm_calls':statistics.mean(r['llm_calls'] for r in rows),
                    'searches':sum(r['searches'] for r in rows),'reads':sum(r['reads'] for r in rows),
                    'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}})
print(json.dumps(summary,ensure_ascii=False,indent=2))
