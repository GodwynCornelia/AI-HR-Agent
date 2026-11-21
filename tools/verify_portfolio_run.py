"""Recompute objectively observable measurements from a saved paired run."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for line in (args.run / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
keys = {(row['question_id'], row['mode']) for row in rows}
assert len(keys) == len(rows), 'Duplicate question/mode'

def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)

out = {'source_commit': manifest['source_commit'], 'expected_rows': 2 * len(manifest['ids']),
       'saved_rows': len(rows), 'complete': (args.run / 'completion.json').exists(), 'modes': {}}
if out['complete']:
    assert keys == {(question, mode) for question in manifest['ids'] for mode in ('rag', 'agent')}
for mode in ('rag', 'agent'):
    section = [row for row in rows if row['mode'] == mode]
    if not section:
        continue
    diagnostics = [row['diagnostics'] for row in section]
    delays = [row['seconds'] for row in section]
    usage = [d.get('model_usage', {}).get('llm_calls') for d in diagnostics]
    usage = [n for n in usage if n is not None]
    context = [row['provided_metrics']['all_required_evidence_covered'] for row in section]
    context = [v for v in context if v is not None]
    first_recall = [row.get('first_search_metrics',{}).get('recall_at_k') for row in section]
    first_recall = [v for v in first_recall if v is not None]
    citations = [source for row in section for source in row.get('result', {}).get('sources', [])]
    out['modes'][mode] = {
        'rows': len(section), 'p50_seconds': statistics.median(delays), 'p95_seconds': percentile(delays,.95),
        'statuses': dict(Counter(row['status'] for row in section)),
        'searches': sum(d.get('agent', {}).get('searches', int('retrieved' in d)) for d in diagnostics),
        'reads': sum(d.get('agent', {}).get('reads', 0) for d in diagnostics),
        'llm_calls': {'sum': sum(usage), 'observed_rows': len(usage)},
        'partial_answers': sum(bool(row.get('result', {}).get('partial')) for row in section),
        'stop_reasons': dict(Counter(d.get('agent', {}).get('stop_reason','linear') for d in diagnostics)),
        'literal_context_coverage': {'covered': sum(context), 'applicable': len(context)},
        'first_recall_at_8': {'sum': sum(first_recall), 'applicable': len(first_recall),
                              'mean': statistics.mean(first_recall) if first_recall else None},
        'literal_citation_match': {'exact': sum(bool(c.get('quote')) and c['quote'] in c.get('text','') for c in citations),
                                  'total': len(citations)},
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
