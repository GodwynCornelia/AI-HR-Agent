"""Recompute objectively observable measurements from a saved paired run."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--protocol',type=Path)
parser.add_argument('--check',type=Path)
parser.add_argument('--corpus-manifest',type=Path)
parser.add_argument('--grades',type=Path)
args = parser.parse_args()
manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for line in (args.run / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
keys = {(row['question_id'], row['mode']) for row in rows}
assert len(keys) == len(rows), 'Duplicate question/mode'
if args.protocol:
    import hashlib
    raw=args.protocol.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['set_sha256']
    protocol=json.loads(raw)
    assert manifest['source_commit']==protocol['source_commit']
    assert manifest['ids']==[q['id'] for q in protocol['questions']]
    questions={q['id']:q for q in protocol['questions']}
    assert all(r['question']==questions[r['question_id']]['question'] for r in rows)
else:
    questions={}
if args.corpus_manifest:
    import sys
    here=Path(__file__).resolve().parent
    if (here/'evaluation.py').exists():
        from evaluation import retrieval_metrics
    else:
        sys.path.insert(0,str(here.parent))
        from assistant.evaluation import retrieval_metrics
    corpus_bytes=args.corpus_manifest.read_bytes()
    assert hashlib.sha256(corpus_bytes).hexdigest()==manifest['corpus_manifest_sha256']
    corpus=json.loads(corpus_bytes)
    by_file={d['file']:d['sha256'] for d in corpus}
    by_title={d['title']:d['sha256'] for d in corpus}
    assert len(by_title)==len(corpus)
    for row in rows:
        refs=[dict(ref,sha256=by_file[ref['document']]) for ref in questions[row['question_id']]['evidence']]
        def measure(sources,k):
            documents={s['document_id']:by_title[s['title']] for s in sources}
            return retrieval_metrics(sources,refs,documents,k)
        searches=row.get('ranked_searches',[])
        first=measure(searches[0]['sources'] if searches else [],8)
        assert first['recall_at_k']==row['first_search_metrics']['recall_at_k']
        provided=measure(row['diagnostics'].get('provided_sources',[]),len(row['diagnostics'].get('provided_sources',[])))
        assert provided['all_required_evidence_covered']==row['provided_metrics']['all_required_evidence_covered']
        pool={s['id']:s for step in searches for s in step['sources']}
        for s in row['diagnostics'].get('retrieved',[]):
            pool[s['id']]=s
        row['pool_recall']=measure(list(pool.values()),len(pool))['recall_at_k']
        cited=[{**s,'text':s['quote']} for s in row.get('result',{}).get('sources',[])]
        row['cited_recall']=measure(cited,len(cited))['recall_at_k']

def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)

out = {'source_commit': manifest['source_commit'], 'expected_rows': 2 * len(manifest['ids']),
       'saved_rows': len(rows), 'complete': (args.run / 'completion.json').exists(), 'modes': {}}
out['completed_pairs']=sum((i,'rag') in keys and (i,'agent') in keys for i in manifest['ids'])
if out['complete']:
    assert keys == {(question, mode) for question in manifest['ids'] for mode in ('rag', 'agent')}
    completion=json.loads((args.run/'completion.json').read_text(encoding='utf-8'))
    assert completion['expected']==completion['records']==len(rows)
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
    if args.corpus_manifest:
        applicable=[r for r in section if r['pool_recall'] is not None]
        gains=[r['pool_recall']-r['first_search_metrics']['recall_at_k'] for r in applicable]
        out['modes'][mode]['pool_evidence']={'n':len(gains),'positive_gain':sum(g>0 for g in gains),'gain_sum':sum(gains)}
    if questions:
        by_category={}
        for category in sorted({q['category'] for q in questions.values()}):
            part=[r for r in section if questions[r['question_id']]['category']==category]
            first=[r.get('first_search_metrics',{}).get('recall_at_k') for r in part]
            first=[x for x in first if x is not None]
            context=[r['provided_metrics']['all_required_evidence_covered'] for r in part]
            context=[x for x in context if x is not None]
            by_category[category]={'rows':len(part),'first_recall_sum':sum(first),'first_recall_n':len(first),
                                   'full_context':sum(context),'applicable_context':len(context),
                                   'statuses':dict(Counter(r['status'] for r in part))}
        out['modes'][mode]['categories']=by_category
if args.grades:
    grades=[json.loads(line) for line in args.grades.read_text(encoding='utf-8').splitlines()]
    indexed={(g['question_id'],g['mode']):g for g in grades}
    assert len(indexed)==len(grades) and indexed.keys()<=keys
    def success(g):
        return bool(g and g['coverage']==1 and not g['critical_error'] and g['action_correct']==1)
    out['preliminary_llm']={}
    for mode in ('rag','agent'):
        part=[g for g in grades if g['mode']==mode]
        available=[r for r in rows if r['mode']==mode]
        failures={r['question_id'] for r in available if r['status']=='error'}
        successful={g['question_id'] for g in part if success(g)}-failures
        out['preliminary_llm'][mode]={'attempted':len(available),'graded':len(part),'success':len(successful),
                                    'critical_errors':sum(g['critical_error'] for g in part),
                                    'correct_actions':sum(g['action_correct'] for g in part)}
        if args.corpus_manifest:
            possible=[r for r in available if r['pool_recall'] is not None and r['first_search_metrics']['recall_at_k']<1]
            recovered=[r['question_id'] for r in possible
                       if r['pool_recall']>r['first_search_metrics']['recall_at_k'] and
                       r['cited_recall']==1 and r['question_id'] in successful]
            out['preliminary_llm'][mode]['recovery_candidates']={'n':len(possible),'qualified_ids':recovered}
    groups={}
    for q in questions.values():
        groups.setdefault(q['group_id'],[]).append(q['id'])
    deltas=Counter()
    for ids in groups.values():
        if not all((i,m) in indexed for i in ids for m in ('rag','agent')):
            continue
        statuses={(r['question_id'],r['mode']):r['status'] for r in rows}
        a,b=(all(success(indexed[i,m]) and statuses[i,m]!='error' for i in ids) for m in ('rag','agent'))
        deltas['agent_better' if b>a else 'rag_better' if a>b else 'equal']+=1
    out['preliminary_llm']['paired_scenarios']=dict(deltas)
if args.check:
    assert out==json.loads(args.check.read_text(encoding='utf-8')), 'Summary does not match saved rows'
print(json.dumps(out, ensure_ascii=False, indent=2))
