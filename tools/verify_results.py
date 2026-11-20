import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from assistant.assessment import percentile
for series in ['series_112','holdout_22']:
 base=ROOT/'experiments'/series
 records=json.loads((base/'records.json').read_text(encoding='utf-8'))
 summary=json.loads((base/'summary.json').read_text(encoding='utf-8'))
 for mode in ['no_rag','rag']:
  part=[r for r in records if r['mode']==mode]
  stored=next(r for r in summary['by_category'] if r['mode']==mode and r['category']=='all')
  assert len(part)==stored['examples']
  for key,prob in [('p50',.5),('p95',.95)]:assert abs(percentile([r['total_seconds'] for r in part],prob)-stored['latency_seconds'][key])<1e-7
  if mode=='rag':
   for metric in ['recall_at_k','ndcg_at_k','span_recall_at_k','all_required_evidence_covered']:
    values=[r['retrieval_metrics'][metric] for r in part if r.get('retrieval_metrics',{}).get(metric) is not None]
    assert len(values)==stored[metric]['n']
    assert abs(sum(values)/len(values)-stored[metric]['mean'])<1e-7
 print(series,len(records),'OK')
manifest=json.loads((ROOT/'data/rgr_20261118/corpus_manifest.json').read_text(encoding='utf-8'))
for row in manifest:assert hashlib.sha256((ROOT/'data/rgr_20261118/corpus'/row['file']).read_bytes()).hexdigest()==row['sha256']
assert len(manifest)==41
print('41 corpus files: hashes OK; tables verified from compact records, not regeneration of historical answers.')
