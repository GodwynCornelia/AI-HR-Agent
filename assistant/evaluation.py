"""Автоматические проверки retrieval. Оценка смысловой корректности остаётся ручной."""
import math
import re


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def reference_spans(reference):
    return [normalized(span) for span in reference.get('spans',[reference['quote']])]


def chunk_relevant(source, reference, sha_by_document):
    return (sha_by_document.get(str(source['document_id']))==reference['sha256'] and
            any(span in normalized(source['text']) for span in reference_spans(reference)))


def available_texts(sources, document_hash, sha_by_document):
    selected=[source for source in sources if sha_by_document.get(str(source['document_id']))==document_hash]
    texts=[normalized(source['text']) for source in selected]
    # Объединять можно только действительно соседние чанки той же страницы/части.
    ordered=sorted([source for source in selected if 'ordinal' in source],key=lambda source:(source['document_id'],source['ordinal']))
    merged=''; previous=None
    for source in ordered:
        current=normalized(source['text'])
        adjacent=(previous is not None and source['document_id']==previous['document_id'] and
                  source['location']==previous['location'] and source['ordinal']==previous['ordinal']+1)
        overlap=next((size for size in range(min(len(merged),len(current),300),4,-1) if merged[-size:]==current[:size]),0) if adjacent else 0
        merged=merged+current[overlap:] if overlap else current
        texts.append(merged)
        previous=source
    return texts


def retrieval_metrics(sources, references, sha_by_document, k, relevant_chunk_count=None):
    if not references:
        return {'recall_at_k': None, 'ndcg_at_k': None, 'all_required_evidence_covered':None, 'span_recall_at_k':None}
    found = set()
    gains = []
    # Доказательство определяется документом и цитатой, не случайным UUID чанка.
    for source in sources[:k]:
        matches = set()
        for index, reference in enumerate(references):
            if chunk_relevant(source,reference,sha_by_document):
                matches.add(index)
        # Бинарная релевантность чанка для стандартного nDCG.
        gains.append(int(bool(matches)))
    found=set(); covered_spans=0; total_spans=0
    for index,reference in enumerate(references):
        texts=available_texts(sources[:k],reference['sha256'],sha_by_document)
        spans=reference_spans(reference)
        coverage=[any(span in text for text in texts) for span in spans]
        covered_spans+=sum(coverage); total_spans+=len(spans)
        if all(coverage):
            found.add(index)
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, relevant_chunk_count or 0)))
    return {'recall_at_k': len(found) / len(references),
            'ndcg_at_k': dcg / idcg if idcg else None,
            'evidence_found': sorted(found), 'evidence_count': len(references),
            'all_required_evidence_covered':len(found)==len(references),
            'span_recall_at_k':covered_spans/total_spans if total_spans else None}
