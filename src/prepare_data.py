"""Build Day 7 chunks/benchmark from the Day 6 corpus (read-only) and index them.

Run from repo root:
    .venv/Scripts/python src/prepare_data.py --day6 "<path to Day 6 repo>"

- Chunking = Day 6 strategy B (paragraph groups, max 1200 / min 200 chars),
  copied from Day 6 `starter/src/chunking.py::by_paragraph` so chunk IDs match.
- Embedding = BAAI/bge-small-en-v1.5 via fastembed 0.9.1 (384 dims, cosine),
  documents with `embed`, queries with `query_embed`, same as Day 6.
- Vector store = persistent Chroma 1.5.9, collection `day6_B` (Day 6 name).
Day 6's chroma_db was not committed, so the index is rebuilt here with the same
code, model and chunk IDs. The benchmark gold never enters the index.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import CHUNKS_PATH, INDEX_DIR, QUESTIONS_PATH, CASES_PATH, DATA_DIR  # noqa: E402

DAY6_COMMIT = '9aab448'


def by_paragraph(page, max_chars=1200, min_chars=200):
    """Same algorithm as Day 6 by_paragraph, for one page dict."""
    text = page['text']
    pieces = []
    for m in re.finditer(r'[^\n]+', text):
        if len(m[0]) <= max_chars:
            pieces.append((m.start(), m.end()))
            continue
        for s in re.finditer(r'.+?(?:[.!?](?=\s|$)|$)', m[0]):
            a = m.start() + s.start(); b = m.start() + s.end()
            while a < b and text[a].isspace():
                a += 1
            if b > a:
                pieces.append((a, b))
    groups = []
    for a, b in pieces:
        if groups and b - groups[-1][0] <= max_chars:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    if len(groups) > 1 and groups[-1][1] - groups[-1][0] < min_chars:
        groups[-2][1] = groups[-1][1]; groups.pop()
    out = []
    for i, (a, b) in enumerate(groups, 1):
        meta = {**page['metadata'], 'chunk_index': i, 'char_len': b - a,
                'chunk_char_start': a, 'chunk_char_end': b, 'chunking': 'day6_B'}
        out.append({'id': f"{meta['page_id']}-B{i:03d}", 'text': text[a:b],
                    'source_id': page['document_id'], 'source': page['source'],
                    'page': page['page'], 'metadata': meta})
    return out


def page_number(page_id):
    return int(page_id.split('-p')[1])


def convert_question(q):
    return {'id': q['id'], 'split': q['split'], 'type': q['type'], 'question': q['question'],
            'expected_answer': q['expected_answer'],
            'gold': [{'source_id': q['gold_source_id'], 'page': page_number(p), 'evidence': q['evidence_quote']}
                     for p in q['gold_page']],
            'filter': q.get('filter'), 'answerable': True, 'origin': 'day6 benchmark/questions.jsonl'}


# Added for Day 7: near-topic questions whose answer is absent from the corpus.
# Absence checked by grepping the whole corpus (not top-k): see absence_reason.
UNANSWERABLE = [
    {'id': 'u01', 'split': 'dev', 'question': "On what date did Machiavelli dedicate The Prince to Lorenzo de' Medici?",
     'absence_reason': "The dedication precedes CHAPTER I and was excluded by Day 6 ingestion; 'Lorenzo' and 'dedicat' have 0 matches in the 204 units."},
    {'id': 'u02', 'split': 'dev', 'question': 'What exact percentage tax rate does Machiavelli recommend a prince levy on merchants?',
     'absence_reason': "Taxes are discussed (PRN chapters on liberality/encouraging trade) but no rate is given; 'merchant', 'percent', 'per cent' have 0 matches in PRN units."},
    {'id': 'u03', 'split': 'test', 'question': 'What does Sun Tzu advise about using gunpowder and firearms in siege warfare?',
     'absence_reason': "'gunpowder', 'firearm', 'musket', 'cannon' have 0 matches in AOW units; the only 'artillery' match is in The Prince. Sieges and fire attacks are discussed without gunpowder."},
    {'id': 'u04', 'split': 'test', 'question': 'How many pounds of rice per day does Sun Tzu allocate to each soldier?',
     'absence_reason': "AOW discusses foraging and the picul (133.3 pounds) as a unit of provender equivalence, but no daily per-soldier ration; 'ration' only matches 'deliberations'/'operations'."},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--skip-index', action='store_true')
    args = parser.parse_args()
    units = [json.loads(line) for line in (args.day6 / 'starter/data/units.jsonl').read_text(encoding='utf-8').splitlines() if line]
    day6_q = [json.loads(line) for line in (args.day6 / 'benchmark/questions.jsonl').read_text(encoding='utf-8').splitlines() if line]
    chunks = [c for u in units for c in by_paragraph(u)]
    assert len(units) == 204 and len(chunks) == 476, (len(units), len(chunks))  # matches Day 6 ingest_stats B
    questions = [convert_question(q) for q in day6_q]
    for q in questions:
        for g in q['gold']:
            if not any(c['source_id'] == g['source_id'] and c['page'] == g['page'] and g['evidence'] in c['text'] for c in chunks):
                raise ValueError(f"{q['id']}: evidence not inside one chunk")
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': True} for q in questions]
    for u in UNANSWERABLE:
        cases.append({**u, 'answerable': False, 'expected_behavior': 'search_and_abstain'})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copy(args.day6 / 'starter/data/manifest.json', DATA_DIR / 'day6_manifest.json')
    counts = {s: {'answerable': sum(c['split'] == s and c['answerable'] for c in cases),
                  'unanswerable': sum(c['split'] == s and not c['answerable'] for c in cases)} for s in ('dev', 'test')}
    info = {'day6_repo': 'tcaopham216-gif/K5-L3B-Day06-CaoPhamThanhDat-2A202603970-DataIngestion',
            'day6_commit': DAY6_COMMIT, 'units': len(units), 'chunks': len(chunks), 'benchmark': counts}
    if not args.skip_index:
        from common import Embedder
        import chromadb
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        started = time.perf_counter()
        emb = Embedder()
        vectors = emb.documents([c['text'] for c in chunks])
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        col = client.get_or_create_collection('day6_B', metadata={'hnsw:space': 'cosine'})
        for s in range(0, len(chunks), 128):
            b = chunks[s:s + 128]
            col.upsert(ids=[c['id'] for c in b], documents=[c['text'] for c in b], embeddings=vectors[s:s + 128],
                       metadatas=[{**{k: v for k, v in c['metadata'].items() if isinstance(v, (str, int, float, bool))},
                                   'document_id': c['source_id'], 'source': c['source'], 'page': c['page']} for c in b])
        info.update(index_seconds=time.perf_counter() - started, collection='day6_B', count=col.count(), **emb.info,
                    metric='cosine', hnsw_approximate=True, cost_usd=0)
        assert col.count() == len(chunks)
    (DATA_DIR / 'prepare_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
