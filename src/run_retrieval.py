"""Retrieval/context benchmark on one split, original query only (no agent).

    .venv/Scripts/python src/run_retrieval.py --split dev --config configs/retrieval_dev.json

Hit uses starter `hit`: correct source_id + page and evidence fully inside the chunk.
Context = starter `pack_context` over the top-k results with the config budget (chars).
Gold is used only for scoring after retrieval; filters come from the question input.
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHUNKS_PATH, QUESTIONS_PATH, RESULTS_DIR, Timer, build_retrievers, dump_json, hit,  # noqa: E402
                    load_json, pack_context)


def run(cfg, retriever, questions):
    traces = []
    for q in questions:
        with Timer() as t:
            ranked = retriever.search(q['question'], filters=q.get('filter'), top_k=max(cfg['top_k'], 10))
        top = ranked[:cfg['top_k']]
        context, ids = pack_context(top, budget=cfg['budget'])
        marks = [hit(c, q['gold']) for c in ranked]
        first = next((i for i, m in enumerate(marks, 1) if m), None)
        traces.append({'query_id': q['id'], 'question': q['question'], 'filter': q.get('filter'),
                       'results': [{'id': c['id'], 'page': c['page'], 'source_id': c['source_id'], 'rank': c['rank'],
                                    'score': round(c['score'], 5), 'hit': m} for c, m in zip(ranked[:10], marks)],
                       'hit1': bool(marks[:1] and marks[0]), 'hit3': any(marks[:3]), 'mrr10': 1 / first if first and first <= 10 else 0,
                       'context_ids': ids, 'context_chars': len(context),
                       'context_hit': any(m and c['id'] in ids for c, m in zip(top, marks)), 'ms': t.ms})
    n = len(traces)
    frac = lambda k: {'n': sum(t[k] for t in traces), 'N': n, 'rate': round(sum(t[k] for t in traces) / n, 3)}
    return {'name': cfg['name'], 'method': cfg['method'], 'top_k': cfg['top_k'], 'budget_chars': cfg['budget'],
            'hit1': frac('hit1'), 'evidence_hit3': frac('hit3'), 'context_evidence_rate': frac('context_hit'),
            'mrr10': round(statistics.mean(t['mrr10'] for t in traces), 3),
            'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces)),
            'median_ms': round(statistics.median(t['ms'] for t in traces), 1), 'cost_usd': 0}, traces


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--config', required=True, type=Path)
    a = p.parse_args()
    configs = load_json(a.config)
    chunks = load_json(CHUNKS_PATH)
    questions = [q for q in load_json(QUESTIONS_PATH) if q['split'] == a.split]
    retrievers = build_retrievers(chunks, sorted({c['method'] for c in configs}))
    for r in retrievers.values():  # warm-up so model load is not counted as query latency
        r.search('warm up', top_k=1)
    rows = []
    for cfg in configs:
        summary, traces = run(cfg, retrievers[cfg['method']], questions)
        summary['split'] = a.split
        rows.append(summary)
        dump_json(RESULTS_DIR / 'retrieval' / a.split / f"{cfg['name']}.traces.json", traces)
    dump_json(RESULTS_DIR / 'retrieval' / a.split / 'summary.json', rows)
    print('| Config | Method | k | Budget | Hit@1 | Hit@3 | Context | MRR@10 | ctx chars | median ms |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        f = lambda m: f"{r[m]['n']}/{r[m]['N']} ({r[m]['rate']:.0%})"
        print(f"| {r['name']} | {r['method']} | {r['top_k']} | {r['budget_chars']} | {f('hit1')} | {f('evidence_hit3')} | "
              f"{f('context_evidence_rate')} | {r['mrr10']} | {r['avg_context_chars']} | {r['median_ms']} |")


if __name__ == '__main__':
    main()
