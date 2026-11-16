"""Shared paths, Day 6 embedder and the retrieval methods compared in Day 7.

Methods (each returns chunks with id/text/source_id/source/page/metadata/score/rank):
  dense   - Day 6 Chroma `day6_B` + bge-small query_embed (cosine, HNSW).
  bm25    - Okapi BM25 over the same 476 chunks (lexical, k1=1.5, b=0.75).
  hybrid  - Reciprocal Rank Fusion of dense and bm25 candidate lists (k=60).
  rerank  - hybrid candidates re-scored by a cross-encoder (ms-marco-MiniLM-L-6-v2).
Filters only come from the question's user-supplied `filter` field ({'book': ...}).
"""
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from baseline import hit, pack_context  # noqa: E402,F401  (reuse starter definitions)

DATA_DIR = ROOT / 'data' / 'day7'
CHUNKS_PATH = DATA_DIR / 'chunks.json'
QUESTIONS_PATH = DATA_DIR / 'questions.json'
CASES_PATH = DATA_DIR / 'cases.json'
INDEX_DIR = ROOT / 'starter' / 'data' / 'chroma_day6_B'  # rebuilt locally, not committed
CACHE_DIR = ROOT / 'starter' / 'data' / 'fastembed_cache'
RESULTS_DIR = ROOT / 'results'

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBED_DIMS = 384
RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class Embedder:
    """Day 6 embedding setup; failures propagate (no random/mock fallback)."""
    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE_DIR))
        self.info = {'embedding_model': EMBED_MODEL, 'dims': EMBED_DIMS, 'fastembed_version': version('fastembed'),
                     'chromadb_version': version('chromadb')}

    def documents(self, texts):
        out = [v.tolist() for v in self.model.embed(texts)]
        if len(out) != len(texts) or any(len(v) != EMBED_DIMS for v in out):
            raise ValueError('invalid document embeddings')
        return out

    def query(self, text):
        v = next(iter(self.model.query_embed(text))).tolist()
        if len(v) != EMBED_DIMS:
            raise ValueError('invalid query dimension')
        return v


def _matches(chunk, filters):
    return not filters or all(chunk['metadata'].get(k) == v for k, v in filters.items())


def _ranked(items, strategy):
    return [{**c, 'rank': i, 'strategy': strategy} for i, c in enumerate(items, 1)]


class DenseRetriever:
    name = 'dense'

    def __init__(self, chunks, embedder=None):
        import chromadb
        if not INDEX_DIR.exists():
            raise FileNotFoundError(f'{INDEX_DIR} missing: run src/prepare_data.py first')
        self.by_id = {c['id']: c for c in chunks}
        self.embedder = embedder or Embedder()
        self.col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection('day6_B')
        if self.col.count() != len(chunks):
            raise ValueError('index size does not match chunks.json')

    def search(self, query, *, filters=None, top_k=3):
        vec = self.embedder.query(query)
        where = None
        if filters:
            where = {k: v for k, v in filters.items()} if len(filters) == 1 else {'$and': [{k: v} for k, v in filters.items()]}
        r = self.col.query(query_embeddings=[vec], n_results=top_k, where=where, include=['distances'])
        out = [{**self.by_id[i], 'score': 1 - d} for i, d in zip(r['ids'][0], r['distances'][0])]
        return _ranked(out, self.name)


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


STOP = set('the a an of to and in is are be by for on or that this it as with what which who how why should does do '
           'can his their its from at when into than then there these those was were may according'.split())


class BM25Retriever:
    name = 'bm25'

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(t for t in tokenize(c['text']) if t not in STOP) for c in chunks]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, *, filters=None, top_k=3):
        terms = [t for t in tokenize(query) if t not in STOP and t in self.idf]
        scored = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if not _matches(c, filters):
                continue
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    for t in terms if t in tf)
            if s > 0:
                scored.append({**c, 'score': s})
        scored.sort(key=lambda c: (-c['score'], c['id']))
        return _ranked(scored[:top_k], self.name)


class HybridRetriever:
    """RRF over ranks, so dense cosine and BM25 scores are never added directly."""
    name = 'hybrid'

    def __init__(self, dense, bm25, candidates=20, k=60):
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def search(self, query, *, filters=None, top_k=3):
        fused, keep = Counter(), {}
        for r in (self.dense.search(query, filters=filters, top_k=self.candidates),
                  self.bm25.search(query, filters=filters, top_k=self.candidates)):
            for c in r:
                fused[c['id']] += 1 / (self.k + c['rank'])
                keep.setdefault(c['id'], {k: v for k, v in c.items() if k not in ('score', 'rank', 'strategy')})
                keep[c['id']][f"{c['strategy']}_rank"] = c['rank']
        items = [{**keep[i], 'score': s} for i, s in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
        return _ranked(items, self.name)


class RerankRetriever:
    name = 'rerank'

    def __init__(self, first_stage, candidates=20):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.first, self.candidates = first_stage, candidates
        self.model = TextCrossEncoder(RERANK_MODEL, cache_dir=str(CACHE_DIR))

    def search(self, query, *, filters=None, top_k=3):
        cands = self.first.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = list(self.model.rerank(query, [c['text'] for c in cands]))
        items = [{**c, 'first_stage_rank': c['rank'], 'score': float(s)} for c, s in zip(cands, scores)]
        items.sort(key=lambda c: (-c['score'], c['id']))
        return _ranked(items[:top_k], self.name)


def build_retrievers(chunks, names):
    out, dense, bm25 = {}, None, None
    for n in names:
        if n in ('dense', 'hybrid', 'rerank') and dense is None:
            dense = DenseRetriever(chunks)
        if n in ('bm25', 'hybrid', 'rerank') and bm25 is None:
            bm25 = BM25Retriever(chunks)
    for n in names:
        if n == 'dense': out[n] = dense
        elif n == 'bm25': out[n] = bm25
        elif n == 'hybrid': out[n] = HybridRetriever(dense, bm25)
        elif n == 'rerank': out[n] = RerankRetriever(HybridRetriever(dense, bm25))
        else: raise ValueError(n)
    return out


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1000
