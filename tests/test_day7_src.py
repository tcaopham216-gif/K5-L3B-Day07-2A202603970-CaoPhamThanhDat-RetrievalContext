"""Offline tests for src/ (no model, no vector store): BM25, RRF fusion and filters."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import BM25Retriever, HybridRetriever  # noqa: E402


def chunk(i, text, book='art_of_war'):
    return {'id': i, 'text': text, 'source_id': book, 'source': f'{book}.txt', 'page': 1, 'metadata': {'book': book}}


CHUNKS = [chunk('a', 'All warfare is based on deception.'),
          chunk('b', 'The best fortress is not to be hated by the people.', 'the_prince'),
          chunk('c', 'Send colonies to one or two places, which may be as keys to that state.', 'the_prince')]


class FakeRanked:
    """Stands in for the dense retriever only to test rank fusion, never for benchmark numbers."""
    name = 'dense'

    def __init__(self, order):
        self.order = order

    def search(self, query, *, filters=None, top_k=3):
        return [{**next(c for c in CHUNKS if c['id'] == i), 'score': 1.0, 'rank': r, 'strategy': 'dense'}
                for r, i in enumerate(self.order[:top_k], 1)]


class Day7SrcTests(unittest.TestCase):
    def test_bm25_ranks_and_filters(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search('fortress hated people')[0]['id'], 'b')
        self.assertEqual(bm25.search('fortress', filters={'book': 'art_of_war'}), [])
        self.assertEqual([r['rank'] for r in bm25.search('colonies state places', top_k=3)], [1])

    def test_rrf_fuses_ranks_not_raw_scores(self):
        hybrid = HybridRetriever(FakeRanked(['c', 'a', 'b']), BM25Retriever(CHUNKS), candidates=3)
        out = hybrid.search('colonies keys state', top_k=2)
        self.assertEqual(out[0]['id'], 'c')  # rank 1 in both lists
        self.assertEqual(len({r['id'] for r in out}), len(out))
        self.assertEqual([r['rank'] for r in out], [1, 2])


if __name__ == '__main__':
    unittest.main()
