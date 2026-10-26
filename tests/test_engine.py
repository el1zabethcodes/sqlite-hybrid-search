"""Offline unit tests: run with `python -m unittest discover -s tests`.

A deterministic bag-of-words embedder stands in for FastEmbed so the tests
never touch the network or an ONNX model. Both vector backends (sqlite-vec
KNN and the pure-Python brute-force fallback) are exercised.
"""

import os
import tempfile
import unittest
import zlib

from sqlite_hybrid_search import HybridSearchEngine, rrf

DIM = 16

DOCS = [
    "SQLite is a lightweight C-language library that implements a SQL database engine.",
    "Vector databases allow high-dimensional embeddings similarity search.",
    "Reciprocal Rank Fusion fuses search rankings from multiple retrieval algorithms.",
    "FastEmbed is a lightweight and ultra-fast Python library for embeddings generation.",
    "Local-first software preserves user privacy by keeping data on the device.",
]


def fake_embed(texts):
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for token in text.lower().split():
            vec[zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
        out.append(vec)
    return out


class RrfTests(unittest.TestCase):
    def test_known_scores(self):
        scores = rrf([[1, 2], [2, 3]], k=60)
        self.assertAlmostEqual(scores[1], 1 / 61)
        self.assertAlmostEqual(scores[2], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[3], 1 / 62)

    def test_consensus_document_wins(self):
        # Doc 2 is the only one retrieved by both lists, so it must win.
        scores = rrf([[1, 2, 3], [2, 4, 5]], k=60)
        self.assertEqual(max(scores, key=scores.get), 2)

    def test_configurable_k(self):
        scores = rrf([[1, 2]], k=10)
        self.assertAlmostEqual(scores[1], 1 / 11)


class EngineTests(unittest.TestCase):
    def _make_engine(self, use_vec: bool, db_path: str) -> HybridSearchEngine:
        return HybridSearchEngine(
            db_path=db_path,
            embed_fn=fake_embed,
            load_vec_extension=use_vec,
        )

    def test_crud_and_hybrid_on_both_backends(self):
        for use_vec in (True, False):
            with self.subTest(backend="sqlite-vec" if use_vec else "python"):
                with tempfile.TemporaryDirectory() as tmp:
                    engine = self._make_engine(use_vec, os.path.join(tmp, "t.db"))
                    try:
                        ids = engine.insert_documents(DOCS)
                        self.assertEqual(ids, [1, 2, 3, 4, 5])

                        # Keyword leg (BM25): "lightweight" hits docs 1 and 4.
                        self.assertCountEqual(engine.search_fts("lightweight"), [1, 4])
                        self.assertEqual(engine.search_fts("privacy"), [5])
                        # OR maximizes recall; AND requires every token.
                        self.assertCountEqual(
                            engine.search_fts("sqlite lightweight"), [1, 4]
                        )
                        self.assertEqual(
                            engine.search_fts("sqlite lightweight", operator="AND"), [1]
                        )
                        # Punctuation and FTS5 operators are neutralized.
                        self.assertEqual(engine.search_fts('"; DROP TABLE documents; --'), [])
                        self.assertEqual(engine.search_fts("!!!"), [])

                        # Vector leg: "database engine" tokens appear in doc 1.
                        self.assertEqual(engine.search_vector("database engine")[0], 1)

                        # Hybrid fusion returns well-formed, score-ordered hits;
                        # the leader is a consensus doc retrieved by both legs.
                        hits = engine.hybrid_search_rrf("fast local vector search", top_k=3)
                        self.assertTrue(1 <= len(hits) <= 3)
                        self.assertIsNotNone(hits[0]["fts_rank"])
                        self.assertIsNotNone(hits[0]["vec_rank"])
                        for hit in hits:
                            self.assertIn("id", hit)
                            self.assertIn("score", hit)
                            self.assertIn("content", hit)
                        scores = [hit["score"] for hit in hits]
                        self.assertEqual(scores, sorted(scores, reverse=True))

                        # UPDATE keeps the FTS index in sync.
                        engine.update_document(
                            5, "Postgres is a heavyweight client-server database engine."
                        )
                        self.assertEqual(engine.search_fts("privacy"), [])

                        # DELETE keeps the FTS index in sync.
                        engine.delete_document(3)
                        self.assertEqual(engine.search_fts("fusion"), [])

                        # A keyword with no lexical match never reaches FTS,
                        # but the vector leg (no similarity threshold) still
                        # ranks the corpus, so hybrid hits carry vec_rank only.
                        self.assertEqual(engine.search_fts("zzzqqq"), [])
                        for hit in engine.hybrid_search_rrf("zzzqqq"):
                            self.assertIsNone(hit["fts_rank"])
                            self.assertIsNotNone(hit["vec_rank"])
                    finally:
                        engine.close()

    def test_reopen_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.db")
            engine = self._make_engine(False, path)
            engine.insert_documents(DOCS)
            engine.close()

            reopened = self._make_engine(False, path)
            try:
                self.assertEqual(reopened.search_fts("privacy"), [5])
                self.assertEqual(reopened.search_vector("database engine")[0], 1)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
