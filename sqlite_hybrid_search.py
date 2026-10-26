"""sqlite-hybrid-search: a zero-dependency local hybrid search engine.

Three retrieval components live inside a single standard SQLite file:

* BM25 keyword search  -> SQLite FTS5 as an external-content virtual table,
                          kept in sync by triggers, unicode61 tokenizer.
* Semantic search      -> FastEmbed ONNX embeddings stored as little-endian
                          float32 blobs. Uses the sqlite-vec extension
                          (vec0 virtual table, cosine metric) when it is
                          installed, with a pure-Python brute-force cosine
                          fallback otherwise.
* Rank fusion          -> Reciprocal Rank Fusion (Cormack et al., 2009) as a
                          pure function with a configurable k (default 60).

After the embedding model is cached on disk the engine runs fully offline:
no servers, no daemons, no cloud calls.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "HybridSearchEngine",
    "rrf",
    "pack_vector",
    "unpack_vector",
    "cosine_similarity",
    "DEFAULT_MODEL",
    "DEFAULT_RRF_K",
]
__version__ = "0.1.0"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RRF_K = 60

# Maps a batch of texts to a batch of equal-length float vectors.
EmbedFn = Callable[[Sequence[str]], List[List[float]]]


def rrf(rankings: Sequence[Sequence[int]], k: int = DEFAULT_RRF_K) -> Dict[int, float]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

    Combines several ranked lists of document ids into one score per document:

        score(d) = sum over lists of 1 / (k + rank_i(d))

    Ranks are 1-based internally (the first entry of a list has rank 1);
    a higher fused score is better. ``k`` dampens the contribution of top
    ranks; k=60 is the value used in the original paper and works well in
    practice anywhere between 10 and 100.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return scores


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialize a float vector as little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> List[float]:
    """Deserialize little-endian float32 bytes back into a list of floats."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 for zero vectors or dimension mismatch."""
    if len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _fts_safe_query(query: str, operator: str = "OR") -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Every whitespace-separated token is wrapped in double quotes (with
    embedded quotes escaped), which neutralizes FTS5 operators such as
    NEAR, AND, OR, * and column filters. Tokens are then joined with the
    given boolean operator: OR maximizes recall so the fusion ranker sees
    candidates from both legs; AND requires every token to match. Tokens
    without any alphanumeric character are dropped.
    """
    joiner = " AND " if operator.upper() == "AND" else " OR "
    tokens = []
    for token in query.split():
        if not any(ch.isalnum() for ch in token):
            continue
        tokens.append('"%s"' % token.replace('"', '""'))
    return joiner.join(tokens)


class HybridSearchEngine:
    """Local hybrid (BM25 + vector) search engine on a single SQLite file.

    Args:
        db_path: path to the SQLite database file, or ":memory:".
        model_name: FastEmbed model used when no ``embed_fn`` is given.
        rrf_k: default k for Reciprocal Rank Fusion (overridable per call).
        embed_fn: optional custom embedding function mapping a batch of
            texts to float vectors; overrides FastEmbed entirely (useful
            for tests or for plugging in another local model).
        load_vec_extension: try to load the sqlite-vec extension so vector
            search uses vec0 KNN queries; when the extension is absent or
            cannot be loaded the engine falls back to an in-Python
            brute-force cosine scan with identical semantics.
    """

    def __init__(
        self,
        db_path: str = "search.db",
        model_name: str = DEFAULT_MODEL,
        rrf_k: int = DEFAULT_RRF_K,
        embed_fn: Optional[EmbedFn] = None,
        load_vec_extension: bool = True,
    ) -> None:
        self.db_path = db_path
        self.model_name = model_name
        self.rrf_k = rrf_k
        self._embed_fn = embed_fn
        self._model: Any = None
        self.dim: Optional[int] = None
        self._vec_table_ready = False
        self.vector_backend = "python"

        self.conn = sqlite3.connect(db_path)
        if db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        if load_vec_extension:
            self.vector_backend = self._try_load_vec_extension()

    # ------------------------------------------------------------------ #
    # Schema                                                              #
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    content   TEXT NOT NULL,
                    embedding BLOB
                )
                """
            )
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    content,
                    content='documents',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            # An external-content FTS5 table is never written directly:
            # all three triggers are required to keep the index consistent.
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS documents_fts_ai
                AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, content)
                        VALUES (new.id, new.content);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS documents_fts_au
                AFTER UPDATE OF content ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    INSERT INTO documents_fts(rowid, content)
                        VALUES (new.id, new.content);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS documents_fts_ad
                AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                END
                """
            )

    def _try_load_vec_extension(self) -> str:
        """Load sqlite-vec if present; report the active vector backend."""
        try:
            import sqlite_vec  # type: ignore

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items'"
            ).fetchone()
            if row:
                # Reopened database: reuse the existing mirror table.
                self._vec_table_ready = True
            return "sqlite-vec"
        except Exception:
            return "python"

    def _ensure_vec_table(self) -> bool:
        """Create the vec0 mirror table once the embedding dim is known."""
        if self._vec_table_ready or self.vector_backend != "sqlite-vec":
            return self._vec_table_ready
        if self.dim is None:
            return False
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0("
                "item_id INTEGER PRIMARY KEY, "
                f"embedding float[{int(self.dim)}] distance_metric=cosine)"
            )
            self._vec_table_ready = True
        except sqlite3.Error:
            # Older sqlite-vec builds without cosine support: degrade.
            self.vector_backend = "python"
            self._vec_table_ready = False
        return self._vec_table_ready

    # ------------------------------------------------------------------ #
    # Embedding                                                           #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._embed_fn is not None:
            return [[float(x) for x in vec] for vec in self._embed_fn(texts)]
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for automatic embedding. Install it "
                    "with `pip install fastembed`, or pass a custom embed_fn "
                    "to HybridSearchEngine."
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name)
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #

    def insert_documents(self, texts: Sequence[str]) -> List[int]:
        """Embed and insert a batch of documents; returns their row ids."""
        if not texts:
            return []
        vectors = self._embed(texts)
        if self.dim is None:
            self.dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != self.dim:
                raise ValueError(
                    f"Embedding dimension changed from {self.dim} to {len(vec)}; "
                    "switching models requires a fresh index with all documents "
                    "re-embedded."
                )
        self._ensure_vec_table()
        ids: List[int] = []
        with self.conn:
            for text, vec in zip(texts, vectors):
                cursor = self.conn.execute(
                    "INSERT INTO documents (content, embedding) VALUES (?, ?)",
                    (text, pack_vector(vec)),
                )
                doc_id = int(cursor.lastrowid)
                ids.append(doc_id)
                if self._vec_table_ready:
                    self.conn.execute(
                        "INSERT INTO vec_items (item_id, embedding) VALUES (?, ?)",
                        (doc_id, pack_vector(vec)),
                    )
        return ids

    def insert_document(self, text: str) -> int:
        """Embed and insert one document; returns its row id."""
        return self.insert_documents([text])[0]

    def update_document(self, doc_id: int, text: str) -> None:
        """Replace a document's content, re-embedding and re-indexing it."""
        vec = self._embed([text])[0]
        if self.dim is None:
            self.dim = len(vec)
        self._ensure_vec_table()
        with self.conn:
            self.conn.execute(
                "UPDATE documents SET content = ?, embedding = ? WHERE id = ?",
                (text, pack_vector(vec), doc_id),
            )
            if self._vec_table_ready:
                self.conn.execute(
                    "DELETE FROM vec_items WHERE item_id = ?", (doc_id,)
                )
                self.conn.execute(
                    "INSERT INTO vec_items (item_id, embedding) VALUES (?, ?)",
                    (doc_id, pack_vector(vec)),
                )

    def delete_document(self, doc_id: int) -> None:
        """Delete a document together with all of its index entries."""
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            if self._vec_table_ready:
                self.conn.execute(
                    "DELETE FROM vec_items WHERE item_id = ?", (doc_id,)
                )

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def search_fts(
        self, query: str, limit: int = 20, operator: str = "OR"
    ) -> List[int]:
        """BM25 keyword search; returns document ids, best first.

        ``operator`` is "OR" (default: recall-friendly, ranked by BM25) or
        "AND" (every token must appear).
        """
        match_expr = _fts_safe_query(query, operator=operator)
        if not match_expr:
            return []
        cursor = self.conn.execute(
            "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match_expr, limit),
        )
        return [int(row[0]) for row in cursor.fetchall()]

    def search_vector(self, query: str, limit: int = 20) -> List[int]:
        """Semantic similarity search; returns document ids, best first."""
        query_vec = self._embed([query])[0]
        if self.vector_backend == "sqlite-vec" and self._vec_table_ready:
            cursor = self.conn.execute(
                "SELECT item_id FROM vec_items WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (pack_vector(query_vec), limit),
            )
            return [int(row[0]) for row in cursor.fetchall()]
        # Brute-force fallback: cosine scan over the stored float32 blobs.
        cursor = self.conn.execute(
            "SELECT id, embedding FROM documents WHERE embedding IS NOT NULL"
        )
        scored: List[Tuple[int, float]] = [
            (int(doc_id), cosine_similarity(query_vec, unpack_vector(blob)))
            for doc_id, blob in cursor.fetchall()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:limit]]

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 5,
        k: Optional[int] = None,
        candidate_factor: int = 3,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: FTS5 BM25 + vector retrieval, fused with RRF.

        Fetches ``top_k * candidate_factor`` candidates from each leg,
        fuses the two rankings with Reciprocal Rank Fusion (k defaults to
        ``self.rrf_k``) and returns up to ``top_k`` documents as dicts with
        ``id``, ``score`` (fused RRF score), ``content`` and the per-leg
        ranks ``fts_rank`` / ``vec_rank`` (None when that leg did not
        retrieve the document). The FTS leg uses the recall-friendly OR
        operator (pass operator="AND" through search_fts for strict
        matching). Ties favor documents ranked higher by BM25.
        """
        limit = max(top_k * candidate_factor, top_k)
        fts_ids = self.search_fts(query, limit=limit)
        vec_ids = self.search_vector(query, limit=limit)
        fused = rrf([fts_ids, vec_ids], k=self.rrf_k if k is None else k)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        if not ranked:
            return []
        fts_rank = {doc_id: pos + 1 for pos, doc_id in enumerate(fts_ids)}
        vec_rank = {doc_id: pos + 1 for pos, doc_id in enumerate(vec_ids)}
        ids = [doc_id for doc_id, _ in ranked]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, content FROM documents WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        contents = {int(row[0]): row[1] for row in rows}
        return [
            {
                "id": doc_id,
                "score": score,
                "content": contents.get(doc_id, ""),
                "fts_rank": fts_rank.get(doc_id),
                "vec_rank": vec_rank.get(doc_id),
            }
            for doc_id, score in ranked
        ]

    # Backwards-compatible alias for the original draft API.
    hybrid_search = hybrid_search_rrf

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HybridSearchEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
