---
name: sqlite-hybrid-search
description: Scaffold and run a zero-dependency local hybrid search engine (FTS5 BM25 + FastEmbed vector embeddings + Reciprocal Rank Fusion) inside a standard SQLite database. Use when the user asks for local or offline search, "hybrid search sqlite", RRF fusion without external databases, local RAG, desktop-app document retrieval, or replacing Pinecone/Qdrant/Elasticsearch on small-to-medium local datasets.
patterns:
  - "hybrid search"
  - "fts5"
  - "sqlite vector"
  - "local rag"
  - "rrf"
  - "локальный поиск"
  - "гибридный поиск"
---

# Skill: Local SQLite Hybrid Search

## When to activate

Activate when the user asks (in English or Russian) for:

- local or offline search: "локальный поиск", "local search", "offline retrieval", "air-gapped"
- hybrid retrieval on SQLite: "hybrid search sqlite", "BM25 + vectors", "fts5", "sqlite-vec"
- rank fusion: "RRF", "reciprocal rank fusion", "RRF поиск без внешних баз"
- local RAG, desktop-app document indexing, or retrieval without external infrastructure
- replacing Pinecone / Qdrant Cloud / Elasticsearch / Milvus for a small-to-medium local dataset

Do NOT activate for distributed clusters, multi-tenant SaaS search, corpora beyond a few
million documents, or many concurrent writers — recommend a real search engine instead.

## Architectural contract

1. **Keyword engine** — native SQLite FTS5 virtual table with external content
   (`content='documents'`), unicode61 tokenizer, BM25 ordering via `ORDER BY rank`.
2. **Vector engine** — local ONNX embeddings (FastEmbed; never PyTorch or
   `transformers` in client apps) stored as little-endian float32 blobs. When the
   `sqlite-vec` extension is available, mirror vectors into a `vec0` virtual table
   and answer with KNN MATCH queries; otherwise brute-force cosine in the host
   language. Both backends MUST return ids ordered best-first.
3. **Fusion ranker** — Reciprocal Rank Fusion over the two rankings, implemented
   as a pure function, default `k = 60`, configurable per call.

## Database schema contract (DDL)

```sql
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content   TEXT NOT NULL,
    embedding BLOB              -- little-endian float32[dim]
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    content='documents',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Created lazily once the embedding dim is known (vec backend only).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    item_id   INTEGER PRIMARY KEY,
    embedding float[384] distance_metric=cosine
);
```

An external-content FTS5 table is never written directly. All three triggers are
MANDATORY — a missing UPDATE or DELETE trigger silently corrupts the index:

```sql
CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_au AFTER UPDATE OF content ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;
```

## Function contract

| Function | Signature | Returns |
|---|---|---|
| `insert_document` | `insert_document(text: str) -> int` | new row id |
| `insert_documents` | `insert_documents(texts) -> List[int]` | row ids (one embed call per batch) |
| `update_document` | `update_document(doc_id: int, text: str) -> None` | re-embeds and re-indexes |
| `delete_document` | `delete_document(doc_id: int) -> None` | removes row and all index entries |
| `search_fts` | `search_fts(query, limit=20, operator="OR") -> List[int]` | ids, BM25 best-first |
| `search_vector` | `search_vector(query: str, limit: int = 20) -> List[int]` | ids, cosine best-first |
| `hybrid_search_rrf` | `hybrid_search_rrf(query, top_k=5, k=None) -> List[dict]` | `[{id, score, content, fts_rank, vec_rank}]` |

## Reference implementations in this repository

- Python (reference): `sqlite_hybrid_search.py`, runnable demo `index.py`, tests `tests/`
- TypeScript (reference): `typescript/hybrid_search.ts` + `typescript/demo.ts`
  (better-sqlite3 + sqlite-vec + fastembed)

## Python template

```python
from sqlite_hybrid_search import HybridSearchEngine

engine = HybridSearchEngine("app.db")          # one file: rows + FTS5 + vectors
engine.insert_documents(["first document", "second document"])

for hit in engine.hybrid_search_rrf("first", top_k=5, k=60):
    print(hit["score"], hit["content"])
```

## TypeScript template

```ts
import { HybridSearchEngine } from "./hybrid_search";

const engine = new HybridSearchEngine({ dbPath: "app.db" });
await engine.insertDocuments(["first document", "second document"]);

for (const hit of await engine.hybridSearchRrf("first", 5)) {
  console.log(hit.score, hit.content);
}
```

## Implementation rules (MUST follow)

1. Tokenizer: `unicode61 remove_diacritics 2` — multilingual-safe (en/ru/uk/...).
   For English-only corpora Porter stemming may be layered:
   `tokenize='porter unicode61 remove_diacritics 2'`.
2. External-content FTS5 tables MUST define all three triggers
   (INSERT / UPDATE / DELETE), as in the DDL above.
3. Serialize vectors as little-endian float32: Python
   `struct.pack(f"<{n}f", *vec)`, JS `Buffer.from(new Float32Array(vec).buffer)`.
4. One embedding model per index. The dimension is fixed at the first insert;
   switching models requires a full re-embed of the corpus into a fresh database.
5. Batch embeddings (`insert_documents`): one ONNX call per batch is an order of
   magnitude faster than per-document calls.
6. When the user asked for local-first, NEVER call cloud embedding APIs
   (OpenAI, Cohere, ...): use FastEmbed/ONNX or an injected local `embed_fn`.
7. User text must never be concatenated into FTS5 MATCH. Quote every
   whitespace-separated token (`"tok"`) to neutralize FTS5 operators; all other
   SQL uses parameter binding.
8. RRF: default `k = 60` (the paper value); sane range 10-100. Fetch
   `top_k * 3` candidates from each leg before fusing. The FTS leg defaults
   to the recall-friendly OR operator (BM25 ranking provides precision);
   strict AND matching is available via `operator="AND"`.
9. For file-backed databases enable WAL (`PRAGMA journal_mode=WAL`). SQLite has
   a single writer — serialize writes or use a dedicated write thread.
10. Prefer sqlite-vec when installed (`pip install sqlite-vec` / `npm i sqlite-vec`);
    the brute-force fallback is correct but O(N) per query.

## Model selection

| Model | Dim | Languages | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | en | fast, small, great default |
| `intfloat/multilingual-e5-small` | 384 | 100+ incl. ru | e5 models require `query: ` / `passage: ` prefixes on inputs |
| `BAAI/bge-small-zh-v1.5` | 512 | zh | Chinese-optimized |

Full supported list: https://qdrant.github.io/fastembed/examples/Supported-Models/

## Verification checklist before delivering

- `python -m unittest discover -s tests` — offline tests for both vector backends
- `python index.py` — end-to-end demo (first run downloads the model, then fully offline)
- Report which `vector_backend` is active and confirm hybrid results on a query
  mixing an exact keyword hit with a semantic-only hit
- TypeScript: `cd typescript && npm install && npm run demo`

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `no such module: fts5` | SQLite built without FTS5 | Use CPython (bundles FTS5) or better-sqlite3 |
| `no such module: vec0` | sqlite-vec not loaded | `pip install sqlite-vec` / `npm i sqlite-vec`, or rely on the fallback |
| Stale FTS results after DELETE/UPDATE | missing external-content triggers | add all three triggers; repair once with `INSERT INTO documents_fts(documents_fts) VALUES('rebuild')` |
| FTS5 syntax error on user input | raw query passed to MATCH | quote every token (see `_fts_safe_query`) |
| vec0 dimension mismatch | embedding model changed | re-embed the corpus into a fresh database |
