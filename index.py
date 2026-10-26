"""Runnable demo: python index.py

Builds an in-memory index over five sample documents and runs a hybrid
query through both retrieval legs and the RRF fusion ranker. The first run
downloads the embedding model (~30-130 MB) into the local fastembed cache;
every later run is fully offline.
"""

from sqlite_hybrid_search import HybridSearchEngine

DOCS = [
    "SQLite is a lightweight C-language library that implements a SQL database engine.",
    "Vector databases allow high-dimensional embeddings similarity search.",
    "Reciprocal Rank Fusion fuses search rankings from multiple retrieval algorithms.",
    "FastEmbed is a lightweight and ultra-fast Python library for embeddings generation.",
    "Local-first software preserves user privacy by keeping data on the device.",
]


def main() -> None:
    engine = HybridSearchEngine(":memory:")
    engine.insert_documents(DOCS)

    query = "fast local vector search"
    print(f"Query: '{query}'")
    print(f"Vector backend: {engine.vector_backend}\n")
    for hit in engine.hybrid_search_rrf(query, top_k=3):
        fts = hit["fts_rank"] if hit["fts_rank"] is not None else "-"
        vec = hit["vec_rank"] if hit["vec_rank"] is not None else "-"
        print(f"[{hit['score']:.5f}] #{hit['id']} (fts:{fts} vec:{vec}): {hit['content']}")


if __name__ == "__main__":
    main()
