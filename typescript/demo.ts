/**
 * Runnable demo: npm run demo
 *
 * Builds an in-memory index over five sample documents and runs a hybrid
 * query through both retrieval legs and the RRF fusion ranker. The first
 * run downloads the embedding model into ./local_cache; later runs are
 * fully offline.
 */
import { HybridSearchEngine } from "./hybrid_search.js";

const DOCS = [
  "SQLite is a lightweight C-language library that implements a SQL database engine.",
  "Vector databases allow high-dimensional embeddings similarity search.",
  "Reciprocal Rank Fusion fuses search rankings from multiple retrieval algorithms.",
  "FastEmbed is a lightweight and ultra-fast Python library for embeddings generation.",
  "Local-first software preserves user privacy by keeping data on the device.",
];

async function main(): Promise<void> {
  const engine = new HybridSearchEngine({ dbPath: ":memory:" });
  await engine.insertDocuments(DOCS);

  const query = "fast local vector search";
  console.log(`Query: '${query}'`);
  console.log(`Vector backend: ${engine.vectorBackend}\n`);
  for (const hit of await engine.hybridSearchRrf(query, 3)) {
    const fts = hit.ftsRank ?? "-";
    const vec = hit.vecRank ?? "-";
    console.log(
      `[${hit.score.toFixed(5)}] #${hit.id} (fts:${fts} vec:${vec}): ${hit.content}`
    );
  }
  engine.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
