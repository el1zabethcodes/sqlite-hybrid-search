/**
 * sqlite-hybrid-search — TypeScript reference implementation.
 *
 * Mirrors sqlite_hybrid_search.py: FTS5 BM25 keyword search + FastEmbed ONNX
 * vector search + Reciprocal Rank Fusion, all inside a single SQLite file.
 *
 * Install: npm install better-sqlite3 sqlite-vec fastembed
 */
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { EmbeddingModel, FlagEmbedding } from "fastembed";

export const DEFAULT_RRF_K = 60;

export type EmbedFn = (texts: string[]) => Promise<number[][]>;

export function rrf(rankings: number[][], k: number = DEFAULT_RRF_K): Map<number, number> {
  const scores = new Map<number, number>();
  for (const ranking of rankings) {
    ranking.forEach((docId, position) => {
      scores.set(docId, (scores.get(docId) ?? 0) + 1 / (k + position + 1));
    });
  }
  return scores;
}

function packVector(vec: number[] | Float32Array): Buffer {
  const f32 = Float32Array.from(vec);
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
}

function unpackVector(blob: Buffer): number[] {
  return Array.from(new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4));
}

/** vec0 requires INTEGER-bound params (primary keys, k). better-sqlite3 v13+
 *  binds JS numbers as REAL, so those params must go through BigInt. */
function intParam(n: number): bigint {
  return BigInt(Math.trunc(n));
}

function cosineSimilarity(v1: number[], v2: number[]): number {
  if (v1.length !== v2.length) return 0.0;
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < v1.length; i++) {
    dot += v1[i] * v2[i];
    normA += v1[i] * v1[i];
    normB += v2[i] * v2[i];
  }
  if (normA === 0.0 || normB === 0.0) return 0.0;
  return dot / Math.sqrt(normA * normB);
}

/** Quote every token so arbitrary user text is safe inside FTS5 MATCH. */
export function ftsSafeQuery(query: string, operator: "OR" | "AND" = "OR"): string {
  const joiner = operator === "AND" ? " AND " : " OR ";
  return query
    .split(/\s+/)
    .filter((token) => /[\p{L}\p{N}]/u.test(token))
    .map((token) => `"${token.replace(/"/g, '""')}"`)
    .join(joiner);
}

export interface HybridHit {
  id: number;
  score: number;
  content: string;
  ftsRank: number | null;
  vecRank: number | null;
}

/** Any fastembed model that ships with the package (everything but CUSTOM). */
export type StandardModel = Exclude<EmbeddingModel, EmbeddingModel.CUSTOM>;

export interface EngineOptions {
  dbPath?: string;
  model?: StandardModel;
  rrfK?: number;
  /** Custom embedding function; overrides fastembed entirely. */
  embedFn?: EmbedFn;
  /** Try to load sqlite-vec for KNN queries (default true). */
  loadVecExtension?: boolean;
}

export class HybridSearchEngine {
  /** Active vector backend; degrades to "python" if sqlite-vec fails. */
  vectorBackend: "sqlite-vec" | "python" = "python";
  readonly rrfK: number;
  private db: Database.Database;
  private embedFn: EmbedFn;
  private embedder: FlagEmbedding | null = null;
  private dim: number | null = null;
  private vecTableReady = false;

  constructor(options: EngineOptions = {}) {
    this.db = new Database(options.dbPath ?? "search.db");
    this.db.pragma("journal_mode = WAL");
    this.rrfK = options.rrfK ?? DEFAULT_RRF_K;
    this.embedFn =
      options.embedFn ??
      ((texts: string[]) =>
        this.defaultEmbed(texts, options.model ?? EmbeddingModel.BGESmallENV15));
    this.initSchema();
    if (options.loadVecExtension !== false) {
      try {
        sqliteVec.load(this.db);
        this.vectorBackend = "sqlite-vec";
        const row = this.db
          .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vec_items'")
          .get() as { name: string } | undefined;
        if (row) this.vecTableReady = true;
      } catch {
        this.vectorBackend = "python";
      }
    }
  }

  private runSql(sql: string): void {
    // Single-statement DDL/schema runner; every query path in this class
    // uses prepared statements with bound parameters for external input.
    this.db.prepare(sql).run();
  }

  private initSchema(): void {
    this.runSql(`CREATE TABLE IF NOT EXISTS documents (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        content   TEXT NOT NULL,
        embedding BLOB
      )`);
    this.runSql(`CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        content,
        content='documents',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
      )`);
    this.runSql(`CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
      END`);
    this.runSql(`CREATE TRIGGER IF NOT EXISTS documents_fts_au AFTER UPDATE OF content ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
      END`);
    this.runSql(`CREATE TRIGGER IF NOT EXISTS documents_fts_ad AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
      END`);
  }

  private async defaultEmbed(texts: string[], model: StandardModel): Promise<number[][]> {
    if (!this.embedder) {
      this.embedder = await FlagEmbedding.init({ model, showDownloadProgress: false });
    }
    const out: number[][] = [];
    for await (const batch of this.embedder.embed(texts)) {
      out.push(...batch);
    }
    return out;
  }

  private ensureVecTable(): boolean {
    if (this.vecTableReady || this.vectorBackend !== "sqlite-vec" || this.dim === null) {
      return this.vecTableReady;
    }
    // The dimension comes from the embedding model, never from user input;
    // it is validated here because DDL cannot be parameter-bound.
    const dim = Math.trunc(Number(this.dim));
    if (!Number.isSafeInteger(dim) || dim <= 0 || dim > 65536) {
      return false;
    }
    try {
      this.runSql(
        `CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
           item_id INTEGER PRIMARY KEY,
           embedding float[${dim}] distance_metric=cosine
         )`
      );
      this.vecTableReady = true;
    } catch {
      this.vectorBackend = "python";
      this.vecTableReady = false;
    }
    return this.vecTableReady;
  }

  async insertDocuments(texts: string[]): Promise<number[]> {
    if (texts.length === 0) return [];
    const vectors = await this.embedFn(texts);
    if (this.dim === null) this.dim = vectors[0].length;
    if (vectors.some((v) => v.length !== this.dim)) {
      throw new Error(
        `Embedding dimension changed from ${this.dim}; switching models requires a full re-embed.`
      );
    }
    this.ensureVecTable();
    const insertDoc = this.db.prepare(
      "INSERT INTO documents (content, embedding) VALUES (?, ?)"
    );
    const insertVec = this.db.prepare(
      "INSERT INTO vec_items (item_id, embedding) VALUES (?, ?)"
    );
    const ids: number[] = [];
    const tx = this.db.transaction((rows: Array<[string, number[]]>) => {
      for (const [text, vec] of rows) {
        const result = insertDoc.run(text, packVector(vec));
        const id = Number(result.lastInsertRowid);
        ids.push(id);
        if (this.vecTableReady) insertVec.run(intParam(id), packVector(vec));
      }
    });
    tx(texts.map((text, i) => [text, vectors[i]] as [string, number[]]));
    return ids;
  }

  async insertDocument(text: string): Promise<number> {
    return (await this.insertDocuments([text]))[0];
  }

  async updateDocument(docId: number, text: string): Promise<void> {
    const [vec] = await this.embedFn([text]);
    if (this.dim === null) this.dim = vec.length;
    this.ensureVecTable();
    const tx = this.db.transaction(() => {
      this.db
        .prepare("UPDATE documents SET content = ?, embedding = ? WHERE id = ?")
        .run(text, packVector(vec), docId);
      if (this.vecTableReady) {
        this.db.prepare("DELETE FROM vec_items WHERE item_id = ?").run(intParam(docId));
        this.db
          .prepare("INSERT INTO vec_items (item_id, embedding) VALUES (?, ?)")
          .run(intParam(docId), packVector(vec));
      }
    });
    tx();
  }

  deleteDocument(docId: number): void {
    const tx = this.db.transaction(() => {
      this.db.prepare("DELETE FROM documents WHERE id = ?").run(docId);
      if (this.vecTableReady) {
        this.db.prepare("DELETE FROM vec_items WHERE item_id = ?").run(intParam(docId));
      }
    });
    tx();
  }

  searchFts(query: string, limit = 20, operator: "OR" | "AND" = "OR"): number[] {
    const matchExpr = ftsSafeQuery(query, operator);
    if (!matchExpr) return [];
    const rows = this.db
      .prepare(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?"
      )
      .all(matchExpr, limit) as Array<{ rowid: number }>;
    return rows.map((r) => r.rowid);
  }

  async searchVector(query: string, limit = 20): Promise<number[]> {
    const [queryVec] = await this.embedFn([query]);
    if (this.vectorBackend === "sqlite-vec" && this.vecTableReady) {
      const rows = this.db
        .prepare(
          "SELECT item_id FROM vec_items WHERE embedding MATCH ? AND k = ? ORDER BY distance"
        )
        .all(packVector(queryVec), intParam(limit)) as Array<{ item_id: number }>;
      return rows.map((r) => r.item_id);
    }
    const rows = this.db
      .prepare("SELECT id, embedding FROM documents WHERE embedding IS NOT NULL")
      .all() as Array<{ id: number; embedding: Buffer }>;
    const scored = rows.map((r) => ({
      id: r.id,
      sim: cosineSimilarity(queryVec, unpackVector(r.embedding)),
    }));
    scored.sort((a, b) => b.sim - a.sim);
    return scored.slice(0, limit).map((s) => s.id);
  }

  async hybridSearchRrf(
    query: string,
    topK = 5,
    k?: number,
    candidateFactor = 3
  ): Promise<HybridHit[]> {
    const limit = Math.max(topK * candidateFactor, topK);
    const ftsIds = this.searchFts(query, limit);
    const vecIds = await this.searchVector(query, limit);
    const fused = rrf([ftsIds, vecIds], k ?? this.rrfK);
    const ranked = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK);
    if (ranked.length === 0) return [];
    const ftsRank = new Map(ftsIds.map((id, i) => [id, i + 1]));
    const vecRank = new Map(vecIds.map((id, i) => [id, i + 1]));
    const placeholders = ranked.map(() => "?").join(",");
    const rows = this.db
      .prepare(`SELECT id, content FROM documents WHERE id IN (${placeholders})`)
      .all(...ranked.map(([id]) => id)) as Array<{ id: number; content: string }>;
    const contents = new Map(rows.map((r) => [r.id, r.content]));
    return ranked.map(([id, score]) => ({
      id,
      score,
      content: contents.get(id) ?? "",
      ftsRank: ftsRank.get(id) ?? null,
      vecRank: vecRank.get(id) ?? null,
    }));
  }

  close(): void {
    this.db.close();
  }
}
