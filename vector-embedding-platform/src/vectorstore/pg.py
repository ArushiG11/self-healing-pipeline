import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from src.vectorstore.base import VectorStore, Hit

load_dotenv()

class PgVectorStore(VectorStore):
    def __init__(self, table: str = "chunks", dim: int = 384):
        self.table = table
        self.dim = dim
        self.conn = psycopg2.connect(os.environ["PG_CONN"])
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn, self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                  id TEXT PRIMARY KEY,
                  embedding vector({self.dim}),
                  document TEXT,
                  rating REAL,
                  asin TEXT,
                  parent_asin TEXT,
                  helpful_vote INT
                );""")

    def upsert(self, ids, vectors, documents, metadatas):
        with self.conn, self.conn.cursor() as cur:
            for i, v, d, m in zip(ids, vectors, documents, metadatas):
                cur.execute(f"""
                    INSERT INTO {self.table}
                      (id, embedding, document, rating, asin, parent_asin, helpful_vote)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      embedding = EXCLUDED.embedding,
                      document = EXCLUDED.document;""",
                    (i, v, d, m.get("rating"), m.get("asin"),
                     m.get("parent_asin"), m.get("helpful_vote")))

    def query(self, vector, k=5, filter=None):
        where, params = "", []
        if filter and "min_rating" in filter:
            where = "WHERE rating >= %s"
            params.append(filter["min_rating"])
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, document, rating, asin, parent_asin,
                       -(embedding <#> %s::vector) AS score
                FROM {self.table} {where}
                ORDER BY embedding <#> %s::vector
                LIMIT %s;""",
                [vector] + params + [vector, k])
            return [Hit(id=r[0], score=float(r[5]), document=r[1],
                        metadata={"rating": r[2], "asin": r[3], "parent_asin": r[4]})
                    for r in cur.fetchall()]

    def count(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return cur.fetchone()[0]