import numpy as np
from src.embedding.base import EmbeddingClient
from src.vectorstore.base import VectorStore, Hit
from src.retrieval.reranker import CrossEncoderReranker

class Retriever:
    def __init__(self, embedder: EmbeddingClient, store: VectorStore,
                 reranker: CrossEncoderReranker | None = None,
                 k: int = 5, candidate_k: int = 20):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.k = k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, filter: dict | None = None) -> list[Hit]:
        vec = np.array(self.embedder.embed([query])[0])
        pool_size = self.candidate_k if self.reranker else self.k
        hits = self.store.query(vec, k=pool_size, filter=filter)
        if self.reranker:
            hits = self.reranker.rerank(query, hits, top_k=self.k)
        return hits
