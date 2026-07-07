from dataclasses import replace
from sentence_transformers import CrossEncoder
from src.vectorstore.base import Hit

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.document) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        return [replace(hits[i], score=float(scores[i])) for i in order[:top_k]]
