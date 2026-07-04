from sentence_transformers import SentenceTransformer
from src.embedding.base import EmbeddingClient

class BgeEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()  # 384