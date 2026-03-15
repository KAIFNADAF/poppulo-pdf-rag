from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EMBEDDING_MODEL_NAME
from src.models import RetrievalChunk


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embedding a list of texts into a dense numpy array.
        """
        if not texts:
            return np.array([], dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_chunks(self, chunks: List[RetrievalChunk]) -> np.ndarray:
        """
        Embedding retrieval chunks.
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embedding a single query and return shape (1, dim).
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")