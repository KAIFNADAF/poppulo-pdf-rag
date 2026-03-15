from typing import List

import faiss

from src.config import TOP_K
from src.embedder import Embedder
from src.models import RetrievalChunk, RetrievedResult


class Retriever:
    def __init__(self, embedder: Embedder, index: faiss.Index, chunks: List[RetrievalChunk]) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[RetrievedResult]:
        """
        Retrieving the top-k most relevant chunks for a query.
        """
        if not query or not query.strip():
            return []

        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[RetrievedResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            results.append(
                RetrievedResult(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    page_numbers=chunk.page_numbers,
                    text=chunk.text,
                    citation_ids=chunk.citation_ids,
                    score=float(score),
                )
            )

        return results