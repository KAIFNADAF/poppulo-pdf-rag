import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.config import (
    FAISS_INDEX_FILE,
    CHUNK_METADATA_FILE,
    CITATION_METADATA_FILE,
)
from src.models import RetrievalChunk, CitationUnit


class IndexStore:
    def __init__(
        self,
        index_file: Path = FAISS_INDEX_FILE,
        chunk_metadata_file: Path = CHUNK_METADATA_FILE,
        citation_metadata_file: Path = CITATION_METADATA_FILE,
    ) -> None:
        self.index_file = Path(index_file)
        self.chunk_metadata_file = Path(chunk_metadata_file)
        self.citation_metadata_file = Path(citation_metadata_file)

        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.citation_metadata_file.parent.mkdir(parents=True, exist_ok=True)

    def clear_all(self) -> None:
        """
        Removing any existing persisted index artifacts.

        This supports the current product behavior where indexing a new
        document fully replaces the previous active knowledge base.
        """
        for path in (
            self.index_file,
            self.chunk_metadata_file,
            self.citation_metadata_file,
        ):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                raise RuntimeError(f"Failed to remove existing artifact: {path}") from exc

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Building a FAISS inner-product index from normalized embeddings.
        """
        if embeddings.size == 0:
            raise ValueError("Cannot build FAISS index from empty embeddings.")

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array of shape (n_samples, dimension), got shape {embeddings.shape}."
            )

        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build FAISS index from zero embedding rows.")

        if embeddings.shape[1] == 0:
            raise ValueError("Cannot build FAISS index with zero embedding dimension.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_index(self, index: faiss.Index) -> None:
        faiss.write_index(index, str(self.index_file))

    def load_index(self) -> faiss.Index:
        if not self.index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_file}")
        return faiss.read_index(str(self.index_file))

    def save_chunk_metadata(self, chunks: List[RetrievalChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot save empty chunk metadata.")

        serializable_chunks = [asdict(chunk) for chunk in chunks]
        with open(self.chunk_metadata_file, "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)

    def load_chunk_metadata(self) -> List[RetrievalChunk]:
        if not self.chunk_metadata_file.exists():
            raise FileNotFoundError(
                f"Chunk metadata file not found: {self.chunk_metadata_file}"
            )

        with open(self.chunk_metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            raise ValueError(
                f"Chunk metadata file is empty: {self.chunk_metadata_file}"
            )

        return [RetrievalChunk(**item) for item in data]

    def save_citation_metadata(self, citation_units: List[CitationUnit]) -> None:
        if not citation_units:
            raise ValueError("Cannot save empty citation metadata.")

        serializable_units = [asdict(unit) for unit in citation_units]
        with open(self.citation_metadata_file, "w", encoding="utf-8") as f:
            json.dump(serializable_units, f, indent=2, ensure_ascii=False)

    def load_citation_metadata(self) -> List[CitationUnit]:
        if not self.citation_metadata_file.exists():
            raise FileNotFoundError(
                f"Citation metadata file not found: {self.citation_metadata_file}"
            )

        with open(self.citation_metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            raise ValueError(
                f"Citation metadata file is empty: {self.citation_metadata_file}"
            )

        return [CitationUnit(**item) for item in data]

    def save_all(
        self,
        embeddings: np.ndarray,
        chunks: List[RetrievalChunk],
        citation_units: List[CitationUnit],
    ) -> None:
        """
        Building and saving the FAISS index, chunk metadata, and citation metadata.

        This replaces any previously persisted artifacts.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding mismatch: got {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if not citation_units:
            raise ValueError("Cannot save index artifacts without citation metadata.")

        # Clearing older artifacts first so the new index fully replaces the old one
        self.clear_all()

        index = self.build_index(embeddings)
        self.save_index(index)
        self.save_chunk_metadata(chunks)
        self.save_citation_metadata(citation_units)

    def load_all(self) -> Tuple[faiss.Index, List[RetrievalChunk], List[CitationUnit]]:
        """
        Loading the FAISS index, chunk metadata, and citation metadata,
        then checking that they are internally consistent.
        """
        index = self.load_index()
        chunks = self.load_chunk_metadata()
        citation_units = self.load_citation_metadata()

        if index.ntotal == 0:
            raise ValueError("Loaded FAISS index is empty.")

        if not chunks:
            raise ValueError("Loaded chunk metadata is empty.")

        if not citation_units:
            raise ValueError("Loaded citation metadata is empty.")

        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/chunk mismatch: FAISS index has {index.ntotal} vectors "
                f"but chunk metadata contains {len(chunks)} chunks."
            )

        citation_ids = {unit.citation_id for unit in citation_units}

        for chunk in chunks:
            missing_ids = [cid for cid in chunk.citation_ids if cid not in citation_ids]
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' references missing citation IDs: {preview}"
                )

        return index, chunks, citation_units