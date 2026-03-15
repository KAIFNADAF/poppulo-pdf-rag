from typing import List

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.models import CitationUnit, RetrievalChunk


def chunk_text_by_words(text: str) -> List[str]:
    """
    Splitting text into words.
    """
    return text.split()


def build_retrieval_chunks(
    citation_units: List[CitationUnit],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[RetrievalChunk]:
    """
    Building retrieval chunks from citation units.

    Chunks are formed by grouping adjacent citation units until the
    running word count reaches chunk_size. Overlap is then applied
    by carrying forward the trailing units whose combined word count
    stays within chunk_overlap.
    """
    if not citation_units:
        return []

    chunks: List[RetrievalChunk] = []
    current_units: List[CitationUnit] = []
    current_word_count = 0
    chunk_index = 0

    for unit in citation_units:
        unit_word_count = len(chunk_text_by_words(unit.text))

        current_units.append(unit)
        current_word_count += unit_word_count

        if current_word_count >= chunk_size:
            chunk_text = " ".join(u.text for u in current_units).strip()
            page_numbers = sorted({u.page_number for u in current_units})
            citation_ids = [u.citation_id for u in current_units]
            document_name = current_units[0].document_name

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"{document_name}_chunk_{chunk_index}",
                    document_name=document_name,
                    page_numbers=page_numbers,
                    text=chunk_text,
                    citation_ids=citation_ids,
                )
            )

            chunk_index += 1

            # Building the overlap window from the trailing units
            overlap_units: List[CitationUnit] = []
            overlap_word_count = 0

            for prev_unit in reversed(current_units):
                prev_count = len(chunk_text_by_words(prev_unit.text))
                if overlap_word_count + prev_count > chunk_overlap and overlap_units:
                    break
                overlap_units.insert(0, prev_unit)
                overlap_word_count += prev_count

            current_units = overlap_units
            current_word_count = overlap_word_count

    # Flushing any remaining units into the final chunk
    if current_units:
        chunk_text = " ".join(u.text for u in current_units).strip()
        page_numbers = sorted({u.page_number for u in current_units})
        citation_ids = [u.citation_id for u in current_units]
        document_name = current_units[0].document_name

        chunks.append(
            RetrievalChunk(
                chunk_id=f"{document_name}_chunk_{chunk_index}",
                document_name=document_name,
                page_numbers=page_numbers,
                text=chunk_text,
                citation_ids=citation_ids,
            )
        )

    return chunks