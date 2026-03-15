import re
from typing import List

from src.models import RawTextBlock, CitationUnit


def _looks_like_reference_line(text: str) -> bool:
    """
    Detecting bibliography-style reference lines.
    """
    lower = text.lower()

    if re.match(r"^\[\d+\]", text):
        return True

    markers = [
        "doi",
        "arxiv",
        "conference on",
        "proceedings of",
        "vol.",
        "pp.",
        "et al.",
    ]

    if sum(marker in lower for marker in markers) >= 2:
        return True

    return False


def _looks_like_caption(text: str) -> bool:
    """
    Detecting figure or table captions.
    """
    lower = text.lower()

    if lower.startswith("figure "):
        return True

    if lower.startswith("table "):
        return True

    return False


def _looks_like_fragment(text: str) -> bool:
    """
    Rejecting short fragments and broken lines that are unlikely
    to form useful citation units.
    """
    words = text.split()

    if len(words) < 8:
        return True

    if text.endswith("..."):
        return True

    if not any(p in text for p in [".", "!", "?"]):
        return True

    return False


def build_citation_units(blocks: List[RawTextBlock]) -> List[CitationUnit]:
    """
    Converting cleaned text blocks into citation units.

    Each citation unit represents a paragraph-like segment tied
    to the document name and page number so it can be displayed
    later as evidence.
    """
    citation_units: List[CitationUnit] = []

    bibliography_mode = False

    for unit_index, block in enumerate(blocks):

        text = block.text.strip()

        if not text:
            continue

        lower = text.lower()

        # Detecting the start of a bibliography section
        if lower in {"references", "bibliography"}:
            bibliography_mode = True
            continue

        # Skipping everything once we enter the references section
        if bibliography_mode:
            continue

        # Skipping figure or table captions
        if _looks_like_caption(text):
            continue

        # Skipping bibliography-style reference lines
        if _looks_like_reference_line(text):
            continue

        # Skipping short fragments or broken text blocks
        if _looks_like_fragment(text):
            continue

        citation_id = (
            f"{block.document_name}_p{block.page_number}_b{block.block_index}_u{unit_index}"
        )

        citation_units.append(
            CitationUnit(
                citation_id=citation_id,
                document_name=block.document_name,
                page_number=block.page_number,
                unit_index=unit_index,
                text=text,
            )
        )

    return citation_units