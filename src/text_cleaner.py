import re
from typing import List

from src.models import RawTextBlock


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while keeping sentence flow readable.
    """
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)  # collapse spaces/tabs
    text = re.sub(r"\n+", "\n", text)    # collapse repeated newlines
    return text.strip()


def merge_broken_lines(text: str) -> str:
    """
    Merge line breaks inside a block to produce smoother paragraph text.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines).strip()


def is_noisy_block(text: str, min_chars: int = 20) -> bool:
    """
    Simple noise filter.

    Very short or empty blocks are treated as noise.
    """
    if not text or not text.strip():
        return True

    if len(text.strip()) < min_chars:
        return True

    return False


def clean_text_block(block: RawTextBlock) -> RawTextBlock | None:
    """
    Clean a single RawTextBlock.

    Returns None if the cleaned block is too noisy to keep.
    """
    cleaned_text = normalize_whitespace(block.text)
    cleaned_text = merge_broken_lines(cleaned_text)

    if is_noisy_block(cleaned_text):
        return None

    return RawTextBlock(
        document_name=block.document_name,
        page_number=block.page_number,
        block_index=block.block_index,
        text=cleaned_text,
    )


def clean_text_blocks(blocks: List[RawTextBlock]) -> List[RawTextBlock]:
    """
    Clean a list of RawTextBlock objects and drop noisy ones.
    """
    cleaned_blocks: List[RawTextBlock] = []

    for block in blocks:
        cleaned_block = clean_text_block(block)
        if cleaned_block is not None:
            cleaned_blocks.append(cleaned_block)

    return cleaned_blocks