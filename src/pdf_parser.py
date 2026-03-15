from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.models import RawTextBlock


def parse_pdf(pdf_path: str | Path) -> List[RawTextBlock]:
    """
    Extract text blocks from a PDF file.

    Each block is returned as a RawTextBlock containing the document name,
    page number, block index, and the extracted text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_blocks: List[RawTextBlock] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from exc

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # PyMuPDF returns blocks roughly in the form:
        # (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")

        for block_idx, block in enumerate(blocks):
            text = block[4] if len(block) > 4 else ""

            if not text or not text.strip():
                continue

            raw_blocks.append(
                RawTextBlock(
                    document_name=pdf_path.name,
                    page_number=page_idx + 1,
                    block_index=block_idx,
                    text=text.strip(),
                )
            )

    doc.close()
    return raw_blocks


if __name__ == "__main__":
    from pathlib import Path
    from src.pipeline import RAGPipeline

    sample_path = Path("data/raw/sample.pdf")

    if sample_path.exists():
        pipeline = RAGPipeline()

        print("Checking generator provider connection...")
        print(f"Provider healthy: {pipeline.healthcheck()}")

        index_info = pipeline.index_pdf(sample_path)

        print("\nINDEX SUMMARY")
        print("=" * 80)
        for key, value in index_info.items():
            print(f"{key}: {value}")

        query = "What is the Transformer architecture?"
        result = pipeline.answer_query(query)

        print("\n" + "=" * 80)
        print("MODEL ANSWER")
        print("=" * 80)
        print(result["answer"])

        print("\n" + "=" * 80)
        print("SUPPORTING CITATIONS")
        print("=" * 80)
        for citation in result["supporting_citations"][:5]:
            print(
                f"{citation.document_name} | page {citation.page_number} | "
                f"{citation.text[:200]}..."
            )
    else:
        print("No sample PDF found at data/raw/sample.pdf")