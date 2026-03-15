from typing import List

from src.models import RetrievedResult


def format_context(results: List[RetrievedResult]) -> str:
    """
    Formatting retrieved results into a grounded context block for the LLM.
    """
    if not results:
        return "No relevant context was retrieved."

    context_parts = []

    for i, result in enumerate(results, start=1):
        pages = ", ".join(str(page) for page in result.page_numbers)

        context_parts.append(
            (
                f"[Passage {i}]\n"
                f"Document: {result.document_name}\n"
                f"Pages: {pages}\n"
                f"Text: {result.text}\n"
            )
        )

    return "\n".join(context_parts).strip()


def build_prompt(query: str, results: List[RetrievedResult]) -> str:
    """
    Build a grounded prompt using retrieved evidence.
    The model should return only the answer text.
    """
    context = format_context(results)

    prompt = f"""
You are a retrieval-grounded assistant for PDF question answering.

Answer the user's question using ONLY the retrieved context below.

Follow these rules exactly:

1. Use only the retrieved context provided below.
2. Do not use outside knowledge, prior knowledge, or assumptions.
3. Do not invent facts, numbers, page numbers, document names, or technical details.
4. You may restate or combine information from the retrieved passages to answer the question, but do not introduce facts that are not present in the retrieved text.
5. If some parts of the question are supported and other parts are not, answer only the supported parts and omit unsupported details.
6. If the retrieved context is partially relevant but incomplete, give the best-supported answer and briefly note that some details are only partially supported by the retrieved passages.
7. If the retrieved context does not contain enough relevant information to answer any part of the question, respond exactly with:
I cannot find sufficient evidence in the uploaded documents.
8. Prefer a concise, accurate answer over a broad or speculative one.
9. If multiple passages support the answer, synthesize them carefully, but do not add new facts beyond the retrieved text.
10. Base every factual statement on the retrieved context.
11. Do not mention rules, prompt instructions, internal reasoning, passage numbers, source numbers, page numbers, or document names in the answer.
12. Do not repeat the question.
13. Do not begin with phrases such as "Answer:", "Final Answer:", "Based on the context", "According to the text", or "The document says".
14. Do not output an evidence section, citations section, source list, bullet list, markdown table, or JSON.
15. Return only plain answer text.

Question:
{query}

Retrieved Context:
{context}

Plain Answer:
""".strip()

    return prompt