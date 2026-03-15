import re
from pathlib import Path
from typing import Dict, List, Any

from src.pdf_parser import parse_pdf
from src.text_cleaner import clean_text_blocks
from src.citation_builder import build_citation_units
from src.chunker import build_retrieval_chunks
from src.embedder import Embedder
from src.index_store import IndexStore
from src.retriever import Retriever
from src.prompt_builder import build_prompt
from src.generator import Generator
from src.models import CitationUnit, RetrievalChunk, RetrievedResult


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.index_store = IndexStore()
        self.generator = Generator()

        self.citation_units: List[CitationUnit] = []
        self.retrieval_chunks: List[RetrievalChunk] = []
        self.citation_lookup: Dict[str, CitationUnit] = {}
        self.retriever: Retriever | None = None
        self.active_document_name: str | None = None

    def reset_active_document(self) -> None:
        """
        Clearing the currently active indexed document from the in-memory state.

        This keeps the previous document from being available once a new one
        is selected and indexed.
        """
        self.citation_units = []
        self.retrieval_chunks = []
        self.citation_lookup = {}
        self.retriever = None
        self.active_document_name = None

    def _require_non_empty(self, items: Any, message: str) -> Any:
        """
        Checking that an intermediate pipeline step returned usable output.

        This covers:
        - None
        - Python containers like lists, dicts, and strings
        - NumPy arrays or array-like objects exposing .size
        """
        if items is None:
            raise ValueError(message)

        if hasattr(items, "size"):
            if items.size == 0:
                raise ValueError(message)
            return items

        try:
            if len(items) == 0:
                raise ValueError(message)
        except TypeError:
            pass

        return items

    def _clean_answer_text(self, answer: str, query: str) -> str:
        """
        Cleaning leftover generator artifacts that can still slip through, such as:
        - repeated 'Answer:' labels
        - echoed 'Question:' blocks
        - duplicated answer prefixes
        """
        cleaned = answer.strip()

        cleaned = re.sub(
            r"^(?:\s*(?:final answer|answer|response)\s*:\s*)+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(
            r"^\s*question\s*:\s*.*?(?:\n|$)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(
            r"\n+\s*(?:final answer|answer|response)\s*:\s*",
            "\n",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        query_escaped = re.escape(query.strip())
        cleaned = re.sub(
            rf"^\s*{query_escaped}\s*[:\-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def index_pdf(
        self,
        pdf_path: str | Path,
        document_name: str | None = None,
    ) -> Dict[str, Any]:
        """
        Parsing, cleaning, chunking, embedding, and indexing a PDF document.

        Product behavior:
        - indexing a new document replaces the current active document
        - clearing the previous in-memory retrieval state before indexing starts

        document_name:
        - using the real display name when available instead of the temp file name
        """
        pdf_path = Path(pdf_path)
        display_name = document_name or pdf_path.name

        self.reset_active_document()
        self.active_document_name = display_name

        raw_blocks = parse_pdf(pdf_path)
        self._require_non_empty(
            raw_blocks,
            "The PDF was opened successfully, but no extractable text was found. "
            "This may be a scanned/image-only PDF or a document with unsupported text encoding.",
        )

        cleaned_blocks = clean_text_blocks(raw_blocks)
        self._require_non_empty(
            cleaned_blocks,
            "Text was extracted from the PDF, but no usable content remained after cleaning. "
            "The document may contain mostly noise, headers, or very short fragments.",
        )

        citation_units = build_citation_units(cleaned_blocks)
        self._require_non_empty(
            citation_units,
            "The document text was extracted, but no valid citation units could be created. "
            "The content may be too fragmented or not paragraph-like enough for evidence extraction.",
        )

        for unit in citation_units:
            if hasattr(unit, "document_name"):
                unit.document_name = display_name

        retrieval_chunks = build_retrieval_chunks(citation_units)
        self._require_non_empty(
            retrieval_chunks,
            "Citation units were created, but no retrieval chunks could be built.",
        )

        for chunk in retrieval_chunks:
            if hasattr(chunk, "document_name"):
                chunk.document_name = display_name

        chunk_embeddings = self.embedder.embed_chunks(retrieval_chunks)
        self._require_non_empty(
            chunk_embeddings,
            "Embeddings could not be created from the document chunks.",
        )

        self.index_store.save_all(chunk_embeddings, retrieval_chunks, citation_units)

        loaded_index, loaded_chunks, loaded_citation_units = self.index_store.load_all()

        self._require_non_empty(
            loaded_chunks,
            "The index metadata was saved, but no retrieval chunks were loaded back.",
        )
        self._require_non_empty(
            loaded_citation_units,
            "The citation metadata was saved, but no citation units were loaded back.",
        )

        if loaded_index.ntotal == 0:
            raise ValueError("The FAISS index was created, but it contains no vectors.")

        for unit in loaded_citation_units:
            if hasattr(unit, "document_name"):
                unit.document_name = display_name

        for chunk in loaded_chunks:
            if hasattr(chunk, "document_name"):
                chunk.document_name = display_name

        self.citation_units = loaded_citation_units
        self.retrieval_chunks = loaded_chunks
        self.citation_lookup = {
            unit.citation_id: unit for unit in loaded_citation_units
        }

        self.retriever = Retriever(
            embedder=self.embedder,
            index=loaded_index,
            chunks=loaded_chunks,
        )

        return {
            "document_name": display_name,
            "raw_blocks": len(raw_blocks),
            "cleaned_blocks": len(cleaned_blocks),
            "citation_units": len(loaded_citation_units),
            "retrieval_chunks": len(loaded_chunks),
            "faiss_vectors": loaded_index.ntotal,
        }

    def _looks_like_heading_or_caption(self, text: str) -> bool:
        """
        Filtering weak citation units like headings, captions, references,
        and title-like fragments.
        """
        stripped = text.strip()

        if not stripped:
            return True

        lower = stripped.lower()
        words = stripped.split()

        if lower.startswith("figure ") or lower.startswith("table "):
            return True

        if stripped.endswith("..."):
            return True

        if len(words) <= 4:
            return True

        first_token = words[0]
        if first_token.replace(".", "").isdigit() and len(words) <= 7:
            return True

        if re.match(r"^\[\d+\]", stripped):
            return True

        reference_markers = [
            "arxiv",
            "doi",
            "proceedings of",
            "conference on",
            "vol.",
            "pp.",
            "pages ",
            "et al.",
        ]
        if sum(marker in lower for marker in reference_markers) >= 2:
            return True

        if stripped.count(",") >= 5 and any(ch.isdigit() for ch in stripped):
            return True

        if lower in {"references", "acknowledgements", "bibliography"}:
            return True

        return False

    def _is_good_supporting_citation(self, unit: CitationUnit) -> bool:
        """
        Keeping explanatory paragraph-like units and skipping weaker display-only ones.

        This version stays softer than the earlier filter:
        - allowing shorter but still meaningful passages
        - not rejecting text just because it starts lowercase
        """
        text = unit.text.strip()
        lower = text.lower()

        if not text:
            return False

        if len(text) < 45:
            return False

        if self._looks_like_heading_or_caption(text):
            return False

        if "python " in lower or "torchrun" in lower or "--" in text:
            return False

        year_matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if len(year_matches) >= 2 and text.count(",") >= 4 and "." in text[:100]:
            return False

        numbers = re.findall(r"\d+\.\d+|\d+", text)
        if len(numbers) >= 7 and len(text.split()) <= 18:
            return False

        words = text.split()
        numeric_tokens = sum(
            1 for w in words if re.fullmatch(r"\d+(?:\.\d+)?", w)
        )
        if len(words) > 0 and numeric_tokens / len(words) > 0.5:
            return False

        boilerplate_markers = [
            "permission to reproduce",
            "copyright",
            "all rights reserved",
        ]
        if any(marker in lower for marker in boilerplate_markers):
            return False

        footer_markers = [
            "conference on neural information processing systems",
            "long beach, ca, usa",
            "equal contribution",
            "work performed while at",
            "31st conference on",
        ]
        if any(marker in lower for marker in footer_markers):
            return False

        alpha_chars = sum(ch.isalpha() for ch in text)
        digit_chars = sum(ch.isdigit() for ch in text)
        if digit_chars > alpha_chars:
            return False

        if not any(p in text for p in [".", "!", "?"]):
            return False

        return True

    def _normalize_numbers(self, text: str) -> List[str]:
        """
        Extracting normalized numeric strings from text.

        This keeps commas and decimals normalized so values like 42,600
        and 42600 still match.
        """
        raw_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?%?", text)
        normalized = []
        for num in raw_numbers:
            cleaned = num.replace(",", "").strip()
            normalized.append(cleaned)
        return normalized

    def _classify_query_type(self, query: str) -> str:
        """
        Classifying the query in a lightweight way for support scoring.

        Returns one of: numeric, yesno, uncertainty, summary, factual
        """
        query_lower = query.lower().strip()

        if re.search(
            r"\b(how many|how much|what percentage|what percent|what date|how long|how often|how large|how big|how small)\b",
            query_lower,
        ):
            return "numeric"

        if any(
            marker in query_lower
            for marker in [
                "not specified",
                "not confirmed",
                "not provided",
                "not mentioned",
                "unknown",
                "unclear",
                "exactly",
                "explicitly",
                "specifically",
                "stated",
                "described",
                "mentioned",
                "provided",
                "confirmed",
            ]
        ):
            return "uncertainty"

        if re.search(r"^(did|does|is|are|was|were|can|could|has|have)\b", query_lower):
            return "yesno"

        if any(
            marker in query_lower
            for marker in [
                "summarize",
                "summary",
                "overview",
                "main idea",
                "main goal",
                "what is this document about",
            ]
        ):
            return "summary"

        return "factual"

    def _answer_contains_negation_or_uncertainty(self, answer: str) -> bool:
        answer_lower = answer.lower()
        markers = [
            "not ",
            "no ",
            "unknown",
            "unclear",
            "not specified",
            "not confirmed",
            "could not",
            "cannot",
            "unable to",
            "approximate",
            "approximately",
            "still unknown",
            "not provided",
            "not discussed",
            "not mentioned",
        ]
        return any(marker in answer_lower for marker in markers)

    def _extract_answer_phrases(self, answer: str) -> List[str]:
        """
        Extracting longer semantic phrases from the answer for
        answer-to-evidence grounding.
        """
        phrases = [
            phrase.strip().lower()
            for phrase in re.split(r"[.;:\n]", answer)
            if len(phrase.strip().split()) >= 4
        ]
        return [p for p in phrases if len(p) >= 20]

    def _extract_query_terms(self, text: str, min_len: int = 3) -> set[str]:
        stop_terms = {
            "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
            "does", "did", "is", "are", "was", "were", "can", "could", "has", "have",
            "paper", "document", "used", "using", "into", "from", "with", "that",
            "this", "these", "those", "their", "about", "there", "would", "should",
            "during", "after", "before", "under", "over"
        }
        return {
            term.lower()
            for term in re.findall(r"\b\w+\b", text)
            if len(term) >= min_len and term.lower() not in stop_terms
        }

    def _answer_polarity(self, answer: str) -> str:
        """
        Detecting the rough polarity for yes/no style answers.

        Returns: yes / no / unknown
        """
        lower = answer.lower().strip()

        no_markers = [
            "no,",
            "no ",
            "does not",
            "did not",
            "is not",
            "are not",
            "was not",
            "were not",
            "cannot",
            "can't",
            "without ",
            "instead ",
            "rather than ",
        ]
        yes_markers = [
            "yes,",
            "yes ",
            "does ",
            "did ",
            "is ",
            "are ",
            "was ",
            "were ",
            "uses ",
            "used ",
            "includes ",
            "contains ",
        ]

        if any(marker in lower for marker in no_markers):
            return "no"
        if any(marker in lower for marker in yes_markers):
            return "yes"
        return "unknown"

    def get_supporting_citations(
        self,
        results: List[RetrievedResult],
        query: str,
        answer: str,
        max_citations: int = 6,
    ) -> List[CitationUnit]:
        """
        Mapping retrieved chunk citation IDs back to citation units and ranking
        them for display quality using both the query and the generated answer.
        """
        seen = set()
        candidate_units: List[tuple[float, CitationUnit]] = []

        query_lower = query.lower()
        query_type = self._classify_query_type(query)

        query_terms = self._extract_query_terms(query, min_len=3)
        answer_terms = self._extract_query_terms(answer, min_len=3)

        answer_phrases = self._extract_answer_phrases(answer)
        query_numbers = self._normalize_numbers(query)
        answer_numbers = self._normalize_numbers(answer)

        broad_summary_markers = [
            "what is this document about",
            "summarize",
            "summary",
            "main idea",
            "overview",
            "main goal",
        ]
        is_broad_summary = any(marker in query_lower for marker in broad_summary_markers)

        uncertainty_markers = [
            "unknown",
            "not specified",
            "not confirmed",
            "could not",
            "cannot",
            "unable to",
            "approximate",
            "approximately",
            "not provided",
            "not discussed",
            "not mentioned",
            "still unknown",
        ]

        negation_markers = [
            "not ",
            "no ",
            "without ",
            "instead ",
            "rather than ",
        ]

        for result in results:
            for citation_id in result.citation_ids:
                if citation_id in seen:
                    continue

                unit = self.citation_lookup.get(citation_id)
                if unit is None:
                    continue

                seen.add(citation_id)

                if not self._is_good_supporting_citation(unit):
                    continue

                if self.active_document_name and hasattr(unit, "document_name"):
                    unit.document_name = self.active_document_name

                text = unit.text.strip()
                text_lower = text.lower()
                score = 0.0

                # Scoring overlap against the query, answer, and retrieved signal
                score += sum(term in text_lower for term in query_terms) * 2.0
                score += sum(term in text_lower for term in answer_terms) * 1.5
                score += sum(num in text.replace(",", "") for num in answer_numbers) * 10.0
                score += sum(num in text.replace(",", "") for num in query_numbers) * 5.0
                score += sum(phrase in text_lower for phrase in answer_phrases) * 8.0

                if query_type == "uncertainty" or self._answer_contains_negation_or_uncertainty(answer):
                    score += sum(marker in text_lower for marker in uncertainty_markers) * 6.0
                    score += sum(marker in text_lower for marker in negation_markers) * 3.0

                if is_broad_summary:
                    if unit.page_number == 1:
                        score += 8.0
                    elif unit.page_number == 2:
                        score += 6.0
                    elif unit.page_number <= 4:
                        score += 2.0

                if result.score is not None:
                    score += float(result.score) * 10.0

                if not text.endswith((".", "!", "?")):
                    score -= 2.0

                if is_broad_summary and "table " in text_lower:
                    score -= 4.0

                if query_type == "numeric" and answer_numbers:
                    if not any(num in text.replace(",", "") for num in answer_numbers):
                        score -= 6.0

                candidate_units.append((score, unit))

        ranked_units = [
            unit
            for _, unit in sorted(candidate_units, key=lambda x: x[0], reverse=True)
        ]

        return ranked_units[:max_citations]

    def compute_support_strength(
        self,
        results: List[RetrievedResult],
        supporting_citations: List[CitationUnit],
        answer: str,
        query: str,
    ) -> str:
        """
        Estimating support strength using:
        - explicit refusal / unsupported-answer detection
        - query-type-aware evidence checks
        - weighted top-6 evidence
        - stricter numeric handling
        - softer factual high-support thresholds
        - basic yes/no polarity handling

        Returns: High / Medium / Low
        """
        answer_lower = answer.lower()
        query_type = self._classify_query_type(query)

        refusal_markers = [
            "i cannot find sufficient evidence",
            "not mentioned",
            "not provided",
            "not discussed",
            "not described",
            "not specified",
            "not in the provided context",
            "not in the retrieved context",
            "insufficient evidence",
        ]
        if any(marker in answer_lower for marker in refusal_markers):
            return "Low"

        if not supporting_citations:
            return "Low"

        scored_citations = supporting_citations[:6]
        weights = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6]

        weighted_evidence_parts = []
        weighted_topic_overlap = 0.0
        weighted_strong_passages = 0.0
        weighted_uncertainty_overlap = 0.0
        weighted_negation_overlap = 0.0

        query_terms = self._extract_query_terms(query, min_len=4)

        uncertainty_markers = [
            "unknown",
            "not specified",
            "not confirmed",
            "could not",
            "cannot",
            "unable to",
            "approximate",
            "approximately",
            "not provided",
            "not discussed",
            "not mentioned",
            "still unknown",
        ]

        negation_markers = [
            "not ",
            "no ",
            "without ",
            "instead ",
            "rather than ",
        ]

        for idx, citation in enumerate(scored_citations):
            weight = weights[idx] if idx < len(weights) else 0.5
            text = citation.text.lower()
            weighted_evidence_parts.append(text)

            if len(citation.text.split()) >= 10 and citation.text.endswith((".", "!", "?")):
                weighted_strong_passages += weight

            weighted_topic_overlap += sum(term in text for term in query_terms) * weight
            weighted_uncertainty_overlap += sum(marker in text for marker in uncertainty_markers) * weight
            weighted_negation_overlap += sum(marker in text for marker in negation_markers) * weight

        evidence_text = " ".join(weighted_evidence_parts)

        if weighted_topic_overlap == 0:
            return "Low"

        answer_numbers = self._normalize_numbers(answer)
        query_numbers = self._normalize_numbers(query)

        answer_number_overlap = sum(
            num in evidence_text.replace(",", "") for num in answer_numbers
        )
        query_number_overlap = sum(
            num in evidence_text.replace(",", "") for num in query_numbers
        )

        answer_terms = self._extract_query_terms(answer, min_len=4)
        answer_term_overlap = sum(term in evidence_text for term in answer_terms)

        answer_phrases = self._extract_answer_phrases(answer)
        answer_phrase_overlap = sum(
            phrase in evidence_text for phrase in answer_phrases
        )

        scored_results = [r.score for r in results[:6] if r.score is not None]
        avg_score = sum(scored_results) / len(scored_results) if scored_results else 0.0

        answer_has_negation = self._answer_contains_negation_or_uncertainty(answer)
        answer_polarity = self._answer_polarity(answer)

        # Keeping numeric questions stricter because number alignment can fail easily
        if query_type == "numeric" or answer_numbers:
            if (
                answer_number_overlap >= 1
                and weighted_strong_passages >= 1.2
                and weighted_topic_overlap >= 1.2
                and (answer_term_overlap >= 1 or answer_phrase_overlap >= 1)
            ):
                return "High"

            if (
                (answer_number_overlap >= 1 and weighted_topic_overlap >= 1.0)
                or (query_number_overlap >= 1 and weighted_strong_passages >= 1.0)
            ):
                return "Medium"

            return "Low"

        # Keeping yes/no questions from reaching High too easily
        if query_type == "yesno":
            if answer_polarity == "no":
                if (
                    weighted_negation_overlap >= 0.6
                    and weighted_strong_passages >= 1.0
                    and weighted_topic_overlap >= 1.0
                    and (answer_term_overlap >= 1 or answer_phrase_overlap >= 1)
                ):
                    return "High"
                if weighted_topic_overlap >= 1.0 and weighted_strong_passages >= 1.0:
                    return "Medium"
                return "Low"

            if answer_polarity == "yes":
                if (
                    weighted_strong_passages >= 1.6
                    and weighted_topic_overlap >= 1.5
                    and (answer_term_overlap >= 1 or answer_phrase_overlap >= 1)
                ):
                    return "High"
                if weighted_strong_passages >= 1.0 and weighted_topic_overlap >= 1.0:
                    return "Medium"
                return "Low"

            return "Low"

        # Handling uncertainty / not-stated / exact-confirmation style questions
        if query_type == "uncertainty" or answer_has_negation:
            if (
                weighted_uncertainty_overlap >= 1.0
                and weighted_strong_passages >= 1.0
                and weighted_topic_overlap >= 1.0
                and (answer_term_overlap >= 1 or answer_phrase_overlap >= 1)
            ):
                return "High"

            if (
                (weighted_uncertainty_overlap >= 0.5 or weighted_negation_overlap >= 0.5)
                and weighted_topic_overlap >= 1.0
            ):
                return "Medium"

            return "Low"

        # Requiring broader support for summary-style answers
        if query_type == "summary":
            if (
                weighted_strong_passages >= 2.2
                and weighted_topic_overlap >= 2.2
                and (answer_term_overlap >= 2 or answer_phrase_overlap >= 1)
            ):
                return "High"

            if weighted_strong_passages >= 1.2 and weighted_topic_overlap >= 1.2:
                return "Medium"

            return "Low"

        # Keeping the default factual High threshold harder so synthesis answers
        # do not get over-labeled as strongly supported
        if (
            weighted_strong_passages >= 2.2
            and weighted_topic_overlap >= 2.2
            and answer_phrase_overlap >= 1
        ):
            return "High"

        if (
            weighted_strong_passages >= 1.0
            and weighted_topic_overlap >= 1.0
            and (answer_term_overlap >= 1 or avg_score >= 0.22)
        ):
            return "Medium"

        return "Low"

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Retrieving relevant chunks and generating a grounded answer.
        """
        if self.retriever is None:
            raise RuntimeError("No indexed document found. Please index a PDF first.")

        if not query or not query.strip():
            raise ValueError("Please enter a non-empty question.")

        results = self.retriever.retrieve(query)

        if self.active_document_name:
            for item in results:
                if hasattr(item, "document_name"):
                    item.document_name = self.active_document_name

        if not results:
            fallback_answer = "I cannot find sufficient evidence in the uploaded documents."
            return {
                "query": query,
                "answer": fallback_answer,
                "retrieved_results": [],
                "supporting_citations": [],
                "support_strength": "Low",
                "prompt": None,
            }

        prompt = build_prompt(query, results)
        raw_answer = self.generator.generate(prompt)
        answer = self._clean_answer_text(raw_answer, query)

        supporting_citations = self.get_supporting_citations(
            results,
            query=query,
            answer=answer,
        )

        support_strength = self.compute_support_strength(
            results=results,
            supporting_citations=supporting_citations,
            answer=answer,
            query=query,
        )

        return {
            "query": query,
            "answer": answer,
            "retrieved_results": results,
            "supporting_citations": supporting_citations,
            "support_strength": support_strength,
            "prompt": prompt,
        }

    def healthcheck(self) -> bool:
        """
        Checking whether the configured generator provider is reachable.
        """
        return self.generator.healthcheck()