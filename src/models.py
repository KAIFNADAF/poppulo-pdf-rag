from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RawTextBlock:
    document_name: str
    page_number: int
    block_index: int
    text: str


@dataclass
class CitationUnit:
    citation_id: str
    document_name: str
    page_number: int
    unit_index: int
    text: str


@dataclass
class RetrievalChunk:
    chunk_id: str
    document_name: str
    page_numbers: List[int]
    text: str
    citation_ids: List[str] = field(default_factory=list)


@dataclass
class RetrievedResult:
    chunk_id: str
    document_name: str
    page_numbers: List[int]
    text: str
    citation_ids: List[str]
    score: Optional[float] = None


#@dataclass
#class GeneratedAnswer:
#    answer: str
#    citations: List[CitationUnit] = field(default_factory=list)
#    retrieved_results: List[RetrievedResult] = field(default_factory=list)
#    low_support_warning: bool = False