"""
DriftShield Semantic Chunker Module.

Processes raw clinical guidelines and breaks them into boundary-aware semantic chunks.
Prevents content truncation by respecting section headers and paragraph structures.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

SECTION_HEADERS: List[str] = [
    r"^(recommendation[s]?:?)",
    r"^(evidence summary:?)",
    r"^(population:?)",
    r"^(benefits:?)",
    r"^(harms:?)",
    r"^(rationale:?)",
    r"^(implementation:?)",
    r"^(other considerations:?)",
    r"^\d+\.\s+",
    r"^[A-Z][A-Z\s]+:",
]

@dataclass
class Chunk:
    """Dataclass holding chunked guideline information.

    Attributes:
        text: Segmented text content.
        token_count: Length of chunk in tokens.
        source_doc_id: Identifier of the source guideline document.
        chunk_index: Chronological index of chunk in document.
        domain: Medical specialty domain.
        year: Year of guidelines publish.
        source_name: Organization name.
    """
    text: str
    token_count: int
    source_doc_id: str
    chunk_index: int
    domain: str
    year: int
    source_name: str


class SemanticChunker:
    """Boundary-aware semantic chunker for structured clinical guidelines.

    Splits text on section boundaries and limits chunk sizes to a maximum token length.
    """
    MAX_TOKENS: int = 512
    MIN_TOKENS: int = 30
    OVERLAP_TOKENS: int = 50

    def __init__(self, tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1") -> None:
        """Initializes the SemanticChunker.

        Args:
            tokenizer_name: HuggingFace tokenizer reference.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the given text string.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _is_section_boundary(self, line: str) -> bool:
        """Determines if a line indicates a section boundary.

        Args:
            line: Text line string.

        Returns:
            True if line matches section header patterns, False otherwise.
        """
        line_lower = line.strip().lower()
        return any(re.match(pattern, line_lower, re.IGNORECASE) for pattern in SECTION_HEADERS)

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunks a single guideline document based on section and paragraph boundaries.

        Args:
            text: Raw document string content.
            metadata: Associated metadata dictionary.

        Returns:
            List of generated Chunk instances.
        """
        lines = text.split("\n")
        sections: List[str] = []
        current_section_lines: List[str] = []

        for line in lines:
            if self._is_section_boundary(line) and current_section_lines:
                sections.append("\n".join(current_section_lines).strip())
                current_section_lines = [line]
            else:
                current_section_lines.append(line)
        if current_section_lines:
            sections.append("\n".join(current_section_lines).strip())

        chunks: List[Chunk] = []
        chunk_idx: int = 0
        for section in sections:
            if not section.strip():
                continue
            token_count = self._count_tokens(section)
            if token_count <= self.MAX_TOKENS:
                if token_count >= self.MIN_TOKENS:
                    chunks.append(Chunk(
                        text=section,
                        token_count=token_count,
                        source_doc_id=metadata["id"],
                        chunk_index=chunk_idx,
                        domain=metadata["domain"],
                        year=metadata["year"],
                        source_name=metadata["source_name"],
                    ))
                    chunk_idx += 1
            else:
                sentences = re.split(r"(?<=[.!?])\s+", section)
                current_chunk_sents: List[str] = []
                current_count = 0
                for sent in sentences:
                    sent_tokens = self._count_tokens(sent)
                    if current_count + sent_tokens > self.MAX_TOKENS and current_chunk_sents:
                        chunk_text = " ".join(current_chunk_sents)
                        if self._count_tokens(chunk_text) >= self.MIN_TOKENS:
                            chunks.append(Chunk(
                                text=chunk_text,
                                token_count=self._count_tokens(chunk_text),
                                source_doc_id=metadata["id"],
                                chunk_index=chunk_idx,
                                domain=metadata["domain"],
                                year=metadata["year"],
                                source_name=metadata["source_name"],
                            ))
                            chunk_idx += 1
                        current_chunk_sents = current_chunk_sents[-2:] if self.OVERLAP_TOKENS > 0 else []
                        current_count = sum(self._count_tokens(s) for s in current_chunk_sents)
                    current_chunk_sents.append(sent)
                    current_count += sent_tokens
                if current_chunk_sents:
                    chunk_text = " ".join(current_chunk_sents)
                    if self._count_tokens(chunk_text) >= self.MIN_TOKENS:
                        chunks.append(Chunk(
                            text=chunk_text,
                            token_count=self._count_tokens(chunk_text),
                            source_doc_id=metadata["id"],
                            chunk_index=chunk_idx,
                            domain=metadata["domain"],
                            year=metadata["year"],
                            source_name=metadata["source_name"],
                        ))
        return chunks
