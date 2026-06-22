"""Sentence-aware text chunking.

Pure and dependency-free (standard library only) so it is unit-testable and
importable without the heavier ingestion dependencies (fastapi, pypdf, pgvector).
"""

import re

# Sentence terminator (., !, ?) followed by whitespace, or a hard newline break.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def smart_split_text(text: str, chunk_size: int = 500, overlap: int = 80):
    """Group whole sentences into ``chunk_size``-bounded chunks.

    Unlike a fixed character window, this never splits a word or a sentence:
    sentences are packed greedily up to ``chunk_size`` and a word-boundary
    ``overlap`` tail is carried into the next chunk so facts that straddle a
    boundary remain retrievable. Keeping tokens (e.g. "receipt", "$75") intact
    is what lets keyword search and grounded generation actually find them.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > chunk_size:
            chunks.append(current.strip())
            # Carry a word-aligned overlap tail into the next chunk.
            if overlap > 0:
                tail = current[-overlap:]
                space = tail.find(" ")
                current = tail[space + 1:] if space != -1 else ""
            else:
                current = ""
        current = f"{current} {sent}".strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks
