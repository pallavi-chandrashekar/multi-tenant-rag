from __future__ import annotations

import re
import uuid
import io
from typing import TYPE_CHECKING

# Heavy / optional dependencies (fastapi, pgvector via models, pypdf) are
# imported lazily inside the functions that need them so that this module --
# and the pure ``smart_split_text`` chunker -- stays importable for unit tests
# and tooling with only the standard library available.
if TYPE_CHECKING:
    from fastapi import UploadFile
    from sqlalchemy.orm import Session

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

async def process_document(file: "UploadFile", tenant_id: str, db: "Session"):
    from backend.models import Document, Chunk
    from backend.services.embeddings import get_embeddings
    from pypdf import PdfReader  # Requires 'pypdf' in requirements.txt

    # 1. Read Content based on File Type
    content = await file.read()
    filename = file.filename.lower()
    text_content = ""

    try:
        if filename.endswith(".pdf"):
            # PDF Parsing Logic
            pdf_stream = io.BytesIO(content)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        else:
            # Fallback for TXT/MD
            text_content = content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Parsing Error: {e}")
        # Try a safe fallback decode if PDF fails or if it's a text file
        text_content = content.decode("utf-8", errors="ignore")

    if not text_content.strip():
        # Avoid crashing, just return empty doc_id or raise explicit error
        raise Exception("File is empty or could not be parsed.")

    # 2. Create Document Record (Security: Linked to Tenant)
    doc_id = uuid.uuid4()
    doc_record = Document(
        id=doc_id,
        filename=file.filename,
        tenant_id=tenant_id 
    )
    db.add(doc_record)
    db.flush() 

    # 3. Smart Chunking
    text_chunks = smart_split_text(text_content)
    
    # 4. Generate Embeddings (Batch Process)
    vectors = get_embeddings(text_chunks)

    # 5. Save Chunks (Security: Linked to Tenant)
    db_chunks = []
    for i, text_segment in enumerate(text_chunks):
        db_chunks.append(Chunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            tenant_id=tenant_id,  # STRICT TENANT ISOLATION
            content=text_segment,
            embedding=vectors[i], 
            metadata_={"source": file.filename, "chunk_index": i}
        ))

    db.add_all(db_chunks)
    db.commit()
    
    return str(doc_id)

def delete_tenant_data(tenant_id: str, db: "Session"):
    """
    Security: Only deletes data belonging to the specific tenant_id.
    """
    from backend.models import Document

    db.query(Document).filter(Document.tenant_id == tenant_id).delete()
    db.commit()