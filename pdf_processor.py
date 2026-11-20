"""
============================================================
pdf_processor.py — PDF Ingestion & Text Chunking
============================================================
PURPOSE:
    1. Opens the PDF file and extracts text from every page
    2. Splits that text into smaller "chunks" with overlap
    3. Each chunk remembers which page it came from

WHY CHUNKING?
    Imagine you have a 50-page textbook and someone asks you
    a question. You don't read all 50 pages — you flip to the
    relevant section. That's what chunking does for our AI:
    it breaks the big document into small, searchable pieces.

WHY OVERLAP?
    Imagine a sentence is split right in the middle between
    two chunks. The overlap ensures that sentence appears in
    BOTH chunks, so we don't lose information at boundaries.
============================================================
"""

import pymupdf  # PyMuPDF library for reading PDFs
import logging

from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# ----------------------------------------------------------
# Set up logging so we can see what's happening
# ----------------------------------------------------------
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str = PDF_PATH) -> list[dict]:
    """
    Opens a PDF and extracts text from each page.

    WHAT IT RETURNS:
        A list of dictionaries, one per page. Example:
        [
            {"page": 1, "text": "Chapter 1 content here..."},
            {"page": 2, "text": "Chapter 2 content here..."},
            ...
        ]

    HOW IT WORKS:
        1. pymupdf.open() opens the PDF file
        2. We loop through each page (doc[0], doc[1], etc.)
        3. page.get_text() extracts all readable text
        4. We store it with the page number (starting from 1)

    Args:
        pdf_path: Full file path to the PDF document.

    Returns:
        List of dicts with 'page' (int) and 'text' (str) keys.
    """
    logger.info("Opening PDF: %s", pdf_path)

    # Open the PDF file
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_index in range(doc.page_count):
        # get_text() pulls out all the readable text from that page
        text = doc[page_index].get_text()

        # Only keep pages that actually have text content
        # strip() removes whitespace — if it's all blank, skip it
        if text.strip():
            pages.append({
                "page": page_index + 1,  # Pages start at 1, not 0
                "text": text.strip()
            })

    doc.close()
    logger.info("Extracted text from %d pages", len(pages))
    return pages


def chunk_text(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Splits page text into smaller overlapping chunks.

    WHAT IT RETURNS:
        A list of dictionaries, one per chunk. Example:
        [
            {"chunk_id": "page_1_chunk_0", "page": 1, "text": "..."},
            {"chunk_id": "page_1_chunk_1", "page": 1, "text": "..."},
            {"chunk_id": "page_2_chunk_0", "page": 2, "text": "..."},
            ...
        ]

    HOW THE SLIDING WINDOW WORKS:
        Imagine the text is: "ABCDEFGHIJ" and chunk_size=4, overlap=2

        Chunk 0: "ABCD" (start=0, end=4)
        Chunk 1: "CDEF" (start=2, end=6)  ← "CD" overlaps with chunk 0
        Chunk 2: "EFGH" (start=4, end=8)  ← "EF" overlaps with chunk 1
        Chunk 3: "GHIJ" (start=6, end=10) ← "GH" overlaps with chunk 2

        The "step" is chunk_size - overlap = 4 - 2 = 2
        So we move forward by 2 characters each time.

    Args:
        pages: Output from extract_text_from_pdf().
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk dicts with 'chunk_id', 'page', and 'text'.
    """
    chunks = []

    # The step is how far we move forward for each new chunk
    # If chunk_size=500 and overlap=100, step=400
    step = chunk_size - chunk_overlap

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        # If the entire page text is shorter than one chunk,
        # just use the whole page as one chunk
        if len(text) <= chunk_size:
            chunk_id = f"page_{page_num}_chunk_0"
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": text
            })
            continue

        # Sliding window: move through the text, step characters at a time
        chunk_index = 0
        start = 0

        while start < len(text):
            # Grab chunk_size characters starting from 'start'
            end = start + chunk_size
            chunk_text_piece = text[start:end]

            # Only keep chunks that have meaningful content
            if chunk_text_piece.strip():
                chunk_id = f"page_{page_num}_chunk_{chunk_index}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text_piece.strip()
                })
                chunk_index += 1

            # Move the window forward by 'step' characters
            start += step

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def load_and_chunk_pdf(pdf_path: str = PDF_PATH) -> list[dict]:
    """
    Convenience function: extracts PDF text AND chunks it.

    This is the main function other modules will call.
    It combines both steps into one call.

    Args:
        pdf_path: Full file path to the PDF document.

    Returns:
        List of chunk dicts ready for embedding.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)
    return chunks
