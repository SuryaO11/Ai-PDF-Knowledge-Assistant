"""
text_chunker.py
---------------
PURPOSE: Splits long pages of text into smaller "chunks".

WHY DO WE NEED CHUNKS?
-----------------------
Imagine you have a 100-page PDF. The AI model cannot read all 100 pages
at once — it has a limited "memory window" (called context length).

Also, if we send the entire PDF, the AI gets confused by irrelevant pages.
Instead, we split text into small pieces (chunks), and only send the
most relevant chunks to the AI.

It's like a library index card system:
- Instead of handing someone the full book, you give them just the right
  paragraph they need to answer the question.

CHUNK SIZE: 900 characters (~150 words) — small enough to be specific,
            large enough to have meaningful content.

CHUNK OVERLAP: 150 characters — the last 150 chars of one chunk are
               repeated at the start of the next chunk. This prevents
               cutting a sentence in half and losing its meaning.
"""

from typing import List, Dict


def split_text_into_chunks(
    pages_data: List[Dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Takes the extracted page data and splits each page's text into chunks.

    Parameters:
        pages_data (List[Dict]): Output from pdf_processor.extract_text_from_pdf()
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): How many characters to repeat between chunks.

    Returns:
        A list of chunk dictionaries, each containing:
        - 'text'      : The chunk text
        - 'page'      : Which page this chunk came from
        - 'source'    : PDF filename
        - 'chunk_id'  : Unique ID for this chunk (e.g., "Python_Notes.pdf_p1_c0")
    """
    all_chunks = []

    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page"]
        source = page_data["source"]
        pdf_path = page_data.get("pdf_path", "")

        # Split this page's text into chunks
        page_chunks = _chunk_text(text, chunk_size, chunk_overlap)

        # For each chunk, store it with its metadata
        for chunk_index, chunk_text in enumerate(page_chunks):

            chunk = {
                "text": chunk_text,
                "page": page_num,
                "source": source,
                "pdf_path": pdf_path,
                # Unique ID: "filename_page1_chunk0"
                "chunk_id": f"{source}_p{page_num}_c{chunk_index}",
            }
            all_chunks.append(chunk)

    return all_chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Internal helper: splits a single string into overlapping chunks.

    HOW IT WORKS:
    - Start at position 0
    - Take 'chunk_size' characters → that's one chunk
    - Move forward by (chunk_size - chunk_overlap) characters
    - Take the next chunk (which starts 'chunk_overlap' chars before the end of the previous chunk)
    - Repeat until we reach the end of the text

    Example with chunk_size=20, chunk_overlap=5:
        Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        Chunk 1: "ABCDEFGHIJKLMNOPQRST"  (positions 0-19)
        Chunk 2: "PQRSTUVWXYZ"            (positions 15-25, 5 char overlap with chunk 1)
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate where this chunk ends
        end = start + chunk_size

        # Extract the chunk text
        chunk = text[start:end].strip()

        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # If we've reached the end of the text, stop
        if end >= text_length:
            break

        # Move the start forward (minus the overlap, so we repeat some chars)
        start = end - chunk_overlap

    return chunks


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """
    Returns statistics about the chunks (useful for debugging).

    Parameters:
        chunks: List of chunk dictionaries.

    Returns:
        Dictionary with stats like total chunks, average size, etc.
    """
    if not chunks:
        return {"total_chunks": 0}

    sizes = [len(c["text"]) for c in chunks]
    sources = set(c["source"] for c in chunks)

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": round(sum(sizes) / len(sizes)),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "unique_documents": len(sources),
        "documents": list(sources),
    }

