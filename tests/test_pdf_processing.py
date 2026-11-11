"""
tests/test_pdf_processing.py
-----------------------------
Tests for PDF text extraction and chunking.

HOW TO RUN:
    cd s:\porttttt\rag
    .\venv\Scripts\activate
    python tests/test_pdf_processing.py
"""

import sys
import os

# Add project root to Python path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_chunker import split_text_into_chunks, _chunk_text, get_chunk_stats


def test_chunking_basic():
    """TEST: Does text chunking work correctly?"""
    print("\n--- Test 1: Basic Text Chunking ---")

    sample_text = "A" * 2000  # 2000 characters of 'A'
    sample_pages = [
        {"text": sample_text, "page": 1, "source": "test.pdf", "pdf_path": "test.pdf"}
    ]

    chunks = split_text_into_chunks(sample_pages, chunk_size=900, chunk_overlap=150)

    print(f"Input text length: {len(sample_text)} chars")
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk['text'])} chars (page {chunk['page']})")

    assert len(chunks) > 1, "Should create multiple chunks from long text"
    assert all(len(c["text"]) <= 900 for c in chunks), "No chunk should exceed chunk_size"
    print("✅ PASSED: Chunking works correctly!")


def test_chunk_overlap():
    """TEST: Does chunk overlap work (do consecutive chunks share content)?"""
    print("\n--- Test 2: Chunk Overlap ---")

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20  # 520 chars
    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=20)

    print(f"Number of chunks: {len(chunks)}")
    if len(chunks) >= 2:
        end_of_first = chunks[0][-20:]
        start_of_second = chunks[1][:20]
        print(f"End of chunk 1:   '{end_of_first}'")
        print(f"Start of chunk 2: '{start_of_second}'")
        overlap_found = end_of_first == start_of_second
        print(f"Overlap detected: {overlap_found}")
        assert overlap_found, "Chunk overlap should repeat content!"

    print("✅ PASSED: Overlap works correctly!")


def test_metadata_preserved():
    """TEST: Does chunk metadata (page, source) carry through correctly?"""
    print("\n--- Test 3: Metadata Preservation ---")

    pages = [
        {"text": "X" * 1000, "page": 1, "source": "doc1.pdf", "pdf_path": "doc1.pdf"},
        {"text": "Y" * 1000, "page": 2, "source": "doc1.pdf", "pdf_path": "doc1.pdf"},
    ]
    chunks = split_text_into_chunks(pages)

    page1_chunks = [c for c in chunks if c["page"] == 1]
    page2_chunks = [c for c in chunks if c["page"] == 2]

    print(f"Chunks from page 1: {len(page1_chunks)}")
    print(f"Chunks from page 2: {len(page2_chunks)}")
    assert len(page1_chunks) > 0, "Should have chunks from page 1"
    assert len(page2_chunks) > 0, "Should have chunks from page 2"
    assert all(c["source"] == "doc1.pdf" for c in chunks), "Source should be preserved"

    print("✅ PASSED: Metadata preserved correctly!")


def test_empty_text():
    """TEST: Does the system handle empty/blank pages gracefully?"""
    print("\n--- Test 4: Empty Text Handling ---")

    pages = [{"text": "", "page": 1, "source": "empty.pdf", "pdf_path": "empty.pdf"}]
    chunks = split_text_into_chunks(pages)
    print(f"Chunks from empty page: {len(chunks)}")
    assert len(chunks) == 0, "Empty text should produce no chunks"
    print("✅ PASSED: Empty text handled correctly!")


def test_chunk_stats():
    """TEST: Does get_chunk_stats return correct info?"""
    print("\n--- Test 5: Chunk Statistics ---")
    pages = [{"text": "Hello world " * 100, "page": 1, "source": "test.pdf", "pdf_path": "test.pdf"}]
    chunks = split_text_into_chunks(pages)
    stats = get_chunk_stats(chunks)
    print(f"Stats: {stats}")
    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(chunks)
    print("✅ PASSED: Stats calculated correctly!")


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING PDF PROCESSING TESTS")
    print("=" * 50)
    test_chunking_basic()
    test_chunk_overlap()
    test_metadata_preserved()
    test_empty_text()
    test_chunk_stats()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✅")
    print("=" * 50)

