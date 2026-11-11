"""
tests/test_retrieval.py
-----------------------
Tests for embeddings and FAISS vector search.

HOW TO RUN:
    cd s:\porttttt\rag
    .\venv\Scripts\activate
    python tests/test_retrieval.py

NOTE: These tests require sentence-transformers to be installed.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_embeddings_shape():
    """TEST: Do embeddings have the correct shape (N, 384)?"""
    print("\n--- Test 1: Embedding Shape ---")
    from src.embeddings import load_embedding_model, embed_texts

    model = load_embedding_model()
    texts = ["Python is a programming language", "Lists store multiple values"]
    embeddings = embed_texts(texts, model)

    print(f"Input texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    assert embeddings.shape == (2, 384), f"Expected (2, 384), got {embeddings.shape}"
    print("✅ PASSED: Embeddings shape is correct!")


def test_similar_texts_closer():
    """TEST: Do similar texts have more similar embeddings?"""
    print("\n--- Test 2: Semantic Similarity ---")
    from src.embeddings import load_embedding_model, embed_texts

    model = load_embedding_model()
    embeddings = embed_texts([
        "Python is a programming language",   # Very similar to query
        "Python programming syntax and rules", # Similar
        "I love eating pizza and pasta",       # Completely unrelated
    ], model)

    # Compute cosine similarities with the first text as query
    query = embeddings[0]
    sim_1_2 = float(np.dot(query, embeddings[1]))  # Should be HIGH
    sim_1_3 = float(np.dot(query, embeddings[2]))  # Should be LOW

    print(f"Similarity: Text1 vs Text2 (similar topic): {sim_1_2:.3f}")
    print(f"Similarity: Text1 vs Text3 (different topic): {sim_1_3:.3f}")
    assert sim_1_2 > sim_1_3, "Similar texts should have higher similarity!"
    print("✅ PASSED: Similar texts are closer in embedding space!")


def test_faiss_search():
    """TEST: Does FAISS retrieve the most relevant chunk?"""
    print("\n--- Test 3: FAISS Vector Search ---")
    from src.embeddings import load_embedding_model, embed_texts, embed_query
    from src.vector_store import VectorStore

    model = load_embedding_model()

    # Create some test chunks
    chunks = [
        {"text": "Python is a high-level programming language.", "page": 1, "source": "test.pdf", "chunk_id": "c1"},
        {"text": "FAISS is used for similarity search.", "page": 2, "source": "test.pdf", "chunk_id": "c2"},
        {"text": "The Eiffel Tower is located in Paris, France.", "page": 3, "source": "test.pdf", "chunk_id": "c3"},
        {"text": "Machine learning uses algorithms to learn from data.", "page": 4, "source": "test.pdf", "chunk_id": "c4"},
    ]

    # Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model)

    # Build vector store
    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    # Search for something about Python
    query_emb = embed_query("What is Python?", model)
    results = store.search(query_emb, top_k=2, relevance_threshold=0.0)

    print(f"Query: 'What is Python?'")
    print(f"Top result: '{results[0]['text']}'")
    print(f"Top result score: {results[0]['score']:.3f}")

    assert "Python" in results[0]["text"], "Top result should be about Python!"
    print("✅ PASSED: FAISS correctly finds the most relevant chunk!")


def test_remove_source():
    """TEST: Does removing a source correctly remove its chunks from search?"""
    print("\n--- Test 4: Remove Source ---")
    from src.embeddings import load_embedding_model, embed_texts, embed_query
    from src.vector_store import VectorStore

    model = load_embedding_model()

    chunks = [
        {"text": "Python is a programming language.", "page": 1, "source": "doc1.pdf", "chunk_id": "c1"},
        {"text": "Machine learning is a field of AI.", "page": 1, "source": "doc2.pdf", "chunk_id": "c2"},
    ]

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model)

    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    # Remove doc1.pdf
    store.remove_source("doc1.pdf")
    store.rebuild_index(embeddings[1:2])

    # Search for Python — should not find doc1.pdf anymore
    query_emb = embed_query("Python programming", model)
    results = store.search(query_emb, top_k=5, relevance_threshold=0.0)

    sources_found = [r["source"] for r in results]
    print(f"Sources in results after removing doc1.pdf: {sources_found}")
    assert "doc1.pdf" not in sources_found, "doc1.pdf should be removed!"
    print("✅ PASSED: Removed source is not in search results!")


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING RETRIEVAL TESTS")
    print("Note: First run downloads the embedding model (~90MB)")
    print("=" * 50)
    test_embeddings_shape()
    test_similar_texts_closer()
    test_faiss_search()
    test_remove_source()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✅")
    print("=" * 50)

