"""
vector_store.py
---------------
PURPOSE: Manages the FAISS vector database — stores and searches embeddings.

WHAT IS FAISS?
--------------
FAISS = Facebook AI Similarity Search.

Imagine you have 10,000 chunks of text, each converted to 384 numbers.
When a user asks a question, we need to find which chunks are most similar.

Brute force: Compare question embedding to all 10,000 chunk embeddings one by one.
             → Slow! Like reading every page of an encyclopedia.

FAISS: Uses smart indexing to find the closest embeddings in milliseconds.
       → Fast! Like using a book's index to jump straight to the right page.

HOW SIMILARITY WORKS:
----------------------
Two embeddings are "similar" if they point in similar directions in 384-dimensional space.
We measure this with "cosine similarity" (or equivalently, inner product after normalization):
- Score near 1.0  = very similar meaning
- Score near 0.0  = unrelated
- Score near -1.0 = opposite meaning

RELEVANCE THRESHOLD:
--------------------
We set a minimum similarity score. If retrieved chunks score below this threshold,
we consider them "not relevant enough" and don't pass them to the AI.
This prevents the AI from getting garbage context and making up answers.
"""

import gc
import logging
import faiss
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Minimum similarity score to consider a chunk as "relevant"
# Score is between 0 and 1 (after normalization)
# 0.3 = at least 30% similar. Chunks below this are ignored.
RELEVANCE_THRESHOLD = 0.3


class VectorStore:
    """
    A wrapper around FAISS that also stores the original text and metadata.

    FAISS only stores numbers (embeddings). But we also need:
    - The original text of each chunk
    - The source PDF filename
    - The page number

    This class keeps both in sync: FAISS index + a list of chunk metadata.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize an empty vector store.

        Parameters:
            embedding_dim (int): Size of each embedding vector.
                                 all-MiniLM-L6-v2 produces 384-dimensional vectors.
        """
        self.embedding_dim = embedding_dim
        self._init_index()

    def _init_index(self) -> None:
        """
        Creates a fresh FAISS index and clears all chunk metadata.

        We call this both during __init__ and when doing a full reset.
        Explicitly deletes the old index object and runs gc.collect() to
        immediately free the RAM that FAISS was holding.
        """
        # Delete the old index if it exists so FAISS releases its C++ memory
        if hasattr(self, "index") and self.index is not None:
            del self.index
            gc.collect()

        # Create a FAISS index that uses Inner Product (dot product) similarity.
        # Because our embeddings are normalized, inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # This list stores the metadata for each chunk, in the SAME ORDER as FAISS.
        # When FAISS returns "chunk at position 5", we look up self.chunks[5]
        # to get the original text, source, and page number.
        self.chunks: List[Dict] = []

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """
        Adds chunks and their embeddings to the store.

        Parameters:
            chunks: List of chunk dicts (text, source, page, chunk_id).
            embeddings: 2D numpy array of shape (num_chunks, 384).
        """
        if len(chunks) == 0:
            return

        # Ensure correct dtype for FAISS
        embeddings = np.array(embeddings, dtype=np.float32)

        # Add embeddings to the FAISS index
        self.index.add(embeddings)

        # Add the corresponding metadata to our list
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Searches for the most relevant chunks to a query.

        Parameters:
            query_embedding: 2D array of shape (1, 384) — the question embedding.
            top_k (int): How many top results to return (default: 4).
                         e.g., top_k=4 returns the 4 most relevant chunks.
            relevance_threshold (float): Minimum similarity score (0 to 1).
                         Chunks with score below this are discarded.
            source_filter: If provided, only return chunks from these PDF files.

        Returns:
            List of relevant chunk dicts, each augmented with a 'score' key.
            Sorted by relevance score (highest first).

        HOW top_k WORKS:
            If you have 1000 chunks and ask top_k=4:
            FAISS compares the query to all 1000 chunks and returns only
            the 4 that are most similar. We then further filter by threshold.
        """
        if self.index.ntotal == 0:
            return []

        # Ensure correct shape and dtype
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # We retrieve more than top_k initially so we have candidates to filter
        search_k = min(top_k * 3, self.index.ntotal)

        # FAISS search!
        # scores: similarity scores (higher = more similar)
        # indices: positions in self.chunks list
        scores, indices = self.index.search(query_embedding, search_k)

        # Flatten from 2D to 1D (FAISS returns 2D arrays)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            # FAISS returns -1 for empty slots — skip those
            if idx == -1:
                continue

            # Apply relevance threshold — skip chunks that are too different
            if score < relevance_threshold:
                continue

            # Get the chunk metadata
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)

            # Apply source filter if specified
            if source_filter and chunk["source"] not in source_filter:
                continue

            results.append(chunk)

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results

    def remove_source(self, source_name: str) -> bool:
        """
        Atomically removes all chunks belonging to source_name, then rebuilds
        the FAISS index with the remaining chunks.

        Returns True if at least one chunk was removed, False otherwise.
        The caller (rag_pipeline) must still supply the remaining embeddings
        via rebuild_index() when it has them.
        """
        remaining_chunks = [c for c in self.chunks if c["source"] != source_name]

        if len(remaining_chunks) == len(self.chunks):
            logger.warning("remove_source: '%s' not found in vector store.", source_name)
            return False

        # Reset and free FAISS memory before rebuilding
        self.index.reset()
        gc.collect()

        self.chunks = remaining_chunks
        logger.info(
            "remove_source: removed '%s'. Remaining chunks: %d",
            source_name, len(remaining_chunks),
        )
        return True

    def rebuild_index(self, embeddings: np.ndarray) -> None:
        """
        Rebuilds the FAISS index from scratch with given embeddings.

        Called after remove_source() with the surviving embeddings.
        Runs gc.collect() after the old index is replaced so Python/FAISS
        releases the stale C++ memory immediately.

        Parameters:
            embeddings: 2D float32 array of shape (num_remaining_chunks, 384).
        """
        old_index = self.index
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        if len(embeddings) > 0:
            embeddings = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings)

        # Explicitly free the old index
        del old_index
        gc.collect()
        logger.info("rebuild_index: index rebuilt with %d vectors.", self.index.ntotal)

    def hard_reset(self) -> None:
        """
        Completely wipes the vector store — index, chunks, and all memory.

        Called when the last document is removed so the app returns to a
        clean initial state without any stale references.
        """
        self._init_index()
        gc.collect()
        logger.info("hard_reset: vector store fully cleared.")

    def get_all_sources(self) -> List[str]:
        """Returns a list of all unique PDF filenames in the store."""
        return list(set(c["source"] for c in self.chunks))

    def get_total_chunks(self) -> int:
        """Returns the total number of chunks stored."""
        return self.index.ntotal

    def is_empty(self) -> bool:
        """Returns True if no documents have been added yet."""
        return self.index.ntotal == 0
