"""
embeddings.py
-------------
PURPOSE: Converts text into numbers (embeddings) using Sentence Transformers.

WHAT ARE EMBEDDINGS?
--------------------
An "embedding" is a list of numbers that REPRESENTS the meaning of a piece of text.

Simple analogy: Think of colors.
- Red can be represented as (255, 0, 0) in RGB numbers.
- Blue can be represented as (0, 0, 255).
- Purple (a mix of red and blue) is (128, 0, 128) — in between!

Embeddings work similarly for text:
- "Dog runs fast"      → [0.21, -0.45, 0.88, ...] (384 numbers)
- "The cat moves quick"→ [0.19, -0.43, 0.85, ...] (similar! close meanings)
- "I love pizza"       → [0.91,  0.12, -0.33, ...] (very different numbers)

The model we use: "all-MiniLM-L6-v2"
- Small (only ~90 MB to download)
- Fast to run on a CPU
- Creates 384-dimensional embeddings
- Good quality for question-answering tasks
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import streamlit as st


# The name of the embedding model we use
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """
    Loads the Sentence Transformer model.

    When called inside a Streamlit app, the result is cached automatically
    via _get_cached_model() so the model loads only ONCE per session.
    When called from tests or scripts, it loads normally without caching.

    Returns:
        The loaded SentenceTransformer model.
    """
    return _get_cached_model()


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_cached_model() -> SentenceTransformer:
    """Streamlit-cached version — only runs once per app session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Converts a list of text strings into embeddings (numerical arrays).

    Parameters:
        texts (List[str]): List of text strings to embed.
        model: The loaded SentenceTransformer model.

    Returns:
        A 2D numpy array of shape (num_texts, 384).
        Each row is the embedding for one text.

    Example:
        Input:  ["Python is a language", "Lists store multiple values"]
        Output: array([[0.21, -0.45, ...],   ← embedding for text 1
                        [0.33,  0.12, ...]])  ← embedding for text 2
    """
    # encode() does the conversion: text → numbers
    # show_progress_bar=False keeps the UI clean
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        # normalize_embeddings=True makes similarity search more accurate
        normalize_embeddings=True,
        # Process in batches of 32 for memory efficiency
        batch_size=32,
    )

    # Return as float32 numpy array (FAISS requires float32)
    return np.array(embeddings, dtype=np.float32)


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Converts a list of chunk dictionaries into embeddings.

    Parameters:
        chunks: List of chunk dicts (output from text_chunker).
        model: The loaded SentenceTransformer model.

    Returns:
        2D numpy array of embeddings, one per chunk.
    """
    # Extract just the text from each chunk dict
    texts = [chunk["text"] for chunk in chunks]

    # Convert all texts to embeddings
    return embed_texts(texts, model)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Converts a single user question into an embedding.

    This uses the SAME model as embed_chunks() — which is essential!
    If the question and chunks use different models, the similarity search
    won't work correctly (like searching for a word in the wrong language).

    Parameters:
        query (str): The user's question text.
        model: The loaded SentenceTransformer model.

    Returns:
        A 2D numpy array of shape (1, 384) — ready for FAISS search.
    """
    # Encode the single query
    embedding = model.encode(
        [query],  # Pass as a list (even though it's just one item)
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.array(embedding, dtype=np.float32)

