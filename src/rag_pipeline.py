"""
rag_pipeline.py
---------------
PURPOSE: The main "brain" of the application.
         Connects all the other modules together into one complete pipeline.

This is the central coordinator. When a user uploads a PDF or asks a question,
this module calls the right functions in the right order:

PDF UPLOAD PIPELINE:
  document_processor → text_chunker → embeddings → vector_store

QUESTION ANSWERING PIPELINE:
  embeddings (query) → vector_store (search) → llm_handler (answer)

Think of this as the "orchestra conductor" — it doesn't do the music itself,
but it tells each musician when to play.
"""

import gc
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.document_processor import (
    extract_text,
    save_uploaded_file,
    get_file_info,
    get_file_type_label,
    is_supported,
)
from src.text_chunker import split_text_into_chunks, get_chunk_stats
from src.embeddings import load_embedding_model, embed_chunks, embed_query
from src.vector_store import VectorStore
from src.llm_handler import ask_ollama, check_ollama_connection, check_model_available

logger = logging.getLogger(__name__)

# Directory where uploaded files are saved
UPLOAD_DIR = "data/uploaded_pdfs"


class RAGPipeline:
    """
    The complete RAG (Retrieval-Augmented Generation) pipeline.

    This class manages:
    - The vector store (FAISS database)
    - Tracking which files have been uploaded
    - Storing embeddings for all chunks (needed to rebuild index after deletion)
    - Running the full Q&A pipeline

    It is stored in Streamlit's session_state so it persists across
    user interactions within one session.
    """

    def __init__(self):
        """Initialize an empty RAG pipeline."""
        # The FAISS vector store
        self.vector_store = VectorStore(embedding_dim=384)

        # Track uploaded documents: {filename: {path, num_pages, num_chunks, ...}}
        self.uploaded_docs: Dict[str, Dict] = {}

        # Store ALL embeddings separately so we can rebuild after deletion.
        # Format: list of (chunk_dict, embedding_1d_array) tuples.
        self.all_chunk_embeddings: List[Tuple[Dict, np.ndarray]] = []

        # Load the embedding model once (heavy — ~90 MB on disk)
        self.embedding_model = load_embedding_model()

    # ------------------------------------------------------------------ #
    #  ADD DOCUMENT                                                        #
    # ------------------------------------------------------------------ #

    def add_document(self, uploaded_file) -> Dict:
        """
        Complete pipeline for adding a new document.

        Steps:
        1. Save the uploaded file to disk
        2. Extract text from each page / slide / paragraph
        3. Split text into chunks
        4. Convert chunks to embeddings
        5. Store in FAISS vector store
        6. Track the document metadata

        Parameters:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Dictionary with status and info about the added document.
        """
        file_name = uploaded_file.name

        # Already uploaded?
        if file_name in self.uploaded_docs:
            return {
                "status": "already_exists",
                "message": f"'{file_name}' is already uploaded.",
            }

        # Supported file type?
        if not is_supported(file_name):
            return {
                "status": "error",
                "message": (
                    f"Unsupported file type: '{file_name}'. "
                    "Supported: PDF, DOCX, PPTX, TXT, MD"
                ),
            }

        try:
            # --- Step 1: Save to disk ---
            file_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)

            # --- Step 2: Extract text ---
            pages_data = extract_text(file_path, file_name)

            if not pages_data:
                return {
                    "status": "error",
                    "message": (
                        f"Could not extract text from '{file_name}'. "
                        "The file might be empty, image-only, or encrypted."
                    ),
                }

            # --- Step 3: Split into chunks ---
            chunks = split_text_into_chunks(
                pages_data, chunk_size=900, chunk_overlap=150
            )

            if not chunks:
                return {
                    "status": "error",
                    "message": f"Could not create chunks from '{file_name}'.",
                }

            # --- Step 4: Create embeddings ---
            embeddings = embed_chunks(chunks, self.embedding_model)

            # --- Step 5: Store in FAISS ---
            self.vector_store.add_chunks(chunks, embeddings)

            # --- Step 6: Store raw embeddings for future rebuild ---
            for chunk, emb in zip(chunks, embeddings):
                self.all_chunk_embeddings.append((chunk, emb))

            # --- Step 7: Track document metadata ---
            file_info = get_file_info(file_path, file_name)
            file_type = get_file_type_label(file_name)
            self.uploaded_docs[file_name] = {
                "path": file_path,
                "num_pages": file_info["num_pages"],
                "num_chunks": len(chunks),
                "file_size_kb": file_info["file_size_kb"],
                "file_type": file_type,
            }

            stats = get_chunk_stats(chunks)
            logger.info("add_document: '%s' added (%d chunks).", file_name, len(chunks))

            return {
                "status": "success",
                "message": f"✅ '{file_name}' processed successfully!",
                "num_pages": file_info["num_pages"],
                "num_chunks": len(chunks),
                "file_size_kb": file_info["file_size_kb"],
                "avg_chunk_size": stats["avg_chunk_size"],
                "file_type": file_type,
            }

        except Exception as exc:
            logger.error("add_document: failed for '%s': %s", file_name, exc)
            return {
                "status": "error",
                "message": f"Error processing '{file_name}': {exc}",
            }

    # ------------------------------------------------------------------ #
    #  REMOVE DOCUMENT  (atomic, GC-safe)                                 #
    # ------------------------------------------------------------------ #

    def remove_document(self, file_name: str) -> Dict:
        """
        Atomically removes a document from every layer of state:
          1. Filters it out of self.all_chunk_embeddings
          2. Tells the vector store to drop its chunks
          3. Rebuilds the FAISS index with the surviving embeddings
          4. If no documents remain, performs a hard reset for full GC
          5. Removes the file from self.uploaded_docs

        Returns a dict:
          {"status": "success"|"not_found"|"error", "message": str}
        """
        if file_name not in self.uploaded_docs:
            return {
                "status": "not_found",
                "message": f"'{file_name}' was not found in the uploaded documents.",
            }

        try:
            # ── 1. Filter embeddings list ──────────────────────────────
            self.all_chunk_embeddings = [
                (chunk, emb)
                for chunk, emb in self.all_chunk_embeddings
                if chunk["source"] != file_name
            ]

            # ── 2. Remove chunks from vector store metadata ────────────
            self.vector_store.remove_source(file_name)

            # ── 3. Rebuild FAISS index or hard-reset ───────────────────
            if self.all_chunk_embeddings:
                # Stack surviving embeddings into one array and rebuild
                remaining_embeddings = np.array(
                    [emb for _, emb in self.all_chunk_embeddings],
                    dtype=np.float32,
                )
                self.vector_store.rebuild_index(remaining_embeddings)
                # Explicitly free the temporary stacked array
                del remaining_embeddings
                gc.collect()
            else:
                # Last document removed — full memory reset
                self.vector_store.hard_reset()
                gc.collect()

            # ── 4. Remove from tracking dict ──────────────────────────
            del self.uploaded_docs[file_name]

            logger.info(
                "remove_document: '%s' removed. Remaining docs: %d",
                file_name,
                len(self.uploaded_docs),
            )
            return {
                "status": "success",
                "message": f"'{file_name}' removed and index rebuilt.",
            }

        except Exception as exc:
            logger.error("remove_document: failed for '%s': %s", file_name, exc)
            return {
                "status": "error",
                "message": f"Error removing '{file_name}': {exc}",
            }

    # ------------------------------------------------------------------ #
    #  ANSWER QUESTION                                                     #
    # ------------------------------------------------------------------ #

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = 4,
    ) -> Dict:
        """
        Complete pipeline: question → search → answer.

        Steps:
        1. Convert question to embedding
        2. Search FAISS for relevant chunks
        3. Send question + context to Ollama
        4. Return answer with source citations

        Parameters:
            question (str): The user's question.
            chat_history: Previous conversation messages.
            top_k (int): Number of relevant chunks to retrieve.

        Returns:
            Dictionary with 'answer', 'sources', 'num_chunks_found', 'status'.
        """
        # Guard: Ollama must be running
        if not check_ollama_connection():
            return {
                "status": "error",
                "answer": (
                    "❌ **Ollama is not running.** Please start it with:\n"
                    "```\nollama serve\n```"
                ),
                "sources": [],
                "num_chunks_found": 0,
            }

        # Guard: index must have content
        if self.vector_store.is_empty():
            return {
                "status": "no_docs",
                "answer": "⚠️ No documents uploaded yet. Please upload a file first.",
                "sources": [],
                "num_chunks_found": 0,
            }

        try:
            # --- Step 1: Embed the question ---
            query_embedding = embed_query(question, self.embedding_model)

            # --- Step 2: Search for relevant chunks ---
            relevant_chunks = self.vector_store.search(query_embedding, top_k=top_k)

            # No chunks above threshold
            if not relevant_chunks:
                return {
                    "status": "no_context",
                    "answer": (
                        "I couldn't find this information in the uploaded document. "
                        "Please ask something related to the available content."
                    ),
                    "sources": [],
                    "num_chunks_found": 0,
                }

            # --- Step 3: Get answer from Ollama ---
            answer = ask_ollama(
                question=question,
                context_chunks=relevant_chunks,
                chat_history=chat_history,
            )

            # --- Step 4: Build source citations ---
            sources = _build_sources(relevant_chunks)

            return {
                "status": "success",
                "answer": answer,
                "sources": sources,
                "num_chunks_found": len(relevant_chunks),
            }

        except Exception as exc:
            logger.error("answer_question: unexpected error: %s", exc)
            return {
                "status": "error",
                "answer": f"❌ An unexpected error occurred: {exc}",
                "sources": [],
                "num_chunks_found": 0,
            }

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def get_all_chunks_for_source(self, source_name: str) -> List[Dict]:
        """
        Returns all chunks from a specific document (used by the summarizer).

        Parameters:
            source_name (str): File name (e.g., "report.pdf").

        Returns:
            List of chunk dicts for that document only.
        """
        return [
            chunk
            for chunk, _ in self.all_chunk_embeddings
            if chunk["source"] == source_name
        ]

    def get_status(self) -> Dict:
        """Returns current pipeline status (number of docs, chunks, etc.)."""
        return {
            "num_documents": len(self.uploaded_docs),
            "total_chunks": self.vector_store.get_total_chunks(),
            "documents": list(self.uploaded_docs.keys()),
            "ollama_connected": check_ollama_connection(),
        }


# ------------------------------------------------------------------ #
#  MODULE-LEVEL HELPERS                                               #
# ------------------------------------------------------------------ #

def _build_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Builds a deduplicated list of source citations from retrieved chunks.

    We deduplicate by (source, page) so we don't show the same page twice.

    Parameters:
        chunks: List of retrieved chunk dicts.

    Returns:
        List of source citation dicts: {source, page, snippet, score}
    """
    seen = set()
    sources = []

    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                # Show first 200 chars as a preview snippet
                "snippet": (
                    chunk["text"][:200] + "..."
                    if len(chunk["text"]) > 200
                    else chunk["text"]
                ),
                "score": round(chunk.get("score", 0.0), 3),
            })

    return sources
