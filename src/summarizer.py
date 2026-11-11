"""
summarizer.py
-------------
PURPOSE: Generates summaries of uploaded PDF documents.

THE CHALLENGE WITH LARGE PDFs:
-------------------------------
AI models have a "context window" — a maximum amount of text they can read
at once. For llama3.2:3b, this is about 4,000-8,000 tokens (~3,000-6,000 words).

A 100-page PDF might have 50,000+ words — WAY too much to send at once!

THE SOLUTION: MAP-REDUCE SUMMARIZATION
---------------------------------------
Instead of sending the whole PDF at once, we:

1. MAP phase: Summarize each chunk individually (small batches the AI can handle).
   "Chunk 1 summary: Python is a language..."
   "Chunk 2 summary: Lists store ordered items..."
   "Chunk 3 summary: Functions use the def keyword..."

2. REDUCE phase: Send all the mini-summaries to the AI and ask it to combine
   them into one final summary.

It's like asking multiple people to each summarize one chapter of a book,
then giving all their summaries to an editor who writes the final version.
"""

import requests
from typing import List, Dict
from src.llm_handler import OLLAMA_API_URL, OLLAMA_MODEL, check_ollama_connection

# How many chunks to summarize per batch in the MAP phase
CHUNKS_PER_BATCH = 5

# System prompt specifically for summarization (strict document-only)
SUMMARIZER_SYSTEM_PROMPT = """You are a document summarization assistant.

STRICT RULES:
1. Summarize ONLY the content provided to you in this message.
2. Do NOT add any information from your training knowledge.
3. Do NOT invent examples, explanations, or additional context.
4. Use clear, simple language.
5. Preserve important facts, numbers, and terms from the document.
6. If the content is about a specific topic, stay on that topic only."""


def summarize_document(
    chunks: List[Dict],
    pdf_name: str,
    progress_callback=None,
) -> str:
    """
    Generates a comprehensive summary of a document using Map-Reduce.

    Parameters:
        chunks (List[Dict]): All chunks from the document to summarize.
        pdf_name (str): The PDF filename (for display purposes).
        progress_callback: Optional function to update a progress bar.
                          Called with a float between 0.0 and 1.0.

    Returns:
        A complete document summary as a string.
    """
    if not chunks:
        return "No content found to summarize."

    if not check_ollama_connection():
        return (
            "❌ Cannot connect to Ollama. Please run `ollama serve` in PowerShell."
        )

    total_chunks = len(chunks)

    # === MAP PHASE: Summarize chunks in batches ===
    batch_summaries = []
    num_batches = (total_chunks + CHUNKS_PER_BATCH - 1) // CHUNKS_PER_BATCH

    for batch_idx in range(num_batches):
        # Get the current batch of chunks
        start = batch_idx * CHUNKS_PER_BATCH
        end = min(start + CHUNKS_PER_BATCH, total_chunks)
        batch = chunks[start:end]

        # Build the text for this batch
        batch_text = "\n\n".join([
            f"[Page {c['page']}]: {c['text']}"
            for c in batch
        ])

        # Get pages range for this batch
        pages = [c["page"] for c in batch]
        page_range = f"pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"page {pages[0]}"

        # Ask Ollama to summarize this batch
        mini_summary = _call_ollama_for_summary(
            prompt=f"""Summarize the following content from {pdf_name} ({page_range}).
Extract key points, important concepts, and main ideas.
Only summarize what is explicitly written — do not add outside information.

CONTENT TO SUMMARIZE:
{batch_text}

Provide a concise summary in 3-5 sentences.""",
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
        )

        if mini_summary:
            batch_summaries.append(
                f"[Pages {min(pages)}-{max(pages)}]: {mini_summary}"
            )

        # Update progress bar if provided
        if progress_callback:
            progress = (batch_idx + 1) / (num_batches + 1)  # +1 for reduce phase
            progress_callback(progress)

    if not batch_summaries:
        return "Could not generate summary. Please check if Ollama is running."

    # === REDUCE PHASE: Combine all mini-summaries into one final summary ===
    combined_summaries = "\n\n".join(batch_summaries)

    final_summary = _call_ollama_for_summary(
        prompt=f"""Below are summaries of different sections from the document "{pdf_name}".
Combine these into ONE comprehensive, well-organized final summary.

Requirements:
- Include: Main topics, key concepts, important points
- Organize by topic or section if possible
- Use clear headings if the document has distinct sections
- Keep only information from the summaries provided
- Do NOT add any outside knowledge or information

SECTION SUMMARIES:
{combined_summaries}

Write a comprehensive final summary:""",
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
    )

    # Update progress to 100%
    if progress_callback:
        progress_callback(1.0)

    return final_summary or "Could not generate final summary."


def _call_ollama_for_summary(prompt: str, system_prompt: str) -> str:
    """
    Internal helper: makes one call to Ollama for summarization.

    Parameters:
        prompt (str): The user message / content to summarize.
        system_prompt (str): Instructions for how to behave.

    Returns:
        The AI's response text, or empty string on error.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for factual summarization
        },
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except Exception as e:
        return ""

