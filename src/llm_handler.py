"""
llm_handler.py
--------------
PURPOSE: Handles all communication with the Ollama local AI model.

HOW OLLAMA WORKS:
-----------------
Ollama runs a local web server on your computer at: http://localhost:11434
When we want the AI to generate text, we send it an HTTP request (like
visiting a webpage, but programmatically).

We send:
  {
    "model": "llama3.2:3b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "What is Python?"}
    ]
  }

Ollama replies with the AI's answer as JSON.

THE SYSTEM PROMPT:
------------------
The system prompt is a special instruction we give the AI EVERY time, before
the user's question. It tells the AI how to behave.

Our system prompt enforces the "document-only" rule:
- Use ONLY the provided context.
- Never use outside knowledge.
- Say "I don't know" if the answer isn't in the context.
"""

import requests
import json
from typing import List, Dict, Optional

# Ollama's local API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# The AI model we use
OLLAMA_MODEL = "llama3.2:3b"

# --- THE ANTI-HALLUCINATION SYSTEM PROMPT ---
# This is the most important piece of text in the entire project.
# It tells the AI: "Only answer from the context. Never make things up."
SYSTEM_PROMPT = """You are a document-based question-answering assistant.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:

1. Your ONLY source of information is the CONTEXT provided below in the user's message.
2. Do NOT use your pre-trained knowledge, general knowledge, assumptions, guesses, or any external information.
3. Do NOT use information from your training data, even if you are very confident about it.
4. If the answer is NOT explicitly stated in the provided context, respond EXACTLY with:
   "I couldn't find this information in the uploaded document. Please ask something related to the available content."
5. Never invent facts, examples, explanations, or citations.
6. Never create fake page numbers or document references.
7. If the context is partially relevant but insufficient to fully answer, say what you CAN answer from the context, then clearly state what is missing.
8. Answer in clear, simple, and understandable language.
9. When answering, refer to the document as "the document" or "the uploaded PDF" — not as your own knowledge.
10. If the user asks something completely unrelated to documents (e.g., "What's the weather?"), say the information is not in the uploaded documents.

Remember: It is BETTER to say "I don't know" than to make something up."""


def check_ollama_connection() -> bool:
    """
    Checks if Ollama is running and accessible.

    Returns:
        True if Ollama is running, False otherwise.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False


def check_model_available(model_name: str = OLLAMA_MODEL) -> bool:
    """
    Checks if the specified model is downloaded and available in Ollama.

    Parameters:
        model_name (str): The model name to check (e.g., "llama3.2:3b").

    Returns:
        True if the model is available, False otherwise.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            available_models = [m["name"] for m in data.get("models", [])]
            # Check if our model (or a version of it) is in the list
            return any(model_name in m for m in available_models)
    except Exception:
        pass
    return False


def ask_ollama(
    question: str,
    context_chunks: List[Dict],
    chat_history: Optional[List[Dict]] = None,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Sends a question + context to Ollama and returns the AI's answer.

    HOW THE CONTEXT IS BUILT:
    --------------------------
    We take the retrieved chunks (most relevant PDF sections) and format them
    into a clear block of text that the AI can read.

    Example of what the AI sees:
    ---
    CONTEXT FROM UPLOADED DOCUMENTS:

    [Source: Python_Notes.pdf | Page: 1]
    Python is a high-level, interpreted programming language known for its
    simplicity and readability...

    [Source: Python_Notes.pdf | Page: 3]
    Functions in Python are defined using the 'def' keyword...

    ---
    QUESTION: What is Python?
    ---

    Parameters:
        question (str): The user's question.
        context_chunks (List[Dict]): Retrieved relevant chunks from FAISS.
        chat_history (List[Dict]): Previous messages in the conversation.
        model (str): The Ollama model to use.

    Returns:
        The AI's answer as a string.
    """
    # Build the context string from retrieved chunks
    context_text = _build_context_string(context_chunks)

    # Build the full user message: context + question
    user_message = f"""CONTEXT FROM UPLOADED DOCUMENTS:
{context_text}

---
QUESTION: {question}
---

Remember: Answer ONLY using the context above. If the answer is not in the context, say so."""

    # Build the messages list for the API
    messages = [
        # System prompt: tells AI how to behave
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add previous chat history (but only user messages, not previous AI answers
    # used as "facts" — history is for conversational context only)
    if chat_history:
        # We include history for conversational flow, but the system prompt
        # keeps the AI grounded in documents only
        for msg in chat_history[-4:]:  # Only last 4 exchanges to save memory
            if msg["role"] in ("user", "assistant"):
                messages.append(msg)

    # Add the current question with context
    messages.append({"role": "user", "content": user_message})

    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # Get the full response at once (not streamed)
        "options": {
            "temperature": 0.1,  # Low temperature = more factual, less creative
            # 0.0 = completely deterministic
            # 1.0 = very creative/random (bad for factual Q&A)
        },
    }

    # Send the request to Ollama
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120,  # Wait up to 2 minutes for a response
        )

        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the JSON response
        data = response.json()

        # Extract the AI's answer from the response
        answer = data["message"]["content"]
        return answer.strip()

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Cannot connect to Ollama.** Please make sure Ollama is running.\n\n"
            "To start Ollama, run this command in PowerShell:\n"
            "```\nollama serve\n```"
        )
    except requests.exceptions.Timeout:
        return (
            "⏱️ **Ollama took too long to respond.** "
            "The model might be loading. Please try again in a moment."
        )
    except Exception as e:
        return f"❌ **Error communicating with Ollama:** {str(e)}"


def _build_context_string(chunks: List[Dict]) -> str:
    """
    Formats retrieved chunks into a readable context string for the AI.

    Parameters:
        chunks: List of chunk dicts with 'text', 'source', 'page', 'score'.

    Returns:
        A formatted string the AI can read and reference.
    """
    if not chunks:
        return "No relevant context found in the uploaded documents."

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")

        part = f"[Source: {source} | Page: {page}]\n{text}"
        context_parts.append(part)

    # Join all parts with a separator
    return "\n\n---\n\n".join(context_parts)

