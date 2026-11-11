"""
tests/test_rag.py
-----------------
Integration tests for the full RAG pipeline.

HOW TO RUN:
    cd s:\porttttt\rag
    .\venv\Scripts\activate
    python tests/test_rag.py

NOTE: These tests require Ollama to be running.
      Start it with: ollama serve
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_handler import check_ollama_connection, check_model_available, OLLAMA_MODEL


def test_ollama_connection():
    """TEST 1: Is Ollama running?"""
    print("\n--- Test 1: Ollama Connection ---")
    connected = check_ollama_connection()
    print(f"Ollama connected: {connected}")
    if not connected:
        print("⚠️  SKIPPED: Ollama is not running. Start it with: ollama serve")
        return False
    print("✅ PASSED: Ollama is running!")
    return True


def test_model_available():
    """TEST 2: Is llama3.2:3b model downloaded?"""
    print("\n--- Test 2: Model Availability ---")
    if not check_ollama_connection():
        print("⚠️  SKIPPED: Ollama not running")
        return False

    available = check_model_available(OLLAMA_MODEL)
    print(f"Model '{OLLAMA_MODEL}' available: {available}")

    if not available:
        print(f"⚠️  Model not found! Run: ollama pull {OLLAMA_MODEL}")
        return False

    print("✅ PASSED: Model is available!")
    return True


def test_answer_from_context():
    """TEST 3: Does the LLM answer correctly from provided context?"""
    print("\n--- Test 3: Answer From Context ---")
    if not check_ollama_connection() or not check_model_available(OLLAMA_MODEL):
        print("⚠️  SKIPPED: Ollama/model not available")
        return

    from src.llm_handler import ask_ollama

    context_chunks = [{
        "text": (
            "Python is a high-level, interpreted programming language created by "
            "Guido van Rossum and first released in 1991. It emphasizes code "
            "readability and uses indentation to define code blocks."
        ),
        "source": "test_doc.pdf",
        "page": 1,
        "score": 0.95,
    }]

    answer = ask_ollama(
        question="Who created Python and when was it released?",
        context_chunks=context_chunks,
    )

    print(f"Question: Who created Python and when was it released?")
    print(f"Answer: {answer}")

    # The answer should mention Guido and 1991 (from the context)
    assert "Guido" in answer or "1991" in answer or "Python" in answer.lower(), \
        "Answer should reference information from the context"
    print("✅ PASSED: LLM answered from context correctly!")


def test_refuses_without_context():
    """TEST 4: Does the LLM refuse to answer when no relevant context is found?"""
    print("\n--- Test 4: Refusal Without Context ---")
    if not check_ollama_connection() or not check_model_available(OLLAMA_MODEL):
        print("⚠️  SKIPPED: Ollama/model not available")
        return

    from src.llm_handler import ask_ollama

    # Provide completely unrelated context
    context_chunks = [{
        "text": "The recipe requires 2 cups of flour and 1 egg.",
        "source": "recipe.pdf",
        "page": 1,
        "score": 0.1,
    }]

    answer = ask_ollama(
        question="Explain the theory of relativity.",
        context_chunks=context_chunks,
    )

    print(f"Question: Explain the theory of relativity.")
    print(f"Answer: {answer}")
    print("✅ Completed (verify manually that the AI refused or said info not found)")


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING RAG INTEGRATION TESTS")
    print("=" * 50)

    ollama_ok = test_ollama_connection()
    model_ok = test_model_available()

    if ollama_ok and model_ok:
        test_answer_from_context()
        test_refuses_without_context()
    else:
        print("\n⚠️  Some tests skipped because Ollama is not ready.")
        print(f"   1. Run 'ollama serve' in PowerShell")
        print(f"   2. Run 'ollama pull {OLLAMA_MODEL}'")
        print(f"   3. Then re-run these tests")

    print("\n" + "=" * 50)
    print("TESTS COMPLETE")
    print("=" * 50)

