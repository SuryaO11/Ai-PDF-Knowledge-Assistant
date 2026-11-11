# PDF Knowledge Assistant

An AI-powered chatbot that answers questions **only from your uploaded PDF documents** — no outside knowledge, no hallucinations.

Built with: Python · Streamlit · Sentence Transformers · FAISS · Ollama (llama3.2:3b)

## Quick Start

```powershell
# 1. Activate virtual environment
cd s:\porttttt\rag
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (in a separate PowerShell window)
ollama serve

# 4. Download the AI model (first time only)
ollama pull llama3.2:3b

# 5. Run the app
streamlit run app.py
```

## Features
- 📤 Upload single or multiple PDFs
- 💬 Ask questions — answered ONLY from document content
- 📋 Summarize entire documents
- 🔍 See page-level source citations for every answer
- 🛡️ Anti-hallucination safeguards at every layer
- 🗑️ Remove documents and rebuild search index

