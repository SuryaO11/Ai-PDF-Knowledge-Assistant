"""
app.py
------
PURPOSE: The main Streamlit application file.

Run with:  streamlit run app.py

Layout:
  Sidebar  │  Left column (chat)  │  Right column (summary panel)

Key state-management design:
  active_summary       – Persistent dict {doc, content} for the summary panel.
                         Written on "Summarize" click, cleared on doc removal.
                         Because it lives in session_state (not in a button's
                         ephemeral scope), it survives reruns and shows the last
                         generated summary until the user removes that document.

  document_summaries   – Per-file summary cache {filename: text}.
                         Cleared atomically in remove_document_from_state().

  remove_document_from_state()  – Single entry-point that cleans EVERY layer
                         (FAISS, embeddings list, summaries cache, active_summary,
                         chat_history when last doc gone) then lets the caller
                         trigger st.rerun() for a synchronised full repaint.
"""

import gc
import logging
import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.summarizer import summarize_document
from src.llm_handler import check_ollama_connection, check_model_available, OLLAMA_MODEL
from src.document_processor import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* ── global ─────────────────────────────── */
.main { background-color: #0e1117; }

/* ── chat bubbles ───────────────────────── */
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user      { background-color:#1e3a5f; border-left:4px solid #4a9eff; }
.chat-message.assistant { background-color:#1a2332; border-left:4px solid #00d4aa; }

/* ── summary panel ──────────────────────── */
.summary-panel {
    background-color: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 0.6rem;
    padding: 1.2rem;
    min-height: 200px;
}
.summary-header {
    color: #4a9eff;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* ── status pills ───────────────────────── */
.status-ok  { color: #00d4aa; font-weight: bold; }
.status-err { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FILE-TYPE ICON MAP  (used in sidebar + source citations)
# ============================================================
TYPE_ICONS: dict = {
    "PDF Document":            "📕",
    "Word Document":           "📘",
    "Word Document (Legacy)":  "📘",
    "PowerPoint Presentation": "📙",
    "Text File":               "📄",
    "Markdown File":           "📝",
}


# ============================================================
# SESSION STATE INITIALISATION
# ============================================================
def init_session_state() -> None:
    """
    Initialises every session-state key exactly once per browser session.

    Keys
    ----
    rag_pipeline         RAGPipeline object — holds FAISS index + embeddings.
    chat_history         List of {role, content, sources?} message dicts.
    processing_status    Transient dict of upload results (shown once, cleared).
    document_summaries   Cache {filename: summary_text}. Cleared on removal.
    active_summary       Persistent display state for the summary panel:
                           {"doc": str | None, "content": str | None}
                         Survives reruns so the right-hand panel stays visible
                         across interactions. Set to None when that doc is removed.
    """
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}

    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}

    # active_summary drives the persistent right-hand summary panel.
    if "active_summary" not in st.session_state:
        st.session_state.active_summary = {"doc": None, "content": None}


init_session_state()

# Convenience alias
pipeline: RAGPipeline = st.session_state.rag_pipeline


# ============================================================
# ATOMIC DOCUMENT REMOVAL HELPER
# ============================================================
def remove_document_from_state(file_name: str) -> bool:
    """
    Single entry-point that cleans EVERY layer of state atomically.

    Layers cleaned
    ──────────────
    1. pipeline.remove_document()   → filters all_chunk_embeddings,
       calls VectorStore.remove_source() + rebuild_index() (or hard_reset()),
       gc.collect() inside VectorStore.
    2. document_summaries cache     → pops file_name key.
    3. active_summary panel state   → cleared if it matches file_name so the
       right-hand summary panel repaints blank after st.rerun().
    4. chat_history                 → cleared only when last doc removed
       (no stale citations left).
    5. Additional gc.collect()      → called when last doc removed.

    Does NOT call st.rerun() — the caller triggers that immediately after
    this returns True, guaranteeing a synchronised full UI repaint.

    Returns True on success, False on failure (error shown via st.error).
    """
    try:
        result = pipeline.remove_document(file_name)

        if result["status"] == "not_found":
            st.warning(f"⚠️ '{file_name}' was not found in the index.")
            return False

        if result["status"] == "error":
            st.error(f"❌ Failed to remove '{file_name}': {result['message']}")
            logger.error("remove_document_from_state: %s", result["message"])
            return False

        # ── Layer 2: invalidate summary cache ──────────────────────────
        st.session_state.document_summaries.pop(file_name, None)

        # ── Layer 3: clear active summary panel if it shows this doc ───
        #    This is the fix for the right-hand panel desync bug.
        #    After st.rerun() the panel reads active_summary["doc"], finds
        #    None, and renders the empty/placeholder state immediately.
        if st.session_state.active_summary.get("doc") == file_name:
            st.session_state.active_summary = {"doc": None, "content": None}
            logger.info(
                "remove_document_from_state: active_summary cleared for '%s'.",
                file_name,
            )

        # ── Layer 4: full reset when last document is gone ─────────────
        if not pipeline.uploaded_docs:
            st.session_state.chat_history = []
            st.session_state.active_summary = {"doc": None, "content": None}
            gc.collect()
            logger.info(
                "remove_document_from_state: last doc removed — full state reset."
            )

        logger.info("remove_document_from_state: '%s' fully cleaned.", file_name)
        return True

    except Exception as exc:
        st.error(f"❌ Unexpected error while removing '{file_name}': {exc}")
        logger.exception("remove_document_from_state: unexpected exception.")
        return False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📚 PDF Knowledge Assistant")
    st.markdown("*AI-powered document Q&A — Locally on your computer*")
    st.divider()

    # ── Ollama / model status ──────────────────────────────────────────
    st.subheader("🔌 System Status")
    ollama_ok = check_ollama_connection()
    model_ok  = check_model_available(OLLAMA_MODEL) if ollama_ok else False

    c1, c2 = st.columns(2)
    with c1:
        label = "✅ Ollama" if ollama_ok else "❌ Ollama"
        css   = "status-ok"  if ollama_ok else "status-err"
        st.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)
    with c2:
        label = "✅ Model"  if model_ok else "❌ Model"
        css   = "status-ok" if model_ok  else "status-err"
        st.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)

    if not ollama_ok:
        st.warning("Ollama is not running!\n```\nollama serve\n```")
    elif not model_ok:
        st.warning(f"Model not found!\n```\nollama pull {OLLAMA_MODEL}\n```")

    st.divider()

    # ── File upload ────────────────────────────────────────────────────
    st.subheader("📤 Upload Documents")
    accepted_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS.keys()]
    st.caption("Supported: PDF · DOCX · PPTX · TXT · MD")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=accepted_types,
        accept_multiple_files=True,
        help="Upload PDF, Word, PowerPoint, TXT or Markdown files.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in pipeline.uploaded_docs:
                with st.spinner(f"Processing {uf.name}…"):
                    try:
                        result = pipeline.add_document(uf)
                        st.session_state.processing_status[uf.name] = result
                    except Exception as exc:
                        st.session_state.processing_status[uf.name] = {
                            "status": "error",
                            "message": f"Unexpected error: {exc}",
                        }
                        logger.exception("Upload failed for '%s'.", uf.name)

    # Show one-shot upload banners then clear them
    for fname, res in list(st.session_state.processing_status.items()):
        if res["status"] == "success":
            st.success(res["message"])
        elif res["status"] == "already_exists":
            st.info(res["message"])
        else:
            st.error(res["message"])
    if st.session_state.processing_status:
        st.session_state.processing_status = {}

    st.divider()

    # ── Uploaded document list ─────────────────────────────────────────
    st.subheader("📂 Uploaded Documents")

    if not pipeline.uploaded_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc_name, doc_info in list(pipeline.uploaded_docs.items()):
            file_type = doc_info.get("file_type", "Document")
            icon      = TYPE_ICONS.get(file_type, "📄")

            # Highlight the doc whose summary is currently displayed
            is_active = (st.session_state.active_summary["doc"] == doc_name)
            label     = f"{icon} {doc_name}" + (" 📋" if is_active else "")

            with st.expander(label, expanded=False):
                st.markdown(f"- **Type:** {file_type}")
                st.markdown(f"- **Pages/Sections:** {doc_info['num_pages']}")
                st.markdown(f"- **Chunks:** {doc_info['num_chunks']}")
                st.markdown(f"- **Size:** {doc_info['file_size_kb']} KB")

                if doc_name in st.session_state.document_summaries:
                    st.caption("📋 Summary cached")

                # ── Remove button ──────────────────────────────────────
                if st.button(
                    "🗑️ Remove",
                    key=f"remove_{doc_name}",
                    type="secondary",
                    use_container_width=True,
                ):
                    with st.spinner("Updating vector index…"):
                        success = remove_document_from_state(doc_name)
                    if success:
                        # st.rerun() triggers an immediate full repaint.
                        # At that point active_summary["doc"] is already None,
                        # so the right-hand panel renders blank synchronously.
                        st.rerun()

    st.divider()

    # ── Stats ──────────────────────────────────────────────────────────
    if pipeline.uploaded_docs:
        st.subheader("📊 Statistics")
        status = pipeline.get_status()
        m1, m2 = st.columns(2)
        m1.metric("Documents", status["num_documents"])
        m2.metric("Total Chunks", status["total_chunks"])

    # ── Clear chat ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# MAIN CONTENT — two columns
#
#   left_col  (ratio 3)  Chat interface
#   right_col (ratio 2)  Persistent summary panel
# ============================================================
st.title("📚 PDF Knowledge Assistant")
st.markdown(
    "Ask questions about your uploaded documents. "
    "The AI answers **only** from document content — never from outside knowledge."
)

if not pipeline.uploaded_docs:
    st.info(
        "👋 **Welcome!** Upload one or more files using the sidebar to get started.\n\n"
        "Once uploaded, you can:\n"
        "- 💬 Ask questions about the content\n"
        "- 📋 Generate a complete document summary (shown in the right panel)\n"
        "- 🔍 See exactly which page each answer came from"
    )

# Split main area into chat (left) and summary panel (right)
left_col, right_col = st.columns([3, 2], gap="large")


# ============================================================
# RIGHT COLUMN — Persistent Summary Panel
# ============================================================
with right_col:
    st.subheader("📋 Document Summary")

    # ── Summarise controls ─────────────────────────────────────────────
    if pipeline.uploaded_docs:
        # Row 1: selectbox (left) + inline remove button (right)
        sel_col, del_col = st.columns([5, 1], gap="small")

        with sel_col:
            doc_to_summarize = st.selectbox(
                "Select document:",
                options=list(pipeline.uploaded_docs.keys()),
                label_visibility="visible",
                key="summary_selectbox",
            )

        with del_col:
            # Align button vertically with the selectbox
            st.markdown("<br>", unsafe_allow_html=True)
            remove_inline_btn = st.button(
                "🗑️",
                key="inline_remove_btn",
                help=f"Remove '{doc_to_summarize}' from the assistant",
                use_container_width=True,
            )

        if remove_inline_btn and doc_to_summarize:
            with st.spinner(f"Removing '{doc_to_summarize}'…"):
                success = remove_document_from_state(doc_to_summarize)
            if success:
                st.rerun()

        # Row 2: summarize button (full width)
        summarize_btn = st.button(
            "📋 Summarize",
            type="primary",
            use_container_width=True,
        )

        if summarize_btn and doc_to_summarize:
            # Serve from cache if available (instant)
            if doc_to_summarize in st.session_state.document_summaries:
                st.session_state.active_summary = {
                    "doc":     doc_to_summarize,
                    "content": st.session_state.document_summaries[doc_to_summarize],
                }
            else:
                # Generate via Map-Reduce summarization
                progress_bar = st.progress(0, text="Generating summary…")

                def _update_progress(val: float) -> None:
                    pct = int(val * 100)
                    progress_bar.progress(val, text=f"Generating summary… {pct}%")

                try:
                    doc_chunks = pipeline.get_all_chunks_for_source(doc_to_summarize)

                    if not doc_chunks:
                        progress_bar.empty()
                        st.error(
                            f"No chunks found for '{doc_to_summarize}'. "
                            "Was it removed?"
                        )
                    else:
                        summary = summarize_document(
                            chunks=doc_chunks,
                            pdf_name=doc_to_summarize,
                            progress_callback=_update_progress,
                        )
                        progress_bar.empty()

                        # ── Persist to both caches ─────────────────────────
                        st.session_state.document_summaries[doc_to_summarize] = summary
                        # active_summary is what the panel reads every rerun.
                        # Setting it here means it survives future reruns until
                        # the doc is removed (which sets it back to None).
                        st.session_state.active_summary = {
                            "doc":     doc_to_summarize,
                            "content": summary,
                        }

                        # Also add to chat history for reference
                        st.session_state.chat_history.append({
                            "role":    "user",
                            "content": f"Summarize the document: {doc_to_summarize}",
                        })
                        st.session_state.chat_history.append({
                            "role":    "assistant",
                            "content": summary,
                            "sources": [],
                        })

                except Exception as exc:
                    progress_bar.empty()
                    st.error(f"❌ Summarization failed: {exc}")
                    logger.exception(
                        "Summarization error for '%s'.", doc_to_summarize
                    )

    st.divider()

    # ── Summary display area ───────────────────────────────────────────
    # This block reads ONLY from active_summary (session state), never from
    # a button's ephemeral scope.  So:
    #   • After "Summarize" → active_summary is set → panel shows content.
    #   • After "Remove"    → active_summary is cleared → panel shows placeholder.
    #   • Both transitions happen on the SAME st.rerun() call → no stale state.
    active = st.session_state.active_summary

    if active["doc"] and active["content"]:
        # Verify the document still exists (defensive guard)
        if active["doc"] in pipeline.uploaded_docs:
            st.markdown(
                f'<div class="summary-panel">'
                f'<div class="summary-header">📄 {active["doc"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(active["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Quick-clear button
            if st.button("✖ Clear summary panel", key="clear_summary_panel"):
                st.session_state.active_summary = {"doc": None, "content": None}
                st.rerun()
        else:
            # Doc was removed but active_summary wasn't cleared (edge case safety)
            st.session_state.active_summary = {"doc": None, "content": None}
            st.rerun()
    else:
        # Placeholder when no summary is active
        st.markdown(
            '<div class="summary-panel">'
            "<p style='color:#4a5568; font-style:italic; text-align:center;"
            "margin-top:3rem;'>"
            "📋 Select a document and click<br><b>Summarize</b> to see its summary here."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# LEFT COLUMN — Chat Interface
# ============================================================
with left_col:
    st.subheader("💬 Ask a Question")

    # Render all previous chat messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(
                        f"📌 Sources ({len(sources)} found)", expanded=False
                    ):
                        for src in sources:
                            pct = int(src["score"] * 100)
                            st.markdown(
                                f"**{TYPE_ICONS.get(src.get('file_type',''), '📄')} "
                                f"{src['source']}** — Page {src['page']} "
                                f"*(Relevance: {pct}%)*"
                            )
                            st.markdown(f"> {src['snippet']}")
                            st.divider()

    # ── Chat input ─────────────────────────────────────────────────────
    if pipeline.uploaded_docs:
        question = st.chat_input("Ask a question about your uploaded documents…")

        if question:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)

            st.session_state.chat_history.append({
                "role":    "user",
                "content": question,
            })

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching documents and generating answer…"):
                    history_for_llm = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                    ]
                    try:
                        result = pipeline.answer_question(
                            question=question,
                            chat_history=history_for_llm,
                        )
                    except Exception as exc:
                        result = {
                            "status": "error",
                            "answer": f"❌ An unexpected error occurred: {exc}",
                            "sources": [],
                        }
                        logger.exception("answer_question raised unexpectedly.")

                st.markdown(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    with st.expander(
                        f"📌 Sources ({len(sources)} found)", expanded=True
                    ):
                        for src in sources:
                            pct = int(src["score"] * 100)
                            st.markdown(
                                f"**📄 {src['source']}** — Page {src['page']} "
                                f"*(Relevance: {pct}%)*"
                            )
                            st.markdown(f"> {src['snippet']}")
                            st.divider()

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })

    else:
        st.chat_input(
            "Upload a document first to start asking questions…",
            disabled=True,
        )
