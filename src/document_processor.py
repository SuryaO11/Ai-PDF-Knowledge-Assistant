"""
document_processor.py
---------------------
PURPOSE: Extracts text from many different file types.

Supported formats:
  - PDF  (.pdf)   → PyPDF
  - Word (.docx)  → python-docx
  - PowerPoint (.pptx) → python-pptx
  - Text (.txt)   → built-in Python
  - Markdown (.md) → built-in Python (treat as plain text)

Every extractor returns the SAME format — a list of dictionaries,
one per "page" (or slide, or paragraph group), so the rest of the
pipeline (chunker → embeddings → FAISS) doesn't need to change at all.

Think of this file as a universal translator:
  PDF / DOCX / PPTX / TXT  →  [{"text": ..., "page": ..., "source": ...}, ...]
"""

import os
from typing import List, Dict

# Supported extensions and their display labels
SUPPORTED_EXTENSIONS = {
    ".pdf":  "PDF Document",
    ".docx": "Word Document",
    ".doc":  "Word Document (Legacy)",
    ".pptx": "PowerPoint Presentation",
    ".txt":  "Text File",
    ".md":   "Markdown File",
}


def get_file_extension(filename: str) -> str:
    """Returns the lowercase file extension, e.g. '.pdf'"""
    return os.path.splitext(filename)[1].lower()


def is_supported(filename: str) -> bool:
    """Returns True if the file type is supported."""
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def get_file_type_label(filename: str) -> str:
    """Returns a human-readable label like 'PDF Document'."""
    ext = get_file_extension(filename)
    return SUPPORTED_EXTENSIONS.get(ext, "Unknown File")


def extract_text(file_path: str, file_name: str) -> List[Dict]:
    """
    Master dispatcher: reads the file and routes to the right extractor.

    Parameters:
        file_path (str): Full path to the file on disk.
        file_name (str): Display name of the file (e.g., "Report.docx").

    Returns:
        List of page dicts: [{"text": ..., "page": ..., "source": ..., "pdf_path": ...}]
        Note: We keep the key "pdf_path" as "file_path" but store it as "pdf_path"
              so the rest of the pipeline stays unchanged.
    """
    ext = get_file_extension(file_name)

    if ext == ".pdf":
        return _extract_pdf(file_path, file_name)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_path, file_name)
    elif ext == ".pptx":
        return _extract_pptx(file_path, file_name)
    elif ext in (".txt", ".md"):
        return _extract_text_file(file_path, file_name)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )


# ─────────────────────────────────────────────
# INDIVIDUAL EXTRACTORS
# ─────────────────────────────────────────────

def _extract_pdf(file_path: str, file_name: str) -> List[Dict]:
    """
    Extracts text from a PDF file, one dict per page.
    Uses PyPDF — same as before.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        pages_data.append({
            "text": text.strip(),
            "page": page_number,
            "source": file_name,
            "pdf_path": file_path,
        })

    return pages_data


def _extract_docx(file_path: str, file_name: str) -> List[Dict]:
    """
    Extracts text from a Word (.docx) file.

    Word files don't have traditional "pages" — instead we group
    every 20 paragraphs into one "virtual page" so the chunker
    and citation system work the same way as with PDFs.

    Think of it like: every 20 paragraphs = 1 "page".
    """
    from docx import Document

    doc = Document(file_path)
    pages_data = []

    # Collect all non-empty paragraphs
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Also extract text from tables (cells)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())

    if not paragraphs:
        return []

    # Group into virtual pages (20 paragraphs per page)
    PARAS_PER_PAGE = 20
    page_number = 1

    for i in range(0, len(paragraphs), PARAS_PER_PAGE):
        group = paragraphs[i : i + PARAS_PER_PAGE]
        combined_text = "\n\n".join(group)

        pages_data.append({
            "text": combined_text,
            "page": page_number,
            "source": file_name,
            "pdf_path": file_path,
        })
        page_number += 1

    return pages_data


def _extract_pptx(file_path: str, file_name: str) -> List[Dict]:
    """
    Extracts text from a PowerPoint (.pptx) file.

    Each slide becomes one "page". This is the most natural mapping:
    Slide 1 → Page 1, Slide 2 → Page 2, etc.

    We extract text from all text boxes, titles, and content areas
    on each slide.
    """
    from pptx import Presentation

    prs = Presentation(file_path)
    pages_data = []

    for slide_number, slide in enumerate(prs.slides, start=1):
        slide_texts = []

        # Go through every shape on the slide
        for shape in slide.shapes:
            # Check if this shape has a text frame
            if not shape.has_text_frame:
                continue

            for paragraph in shape.text_frame.paragraphs:
                # Join all "runs" (text segments) in the paragraph
                para_text = "".join(run.text for run in paragraph.runs).strip()
                if para_text:
                    slide_texts.append(para_text)

        if not slide_texts:
            continue  # Skip blank slides

        combined_text = "\n".join(slide_texts)

        pages_data.append({
            "text": combined_text,
            "page": slide_number,       # Slide number = page number
            "source": file_name,
            "pdf_path": file_path,
        })

    return pages_data


def _extract_text_file(file_path: str, file_name: str) -> List[Dict]:
    """
    Extracts text from plain .txt or .md files.

    Text files have no pages, so we split by blank lines to create
    natural paragraph groups, then group those into virtual pages
    (similar to how we handle DOCX).
    """
    # Try UTF-8 first, fall back to latin-1 (handles most Windows text files)
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        return []

    if not content.strip():
        return []

    # Split into paragraphs by blank lines
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    # Group into virtual pages (~3000 chars per page)
    CHARS_PER_PAGE = 3000
    pages_data = []
    page_number = 1
    current_page_text = ""

    for para in paragraphs:
        if len(current_page_text) + len(para) > CHARS_PER_PAGE and current_page_text:
            pages_data.append({
                "text": current_page_text.strip(),
                "page": page_number,
                "source": file_name,
                "pdf_path": file_path,
            })
            page_number += 1
            current_page_text = para + "\n\n"
        else:
            current_page_text += para + "\n\n"

    # Don't forget the last page
    if current_page_text.strip():
        pages_data.append({
            "text": current_page_text.strip(),
            "page": page_number,
            "source": file_name,
            "pdf_path": file_path,
        })

    return pages_data


def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """
    Saves a Streamlit uploaded file to disk.

    Works for any file type — not just PDFs.

    Parameters:
        uploaded_file: Streamlit UploadedFile object.
        save_dir (str): Folder to save into.

    Returns:
        Full path where the file was saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def get_file_info(file_path: str, file_name: str) -> Dict:
    """
    Returns basic info about any uploaded file.

    Parameters:
        file_path (str): Full path to the file.
        file_name (str): Display name.

    Returns:
        Dict with num_pages and file_size_kb.
    """
    ext = get_file_extension(file_name)
    file_size_kb = round(os.path.getsize(file_path) / 1024, 1)
    num_pages = "?"

    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            num_pages = len(PdfReader(file_path).pages)

        elif ext in (".docx", ".doc"):
            from docx import Document
            doc = Document(file_path)
            para_count = len([p for p in doc.paragraphs if p.text.strip()])
            num_pages = max(1, para_count // 20)  # Estimate virtual pages

        elif ext == ".pptx":
            from pptx import Presentation
            num_pages = len(Presentation(file_path).slides)

        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            num_pages = max(1, len(content) // 3000)  # Estimate virtual pages

    except Exception:
        num_pages = "?"

    return {
        "num_pages": num_pages,
        "file_size_kb": file_size_kb,
    }

