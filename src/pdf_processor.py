"""
pdf_processor.py
----------------
PURPOSE: This file extracts text from uploaded PDF files.

Think of it like a person who reads every page of a PDF and writes down
all the text, keeping track of which page each sentence came from.
"""

import os
from pypdf import PdfReader
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str, pdf_name: str) -> List[Dict]:
    """
    Reads a PDF file and extracts text from every page.

    Parameters:
        pdf_path (str): The full file path to the PDF on disk.
        pdf_name (str): The display name of the PDF (e.g., "Python_Notes.pdf").

    Returns:
        A list of dictionaries. Each dictionary represents ONE PAGE and contains:
        - 'text'      : The text content of that page
        - 'page'      : The page number (starting from 1)
        - 'source'    : The PDF filename
        - 'pdf_path'  : Full path to the PDF file

    Example output:
        [
            {'text': 'Chapter 1: Python is...', 'page': 1, 'source': 'Python_Notes.pdf'},
            {'text': 'Lists are ordered...', 'page': 2, 'source': 'Python_Notes.pdf'},
        ]
    """
    pages_data = []

    # Check if the file exists before trying to read it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Open and read the PDF using PyPDF
    reader = PdfReader(pdf_path)

    # Loop through every page in the PDF
    for page_number, page in enumerate(reader.pages, start=1):

        # Extract the text from this page
        text = page.extract_text()

        # Sometimes a page has no readable text (like a pure image page)
        # We skip those pages
        if not text or text.strip() == "":
            continue

        # Clean up the text a bit — remove extra whitespace
        text = text.strip()

        # Store this page's information as a dictionary
        page_data = {
            "text": text,
            "page": page_number,
            "source": pdf_name,
            "pdf_path": pdf_path,
        }

        pages_data.append(page_data)

    return pages_data


def save_uploaded_pdf(uploaded_file, save_dir: str) -> str:
    """
    Saves a Streamlit uploaded file to disk.

    When a user uploads a PDF through Streamlit, it arrives as a special
    Streamlit object in memory. This function saves it as a real file on disk.

    Parameters:
        uploaded_file: The Streamlit UploadedFile object.
        save_dir (str): The folder where we save the PDF.

    Returns:
        The full file path where the PDF was saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Build the full path: e.g., "data/uploaded_pdfs/Python_Notes.pdf"
    file_path = os.path.join(save_dir, uploaded_file.name)

    # Write the uploaded file bytes to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def get_pdf_info(pdf_path: str) -> Dict:
    """
    Returns basic information about a PDF.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        Dictionary with 'num_pages' and 'file_size_kb'.
    """
    reader = PdfReader(pdf_path)
    file_size = os.path.getsize(pdf_path) / 1024  # Convert bytes to KB

    return {
        "num_pages": len(reader.pages),
        "file_size_kb": round(file_size, 1),
    }

