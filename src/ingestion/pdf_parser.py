"""
pdf_parser.py
We use pdfplumber to extract text from the handbooks.
It's pure Python and extracts text accurately without hitting Windows Security policies!
"""
import pdfplumber
from pathlib import Path
from typing import List, Dict


def parse_pdf(pdf_path: str) -> List[Dict]:
    """Extract text page by page."""
    path = Path(pdf_path)
    pages = []

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "page_num": page_num,
                "text": text,
                "source": path.name,
            })

    return pages
