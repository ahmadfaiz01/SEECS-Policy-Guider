"""
preprocessor.py
PDF text can be pretty messy (weird line breaks, repeated headers/footers).
This file cleans it up before we try to extract meaningful chunks.
"""
import re
from collections import Counter
from typing import List


def clean_text(text: str) -> str:
    """Runs a few regex tricks to fix common PDF extraction issues."""
    text = _fix_hyphenation(text)
    text = _remove_page_numbers(text)
    text = _normalize_whitespace(text)
    return text.strip()


def clean_pages(pages: List[dict]) -> List[dict]:
    """
    Goes through all pages and strips out the headers/footers that repeat everywhere.
    If a line shows up on almost every page, it's probably junk we don't want to index.
    """
    # Grab all non-empty lines from the entire document
    all_lines = [line.strip() for p in pages for line in p["text"].split("\n") if line.strip()]
    counts = Counter(all_lines)
    
    # If a line repeats 3+ times and isn't super long, it's likely a header/footer
    repeated = {line for line, n in counts.items() if n >= 3 and len(line) < 120}

    cleaned = []
    for page in pages:
        lines = page["text"].split("\n")
        # Keep the line only if it's not in our junk list
        filtered = [l for l in lines if l.strip() not in repeated]
        page = dict(page)
        page["text"] = clean_text("\n".join(filtered))
        
        # Don't add empty pages
        if page["text"]:
            cleaned.append(page)
            
    return cleaned


def _fix_hyphenation(text: str) -> str:
    """Fixes words that got split across two lines with a hyphen."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _remove_page_numbers(text: str) -> str:
    """Removes stray page numbers sitting on their own lines."""
    return re.sub(r"^\s*[\|\-]?\s*\d{1,4}\s*[\|\-]?\s*$", "", text, flags=re.MULTILINE)


def _normalize_whitespace(text: str) -> str:
    """Cleans up weird spacing and collapses huge gaps into normal paragraphs."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
