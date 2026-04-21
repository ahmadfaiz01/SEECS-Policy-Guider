"""
chunker.py
Instead of indexing whole pages, we split the text into overlapping chunks.
Why overlap? Because a policy rule might start at the bottom of one page 
and end on the next. Overlapping ensures we don't accidentally split a rule in half.
"""
import re
from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Dict]:
    """
    Takes cleaned pages and chops them into smaller chunks of text.
    We keep track of the page numbers for citations later.
    """
    word_meta: List[Dict] = []
    
    # Flatten the document into a list of words, keeping the page context for each
    for page in pages:
        words = page["text"].split()
        for word in words:
            word_meta.append({"word": word, "page_num": page["page_num"], "source": page["source"]})

    if not word_meta:
        return []

    chunks: List[Dict] = []
    stride = chunk_size - overlap
    chunk_idx = 0

    # Slide our window across the document
    for start in range(0, len(word_meta), stride):
        end = min(start + chunk_size, len(word_meta))
        slice_ = word_meta[start:end]

        # Ignore tiny bits at the end of the doc
        if len(slice_) < 50:
            break

        text = " ".join(w["word"] for w in slice_)
        source = slice_[0]["source"]

        chunks.append({
            "chunk_id": f"{source}__c{chunk_idx:04d}",
            "text": text,
            "source": source,
            "start_page": slice_[0]["page_num"],
            "end_page": slice_[-1]["page_num"],
            "section": _detect_section(text),
            "word_count": len(slice_),
        })
        chunk_idx += 1

    return chunks


def _detect_section(text: str) -> str:
    """
    Tries to figure out the section title by looking at the first few lines.
    Matches stuff like '3.1 Grading System' or 'CHAPTER 5'.
    It's a bit hacky but works decently well for standard handbooks.
    """
    for line in text.split("\n")[:4]:
        line = line.strip()
        if re.match(r"^(\d+\.?\d*\s+[A-Z]|CHAPTER|SECTION|ARTICLE|PART\s)", line, re.IGNORECASE):
            return line[:100]
    return "General"
