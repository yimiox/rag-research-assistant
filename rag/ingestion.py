"""
ingestion.py — PDF loader, chunker, embedder, FAISS builder
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── constants ──────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 512    # characters (not tokens — simpler, works well)
CHUNK_OVERLAP = 50
INDEX_PATH   = "faiss_index"
META_PATH    = "faiss_meta.pkl"

# ── model (loaded once) ───────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ── PDF → raw text ────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Returns list of dicts: {page, text, source}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": page_num,
                "text": text,
                "source": Path(pdf_path).name,
            })
    doc.close()
    return pages


# ── text → overlapping chunks ─────────────────────────────────────────────
def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks = []
    for page_data in pages:
        text  = page_data["text"]
        start = 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "chunk_text": chunk,
                    "source":     page_data["source"],
                    "page":       page_data["page"],
                    "char_start": start,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── chunks → FAISS index ──────────────────────────────────────────────────
def build_index(chunks: List[Dict]) -> tuple[faiss.Index, List[Dict]]:
    model = get_embedder()
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)       # inner-product = cosine after normalisation
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, chunks


# ── save / load ───────────────────────────────────────────────────────────
def save_index(index: faiss.Index, chunks: List[Dict]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index() -> tuple[faiss.Index, List[Dict]] | tuple[None, None]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None
    index  = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── high-level entry point ────────────────────────────────────────────────
def ingest_pdfs(pdf_paths: List[str], append: bool = False) -> tuple[faiss.Index, List[Dict], str]:
    """
    Ingest one or more PDFs.
    append=True merges with existing index (multi-session).
    Returns (index, chunks, status_message).
    """
    all_chunks: List[Dict] = []

    if append:
        _, existing = load_index()
        if existing:
            all_chunks = existing

    for path in pdf_paths:
        pages  = extract_text_from_pdf(path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, [], "No text extracted. Check PDF files."

    index, chunks = build_index(all_chunks)
    save_index(index, chunks)

    sources = set(c["source"] for c in all_chunks)
    msg = (
        f"Indexed {len(all_chunks)} chunks from "
        f"{len(sources)} document(s): {', '.join(sorted(sources))}"
    )
    return index, chunks, msg