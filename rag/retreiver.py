"""
retriever.py — embed query, search FAISS, return ranked chunks
"""

from typing import List, Dict, Tuple

import faiss
import numpy as np

from rag.ingestion import get_embedder


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Returns top_k chunks most relevant to query.
    Each dict has: chunk_text, source, page, score.
    """
    model  = get_embedder()
    q_emb  = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    # deduplicate by (source, page) — keep highest score per page
    seen: dict[Tuple, Dict] = {}
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r

    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]