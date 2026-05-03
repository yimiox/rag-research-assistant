"""
generator.py — build prompt from retrieved chunks, call Ollama, return answer + citations
"""

import re
from typing import List, Dict

import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"          # swap for mistral, phi3, etc.
MAX_CONTEXT  = 3800               # characters of context sent to LLM


def _build_prompt(query: str, chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}] {chunk['source']} — page {chunk['page']}\n"
            f"{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Trim if too long
    if len(context) > MAX_CONTEXT:
        context = context[:MAX_CONTEXT] + "...[truncated]"

    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
For every claim, cite the source number in square brackets like [1] or [2].
If the answer is not in the context, say "I couldn't find relevant information."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with inline citations):"""
    return prompt


def generate_answer(
    query: str,
    chunks: List[Dict],
    model: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """
    Returns (answer_text, citations_block).
    """
    if not chunks:
        return "No relevant context found. Please ingest documents first.", ""

    prompt = _build_prompt(query, chunks)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "Ollama not running. Start it with: `ollama serve` "
            "then `ollama pull llama3`",
            "",
        )
    except Exception as exc:
        return f"LLM error: {exc}", ""

    # Build citations block
    citations = _build_citations(chunks)
    return answer, citations


def _build_citations(chunks: List[Dict]) -> str:
    lines = ["**Sources**"]
    for i, chunk in enumerate(chunks, start=1):
        score_pct = int(chunk.get("score", 0) * 100)
        lines.append(
            f"[{i}] **{chunk['source']}** — page {chunk['page']}  "
            f"(relevance: {score_pct}%)"
        )
    return "\n".join(lines)