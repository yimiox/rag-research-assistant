"""
generator.py — Ollama (local) with HF fallback
"""

import os
from typing import List, Dict, Optional
import requests

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"
HF_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_CONTEXT   = 3800
HF_TOKEN      = os.getenv("HF_TOKEN", "")


def _build_prompt(query: str, chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}] {chunk['source']} — page {chunk['page']}\n"
            f"{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT:
        context = context[:MAX_CONTEXT] + "...[truncated]"

    return f"""You are a research assistant. Answer the question using ONLY the provided context.
For every claim, cite the source number in square brackets like [1] or [2].
If the answer is not in the context, say "I couldn't find relevant information."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with inline citations):"""


def _try_ollama(prompt: str, model: str) -> Optional[str]:
    try:
        # First ping to check if Ollama is up
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        return None  # Ollama not running

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=300,  # 5 min — first load is slow
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"Ollama error: {exc}"


def _try_hf(prompt: str) -> Optional[str]:
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.1}}
    try:
        url  = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "").replace(prompt, "").strip()
        return str(data)
    except Exception as exc:
        return f"HF error: {exc}"


def generate_answer(query: str, chunks: List[Dict], model: str = DEFAULT_MODEL) -> tuple[str, str]:
    if not chunks:
        return "No relevant context found. Please ingest documents first.", ""

    prompt = _build_prompt(query, chunks)
    answer = _try_ollama(prompt, model)
    if answer is None:
        answer = _try_hf(prompt)
    if answer is None:
        answer = (
            "No LLM available.\n\n"
            "1. Open Ollama desktop app\n"
            "2. Run in terminal: ollama pull llama3\n"
            "3. Try again"
        )

    return answer, _build_citations(chunks)


def _build_citations(chunks: List[Dict]) -> str:
    lines = ["**Sources**"]
    for i, chunk in enumerate(chunks, start=1):
        score_pct = int(chunk.get("score", 0) * 100)
        lines.append(
            f"[{i}] **{chunk['source']}** — page {chunk['page']}  "
            f"(relevance: {score_pct}%)"
        )
    return "\n".join(lines)