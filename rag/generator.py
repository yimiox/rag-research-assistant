"""
generator.py — Ollama (local) with HF Inference API fallback (for HF Spaces)
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

    return f"""<s>[INST] You are a research assistant. Answer using ONLY the context below.
Cite sources inline as [1], [2] etc. If answer not in context, say so.

CONTEXT:
{context}

QUESTION: {query} [/INST]"""


def _try_ollama(prompt: str, model: str) -> Optional[str]:
    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        return None
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"Ollama error: {exc}"


def _try_hf(prompt: str) -> Optional[str]:
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
        result = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        # Strip the prompt from the result if echoed back
        return result.replace(prompt, "").strip()
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
            "Local: `ollama pull phi3` then restart app\n"
            "Cloud: set HF_TOKEN in Space secrets"
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