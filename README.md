---
title: RAG Research Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---

# RAG Research Assistant

Local Retrieval-Augmented Generation pipeline. Upload PDFs → ask questions → get cited answers.

## Stack
- **FAISS** — vector index (local disk)
- **HuggingFace Sentence Transformers** — `all-MiniLM-L6-v2` embeddings
- **Ollama** — local LLM inference (llama3, mistral, phi3)
- **PyMuPDF** — PDF text extraction
- **Gradio** — UI + HuggingFace Spaces

## Local Setup

```bash
# 1. Clone
git clone https://github.com/yimiox/rag-research-assistant
cd rag-research-assistant

# 2. Install
pip install -r requirements.txt

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull llama3      # or: mistral, phi3, gemma

# 4. Run
python app.py
# → open http://localhost:7860
```

## HuggingFace Spaces Deploy

```bash
# Push to HF Spaces
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/rag-research-assistant
git push space main
```

**Note for Spaces:** Ollama won't run in HF free tier.  
Swap `generator.py` to use `huggingface_hub.InferenceClient` for a fully cloud version:

```python
from huggingface_hub import InferenceClient
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
answer = client.text_generation(prompt, max_new_tokens=512)
```

## Architecture

```
PDFs → PyMuPDF → Chunks(512 chars, 50 overlap)
     → SentenceTransformer embeds → FAISS IndexFlatIP

Query → embed → FAISS.search(top_k=5)
      → retrieved chunks → Ollama prompt → Answer + [citations]
```

## Files

```
app.py              Gradio UI
rag/
  ingestion.py      PDF → chunks → FAISS index
  retriever.py      Query embed + similarity search
  generator.py      Ollama prompt builder + caller
requirements.txt
```