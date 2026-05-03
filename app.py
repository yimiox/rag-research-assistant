"""
app.py — Gradio 6 compatible RAG Research Assistant
"""

import os
import gradio as gr
from rag import ingest_pdfs, load_index, retrieve, generate_answer
from rag.generator import DEFAULT_MODEL

_index  = None
_chunks = None

def _load_existing():
    global _index, _chunks
    _index, _chunks = load_index()
    if _index:
        sources = set(c["source"] for c in _chunks)
        return f"Loaded existing index: {len(_chunks)} chunks from {len(sources)} doc(s)."
    return "No existing index. Upload PDFs to begin."

def handle_upload(files, append_mode):
    global _index, _chunks
    if not files:
        return "No files uploaded."
    paths = [f.name for f in files]
    _index, _chunks, msg = ingest_pdfs(paths, append=append_mode)
    return msg

def handle_query(question, top_k, model_name):
    global _index, _chunks
    if _index is None:
        return "Please ingest documents first.", ""
    if not question.strip():
        return "Please enter a question.", ""
    chunks = retrieve(question, _index, _chunks, top_k=int(top_k))
    answer, citations = generate_answer(question, chunks, model=model_name)
    return answer, citations

def handle_clear():
    global _index, _chunks
    _index = _chunks = None
    for f in ["faiss_index", "faiss_meta.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    return "Index cleared."

CSS = """
#answer-box textarea {font-size: 15px; line-height: 1.7;}
"""

with gr.Blocks(title="RAG Research Assistant") as demo:
    gr.Markdown("""
# 📚 RAG Research Assistant
**Local, offline, cited.** Upload PDFs → ask questions → get answers with source citations.
> FAISS · HuggingFace Embeddings · Ollama
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Upload Documents")
            file_input    = gr.File(label="PDF files", file_types=[".pdf"], file_count="multiple")
            append_toggle = gr.Checkbox(label="Append to existing index", value=False)
            ingest_btn    = gr.Button("Ingest Documents", variant="primary")
            ingest_status = gr.Textbox(label="Status", lines=2)
            clear_btn     = gr.Button("Clear Index", variant="stop")
            gr.Markdown("### Settings")
            top_k_slider  = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K chunks")
            model_input   = gr.Textbox(value=DEFAULT_MODEL, label="Ollama model name",
                                       placeholder="llama3 / mistral / phi3")

        with gr.Column(scale=2):
            gr.Markdown("### 2 · Ask Questions")
            question_input = gr.Textbox(label="Your question",
                                        placeholder="What is the main contribution of this paper?",
                                        lines=3)
            ask_btn        = gr.Button("Ask", variant="primary")
            answer_box     = gr.Textbox(label="Answer", lines=12, elem_id="answer-box")
            citations_box  = gr.Markdown(label="Citations")

    ingest_btn.click(handle_upload, inputs=[file_input, append_toggle], outputs=[ingest_status])
    clear_btn.click(handle_clear, outputs=[ingest_status])
    ask_btn.click(handle_query, inputs=[question_input, top_k_slider, model_input],
                  outputs=[answer_box, citations_box])
    question_input.submit(handle_query, inputs=[question_input, top_k_slider, model_input],
                          outputs=[answer_box, citations_box])
    demo.load(fn=_load_existing, outputs=[ingest_status])

    gr.Markdown("""
---
**Setup:** `ollama serve` + `ollama pull llama3` · **HF Spaces:** set `HF_TOKEN` secret
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )