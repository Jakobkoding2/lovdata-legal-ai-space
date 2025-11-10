#!/usr/bin/env python3
import os
import threading

try:
    import gradio as gr
except Exception:
    raise RuntimeError("gradio is required. Install with `pip install gradio`")

import requests

# Configuration
PORT = int(os.environ.get("PORT", 7860))
HOST = "0.0.0.0"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "")
USE_HF_INFERENCE = bool(HF_API_TOKEN) and not LOCAL_MODEL_PATH

_model_lock = threading.Lock()
model = None

def load_local_model():
    global model
    with _model_lock:
        if model is not None:
            return
        from transformers import pipeline
        model_name = LOCAL_MODEL_PATH or "facebook/bart-large-cnn"
        model = pipeline("summarization", model=model_name, device=-1)

def hf_inference(text: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{{LOCAL_MODEL_PATH or 'facebook/bart-large-cnn'}}"
    headers = {"Authorization": f"Bearer {{HF_API_TOKEN}}"}
    payload = {"inputs": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        return out[0].get("summary_text") or str(out[0])
    return str(out)

def predict(text: str) -> str:
    if USE_HF_INFERENCE:
        try:
            return hf_inference(text)
        except Exception as e:
            return f"HF inference failed: {e}"
    else:
        try:
            if model is None:
                load_local_model()
            out = model(text, max_length=200, truncation=True)
            if isinstance(out, list) and len(out) > 0:
                return out[0].get("summary_text", str(out[0]))
            return str(out)
        except Exception as e:
            return f"Local model failed: {e}"

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LOVDATA Legal AI — minimal demo")
        inp = gr.Textbox(label="Input text", lines=6, placeholder="Paste text here...")
        out = gr.Textbox(label="Output", interactive=False)
        btn = gr.Button("Run")
        btn.click(fn=predict, inputs=inp, outputs=out)
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name=HOST, server_port=PORT, inbrowser=False, share=False, max_threads=1)