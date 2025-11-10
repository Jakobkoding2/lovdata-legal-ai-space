#!/usr/bin/env python3
import os
import gradio as gr
import requests

PORT = int(os.environ.get("PORT", 7860))
HOST = "0.0.0.0"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "facebook/bart-large-cnn")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN env var is required for HF Inference API mode. Set it in Render.")

def hf_inference(text: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        return out[0].get("summary_text") or str(out[0])
    return str(out)

def predict(text: str) -> str:
    try:
        return hf_inference(text)
    except Exception as e:
        return f"HF inference failed: {e}"

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LOVDATA Legal AI — HF Inference API")
        inp = gr.Textbox(label="Input text", lines=6, placeholder="Paste text here...")
        out = gr.Textbox(label="Output", interactive=False)
        btn = gr.Button("Run")
        btn.click(fn=predict, inputs=inp, outputs=out)
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name=HOST, server_port=PORT, inbrowser=False, share=False, max_threads=1)
