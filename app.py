#!/usr/bin/env python3
import os
import gradio as gr
import requests

# Bind to Render's provided port and 0.0.0.0 so the service is discoverable
PORT = int(os.environ.get("PORT", 7860))
HOST = "0.0.0.0"

# Required env vars
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "facebook/bart-large-cnn")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN env var is required for HF Inference API mode. Set it in Render.")

# Use the new router endpoint per HF docs
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

def hf_inference(text: str) -> str:
    url = f"{HF_ROUTER_BASE}/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"inputs": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = resp.text
        raise RuntimeError(f"HF router request failed: {e}. Response body: {body}")
    # Defensive JSON parsing
    try:
        out = resp.json()
    except ValueError:
        raise RuntimeError(f"HF router returned non-JSON response: {resp.text}")
    # Handle common response shapes
    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict):
            for key in ("generated_text", "summary_text", "translation_text", "text", "label"):
                if key in first:
                    return first.get(key)
            return str(first)
        return str(first)
    if isinstance(out, dict):
        if "error" in out:
            raise RuntimeError(f"HF router returned error: {out['error']}")
        for key in ("generated_text", "summary_text", "translation_text", "text", "label"):
            if key in out:
                return out.get(key)
        return str(out)
    return str(out)

def predict(text: str) -> str:
    try:
        return hf_inference(text)
    except Exception as e:
        return f"HF inference failed: {e}"

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LOVDATA Legal AI — HF Inference (router)")
        inp = gr.Textbox(label="Input text", lines=6, placeholder="Paste text here...")
        out = gr.Textbox(label="Output", interactive=False)
        btn = gr.Button("Run")
        btn.click(fn=predict, inputs=inp, outputs=out)
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name=HOST, server_port=PORT, inbrowser=False, share=False, max_threads=1)
