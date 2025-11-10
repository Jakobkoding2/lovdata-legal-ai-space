#!/usr/bin/env python3
import os
import gradio as gr
import requests

# Bind to Render's provided port and 0.0.0.0 so the service is discoverable
PORT = int(os.environ.get("PORT", 7860))
HOST = "0.0.0.0"

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/ask_law")


def fetch_answer(user_input: str) -> str:
    try:
        payload = {"question": user_input}
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "")
    except Exception as exc:
        return f"Backend request failed: {exc}"


def predict(text: str) -> str:
    if not text.strip():
        return ""
    return fetch_answer(text)


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LOVDATA Legal AI — Backend Router")
        inp = gr.Textbox(label="Input text", lines=6, placeholder="Paste text here...")
        out = gr.Textbox(label="Output", interactive=False)
        btn = gr.Button("Run")
        btn.click(fn=predict, inputs=inp, outputs=out)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name=HOST, server_port=PORT, inbrowser=False, share=False, max_threads=1)
