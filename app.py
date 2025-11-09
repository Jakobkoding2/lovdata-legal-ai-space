#!/usr/bin/env python3
"""
Lovdata Legal AI – Gradio interface
Semantic search and overlap analysis for Norwegian legal texts.
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import gradio as gr

# Optional deps: installed via requirements.txt
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import joblib
except ImportError:
    print("Installing required packages at runtime...")
    os.system("pip install -q sentence-transformers faiss-cpu scikit-learn joblib pandas pyarrow")
    from sentence_transformers import SentenceTransformer
    import faiss
    import joblib

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Globals
corpus_df = None
embeddings = None
faiss_index = None
embedding_model = None
overlap_classifier = None


def load_models():
    """Load models and any local artifacts if present. Tolerant to missing files."""
    global corpus_df, embeddings, faiss_index, embedding_model, overlap_classifier

    print("Loading models and data...")

    corpus_df = None
    embeddings = None
    faiss_index = None
    overlap_classifier = None

    # Try local corpus
    corpus_path = DATA_DIR / "lovdata_corpus.parquet"
    if corpus_path.exists():
        try:
            # Limit for demo to reduce RAM
            corpus_df = pd.read_parquet(corpus_path).head(20000)
            print(f"✓ Loaded corpus: {len(corpus_df)} texts")
        except Exception as e:
            print(f"Corpus load skipped: {e}")

    # Try local embeddings
    emb_path = MODELS_DIR / "lovdata_embeddings.npy"
    if emb_path.exists():
        try:
            embeddings = np.load(emb_path)
            print(f"✓ Loaded embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"Embeddings load skipped: {e}")

    # Try local FAISS index
    index_path = MODELS_DIR / "lovdata_faiss.index"
    if index_path.exists():
        try:
            faiss_index = faiss.read_index(str(index_path))
            print(f"✓ Loaded FAISS index: {faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"Index load skipped: {e}")

    # Embedding model from HF (CPU)
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("✓ Loaded embedding model")

    # Optional classifier
    clf_path = MODELS_DIR / "overlap_classifier.joblib"
    if clf_path.exists():
        try:
            overlap_classifier = joblib.load(clf_path)
            print("✓ Loaded overlap classifier")
        except Exception as e:
            print(f"Classifier load skipped: {e}")

    print("Models ready.")


def semantic_search(query: str, top_k: int = 5, min_similarity: float = 0.7):
    """Semantic search using FAISS index + corpus."""
    if embedding_model is None:
        return "⚠️ Embedding model not loaded."
    if faiss_index is None or corpus_df is None:
        return "⚠️ Index or corpus not loaded. Last ned/bygg index først."

    # Query embedding
    q_emb = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    # Search
    distances, indices = faiss_index.search(q_emb, top_k * 2)
    sims = distances[0]

    results = []
    for idx, sim in zip(indices[0], sims):
        if idx < 0 or idx >= len(corpus_df):
            continue
        if sim < min_similarity:
            continue

        row = corpus_df.iloc[idx]
        text = row.get("text_clean") or row.get("text") or ""
        snippet = (text[:300] + "...") if len(text) > 300 else text

        results.append(
            {
                "Similarity": f"{sim:.2%}",
                "Document": row.get("doc_title", "Ukjent"),
                "Section": row.get("section_num", "N/A"),
                "Type": row.get("group", "N/A"),
                "Text": snippet,
            }
        )
        if len(results) >= top_k:
            break

    if not results:
        return "Ingen treff. Prøv et annet søk eller senk terskelen."

    out = [f"### Fant {len(results)} treff for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        out.append(
            f"**{i}. {r['Document']} § {r['Section']}** ({r['Type']}) "
            f"- Similaritet: {r['Similarity']}\n\n> {r['Text']}\n\n---\n"
        )
    return "".join(out)


def detect_overlap(text1: str, text2: str):
    """Classify semantic overlap between two texts."""
    if embedding_model is None:
        return "⚠️ Embedding model ikke lastet."
    if not text1 or not text2:
        return "Skriv inn begge tekster."

    emb_pair = embedding_model.encode(
        [text1, text2], convert_to_numpy=True, normalize_embeddings=True
    )
    similarity = float(np.dot(emb_pair[0], emb_pair[1]))

    # Basic features
    len1, len2 = len(text1), len(text2)
    words1, words2 = set(text1.lower().split()), set(text2.lower().split())
    features = {
        "similarity": similarity,
        "len_ratio": min(len1, len2) / max(len1, len2) if max(len1, len2) else 0.0,
        "len_diff": abs(len1 - len2),
        "word_overlap": len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0.0,
        "same_doc": 0,
        "cross_group": 0,
        "avg_length": (len1 + len2) / 2.0,
        "max_length": max(len1, len2),
        "min_length": min(len1, len2),
    }

    if overlap_classifier is None:
        # Heuristic if classifier missing
        label = "duplicate" if similarity >= 0.92 else ("subsumption" if similarity >= 0.80 else "different")
        return (
            "### Overlap Analysis\n\n"
            f"**Similarity Score**: {similarity:.2%}\n\n"
            f"**Overlap Type**: {label.upper()}\n\n"
            "*(Klassifiseringsmodell ikke lastet. Heuristikk brukt.)*"
        )

    clf = overlap_classifier["classifier"]
    feature_names = overlap_classifier["feature_names"]
    X = np.array([[features[col] for col in feature_names]])
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]

    lines = [
        "### Overlap Analysis\n",
        f"**Similarity Score**: {similarity:.2%}\n\n",
        f"**Overlap Type**: {pred.upper()}\n\n",
        "**Probabilities:**\n",
    ]
    for label, p in zip(clf.classes_, probs):
        lines.append(f"- {label}: {p:.2%}\n")

    lines.append("\n**Interpretation**:\n")
    if pred == "duplicate":
        lines.append("✅ Tekstene er nesten identiske. Mulig duplisering.")
    elif pred == "subsumption":
        lines.append("⚠️ Den ene teksten impliserer/omfatter den andre. Mulig subsumsjon.")
    else:
        lines.append("ℹ️ Tekstene er semantisk ulike.")
    return "".join(lines)


def get_statistics():
    """Return simple system stats."""
    lines = ["### System Statistics\n\n"]
    if corpus_df is None:
        lines.append("Modeller/korpos ikke lastet.\n")
    else:
        laws = int((corpus_df.get("group") == "law").sum()) if "group" in corpus_df else "N/A"
        regs = int((corpus_df.get("group") == "regulation").sum()) if "group" in corpus_df else "N/A"
        uniq_docs = int(corpus_df.get("doc_id").nunique()) if "doc_id" in corpus_df else "N/A"
        lines += [
            "**Corpus Information:**\n",
            f"- Total texts: {len(corpus_df):,}\n",
            f"- Laws: {laws}\n",
            f"- Regulations: {regs}\n",
            f"- Unique documents: {uniq_docs}\n\n",
        ]

    dim = embeddings.shape[1] if isinstance(embeddings, np.ndarray) else "N/A"
    vecs = embeddings.shape[0] if isinstance(embeddings, np.ndarray) else "N/A"
    indexed = faiss_index.ntotal if faiss_index is not None else "N/A"
    lines += [
        "**Embeddings:**\n",
        f"- Dimension: {dim}\n",
        f"- Total vectors: {vecs}\n\n",
        "**Index:**\n",
        "- Type: FAISS Flat (exact search)\n",
        f"- Vectors indexed: {indexed}\n",
    ]
    return "".join(lines)


# Load everything on startup
load_models()

# UI
with gr.Blocks(title="Lovdata Legal AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🏛️ Lovdata Legal AI

Semantisk søk og analyse for norske lover og forskrifter.

**Datakilde**: Lovdata Public API  
**Repo**: https://github.com/Jakobkoding2/lovdata-legal-ai
"""
    )

    with gr.Tabs():
        with gr.Tab("🔍 Semantic Search"):
            gr.Markdown("Søk i lover/forskrifter med naturlig språk.")
            with gr.Row():
                with gr.Column():
                    q = gr.Textbox(
                        label="Search Query",
                        placeholder="f.eks. 'ansvar for styret' eller 'kontrakt og forpliktelser'",
                        lines=2,
                    )
                    with gr.Row():
                        topk = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
                        minsim = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Minimum Similarity")
                    btn = gr.Button("Search", variant="primary")
            out = gr.Markdown()
            btn.click(semantic_search, inputs=[q, topk, minsim], outputs=out)
            gr.Examples(
                examples=[
                    ["ansvar for styret", 5, 0.7],
                    ["kontrakt og forpliktelser", 5, 0.7],
                    ["forsikring og erstatning", 5, 0.7],
                ],
                inputs=[q, topk, minsim],
            )

        with gr.Tab("🔄 Overlap Detection"):
            gr.Markdown("Analyser semantisk overlapp mellom to tekster.")
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="Text 1", lines=5)
                with gr.Column():
                    t2 = gr.Textbox(label="Text 2", lines=5)
            btn2 = gr.Button("Analyze Overlap", variant="primary")
            out2 = gr.Markdown()
            btn2.click(detect_overlap, inputs=[t1, t2], outputs=out2)

        with gr.Tab("📊 Statistics"):
            gr.Markdown("Systeminfo og statistikk.")
            btn3 = gr.Button("Refresh Statistics")
            out3 = gr.Markdown(value=get_statistics())
            btn3.click(get_statistics, inputs=[], outputs=out3)

    gr.Markdown(
        """
---
**Merk**: Demo-system. Ikke juridisk rådgivning.
"""
    )

if __name__ == "__main__":
    # Bind to Render/Railway provided port
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
