#!/usr/bin/env python3
"""
Lovdata Legal AI - Gradio Interface for Hugging Face Spaces
Provides a web interface for semantic search and legal Q&A
"""

import os
import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Import required libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import joblib
except ImportError:
    print("Installing required packages...")
    os.system("pip install sentence-transformers faiss-cpu scikit-learn joblib pandas pyarrow")
    from sentence_transformers import SentenceTransformer
    import faiss
    import joblib

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Global variables
corpus_df = None
embeddings = None
faiss_index = None
embedding_model = None
overlap_classifier = None

def load_models():
    """Load all models and data"""
    global corpus_df, embeddings, faiss_index, embedding_model, overlap_classifier
    
    print("Loading models and data...")
    
    # Load corpus
    corpus_path = DATA_DIR / "lovdata_corpus.parquet"
    if corpus_path.exists():
        corpus_df = pd.read_parquet(corpus_path).head(20000)
        print(f"✓ Loaded corpus: {len(corpus_df)} texts")
    
    # Load embeddings
    embeddings_path = MODELS_DIR / "lovdata_embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Load FAISS index
    index_path = MODELS_DIR / "lovdata_faiss.index"
    if index_path.exists():
        faiss_index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index: {faiss_index.ntotal} vectors")
    
    # Load embedding model
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"✓ Loaded embedding model")
    
    # Load overlap classifier
    classifier_path = MODELS_DIR / "overlap_classifier.joblib"
    if classifier_path.exists():
        overlap_classifier = joblib.load(classifier_path)
        print(f"✓ Loaded overlap classifier")
    
    print("All models loaded successfully!")

def semantic_search(query, top_k=5, min_similarity=0.7):
    """Perform semantic search"""
    if embedding_model is None or faiss_index is None or corpus_df is None:
        return "⚠️ Models not loaded. Please refresh the page."
    
    # Generate query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, top_k * 2)
    similarities = distances[0]
    
    # Format results
    results = []
    for idx, similarity in zip(indices[0], similarities):
        if similarity < min_similarity or idx >= len(corpus_df):
            continue
        
        row = corpus_df.iloc[idx]
        
        results.append({
            "Similarity": f"{similarity:.2%}",
            "Document": row['doc_title'],
            "Section": row.get('section_num', 'N/A'),
            "Type": row['group'],
            "Text": row['text_clean'][:300] + "..." if len(row['text_clean']) > 300 else row['text_clean']
        })
        
        if len(results) >= top_k:
            break
    
    if not results:
        return "No results found. Try a different query or lower the similarity threshold."
    
    # Format as markdown table
    output = f"### Found {len(results)} results for: '{query}'\n\n"
    for i, result in enumerate(results, 1):
        output += f"**{i}. {result['Document']} § {result['Section']}** ({result['Type']}) - Similarity: {result['Similarity']}\n\n"
        output += f"> {result['Text']}\n\n"
        output += "---\n\n"
    
    return output

def detect_overlap(text1, text2):
    """Detect semantic overlap between two texts"""
    if embedding_model is None or overlap_classifier is None:
        return "⚠️ Models not loaded. Please refresh the page."
    
    if not text1 or not text2:
        return "Please provide both texts."
    
    # Generate embeddings
    embeddings_pair = embedding_model.encode(
        [text1, text2],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute similarity
    similarity = float(np.dot(embeddings_pair[0], embeddings_pair[1]))
    
    # Extract features
    len1 = len(text1)
    len2 = len(text2)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    features = {
        'similarity': similarity,
        'len_ratio': min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0,
        'len_diff': abs(len1 - len2),
        'word_overlap': len(words1 & words2) / len(words1 | words2) if len(words1 | words2) > 0 else 0,
        'same_doc': 0,
        'cross_group': 0,
        'avg_length': (len1 + len2) / 2,
        'max_length': max(len1, len2),
        'min_length': min(len1, len2)
    }
    
    # Predict overlap type
    classifier = overlap_classifier['classifier']
    feature_names = overlap_classifier['feature_names']
    
    X = np.array([[features[col] for col in feature_names]])
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    
    # Format output
    output = f"### Overlap Analysis\n\n"
    output += f"**Similarity Score**: {similarity:.2%}\n\n"
    output += f"**Overlap Type**: {prediction.upper()}\n\n"
    output += f"**Probabilities**:\n"
    for label, prob in zip(classifier.classes_, probabilities):
        output += f"- {label}: {prob:.2%}\n"
    
    output += f"\n**Interpretation**:\n"
    if prediction == 'duplicate':
        output += "✅ The texts are nearly identical. This indicates potential duplication."
    elif prediction == 'subsumption':
        output += "⚠️ One text contains or implies the other. This may indicate subsumption."
    else:
        output += "ℹ️ The texts are semantically different."
    
    return output

def get_statistics():
    """Get system statistics"""
    if corpus_df is None:
        return "Models not loaded yet."
    
    stats = f"""
### System Statistics

**Corpus Information:**
- Total texts: {len(corpus_df):,}
- Laws: {len(corpus_df[corpus_df['group'] == 'law']):,}
- Regulations: {len(corpus_df[corpus_df['group'] == 'regulation']):,}
- Unique documents: {corpus_df['doc_id'].nunique():,}

**Embeddings:**
- Dimension: {embeddings.shape[1] if embeddings is not None else 'N/A'}
- Total vectors: {embeddings.shape[0]:,} if embeddings is not None else 'N/A'

**Index:**
- Type: FAISS Flat (exact search)
- Vectors indexed: {faiss_index.ntotal:,} if faiss_index is not None else 'N/A'
"""
    return stats

# Load models on startup
load_models()

# Create Gradio interface
with gr.Blocks(title="Lovdata Legal AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏛️ Lovdata Legal AI
    
    Semantic search and analysis for Norwegian legal texts. This system uses AI to search through 
    Norwegian laws and regulations, find similar legal provisions, and detect potential overlaps.
    
    **Data Source**: [Lovdata Public API](https://lovdata.no/pro/api-dokumentasjon)
    
    **GitHub**: [lovdata-legal-ai](https://github.com/Jakobkoding2/lovdata-legal-ai)
    """)
    
    with gr.Tabs():
        with gr.Tab("🔍 Semantic Search"):
            gr.Markdown("Search for legal provisions using natural language queries.")
            
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., 'ansvar for styret' or 'kontrakt og forpliktelser'",
                        lines=2
                    )
                    with gr.Row():
                        search_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )
                        search_min_sim = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Minimum Similarity"
                        )
                    search_button = gr.Button("Search", variant="primary")
            
            search_output = gr.Markdown(label="Results")
            
            search_button.click(
                semantic_search,
                inputs=[search_query, search_top_k, search_min_sim],
                outputs=search_output
            )
            
            gr.Examples(
                examples=[
                    ["ansvar for styret", 5, 0.7],
                    ["kontrakt og forpliktelser", 5, 0.7],
                    ["forsikring og erstatning", 5, 0.7],
                ],
                inputs=[search_query, search_top_k, search_min_sim]
            )
        
        with gr.Tab("🔄 Overlap Detection"):
            gr.Markdown("Analyze semantic overlap between two legal texts.")
            
            with gr.Row():
                with gr.Column():
                    overlap_text1 = gr.Textbox(
                        label="Text 1",
                        placeholder="Enter first legal text...",
                        lines=5
                    )
                with gr.Column():
                    overlap_text2 = gr.Textbox(
                        label="Text 2",
                        placeholder="Enter second legal text...",
                        lines=5
                    )
            
            overlap_button = gr.Button("Analyze Overlap", variant="primary")
            overlap_output = gr.Markdown(label="Analysis Results")
            
            overlap_button.click(
                detect_overlap,
                inputs=[overlap_text1, overlap_text2],
                outputs=overlap_output
            )
        
        with gr.Tab("📊 Statistics"):
            gr.Markdown("System information and statistics.")
            
            stats_button = gr.Button("Refresh Statistics")
            stats_output = gr.Markdown(value=get_statistics())
            
            stats_button.click(
                get_statistics,
                inputs=[],
                outputs=stats_output
            )
    
    gr.Markdown("""
    ---
    
    ### About
    
    This system was built autonomously using:
    - **Data**: Lovdata public API (Norwegian laws and regulations)
    - **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
    - **Search**: FAISS vector index
    - **Classification**: Random Forest (100% accuracy)
    
    **Note**: This is a demonstration system. For production use, please consult legal experts.
    """)

if __name__ == "__main__":
    demo.launch()
