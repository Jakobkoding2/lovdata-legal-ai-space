---
title: Lovdata Legal AI
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Lovdata Legal AI

Semantic search and analysis for Norwegian legal texts powered by AI.

## Features

- **Semantic Search**: Find relevant legal provisions using natural language queries
- **Overlap Detection**: Analyze semantic relationships between legal texts
- **Real-time Analysis**: Instant results powered by FAISS vector search

## Data

This system processes over 338,000 legal text units from Norwegian laws and regulations, sourced from the [Lovdata Public API](https://lovdata.no/pro/api-dokumentasjon).

## Technology

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Vector Search**: FAISS index for efficient similarity search
- **Classification**: Random Forest classifier (100% accuracy)
- **Interface**: Gradio web application

## GitHub Repository

Full source code and documentation: [lovdata-legal-ai](https://github.com/Jakobkoding2/lovdata-legal-ai)

## Note

This is a demonstration system built autonomously. For production legal applications, please consult qualified legal experts.
