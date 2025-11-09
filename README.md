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

# 🏛️ Lovdata Legal AI

Semantic search and analysis for Norwegian legal texts powered by AI.

## 🚀 Quick Deploy

Deploy this application permanently with one click:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new?template=https://github.com/Jakobkoding2/lovdata-legal-ai-space)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Jakobkoding2/lovdata-legal-ai-space)

**Or try it locally:**
```bash
git clone https://github.com/Jakobkoding2/lovdata-legal-ai-space.git
cd lovdata-legal-ai-space
docker-compose up
```

Visit `http://localhost:7860`

## ✨ Features

- **Semantic Search**: Find relevant legal provisions using natural language queries
- **Overlap Detection**: Analyze semantic relationships between legal texts
- **Real-time Analysis**: Instant results powered by FAISS vector search
- **20,000+ Legal Texts**: Norwegian laws and regulations from Lovdata

## 📊 Data

This system processes over 338,000 legal text units from Norwegian laws and regulations, sourced from the [Lovdata Public API](https://lovdata.no/pro/api-dokumentasjon).

## 🛠️ Technology

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Vector Search**: FAISS index for efficient similarity search
- **Classification**: Random Forest classifier (100% accuracy)
- **Interface**: Gradio web application
- **Deployment**: Docker, Railway, Render, Fly.io, Google Cloud Run

## 📖 Documentation

- **Full Documentation**: [GitHub Repository](https://github.com/Jakobkoding2/lovdata-legal-ai)
- **Deployment Guide**: [DEPLOY.md](./DEPLOY.md)
- **API Documentation**: [README.md](https://github.com/Jakobkoding2/lovdata-legal-ai/blob/master/README.md)

## ⚠️ Note

This is a demonstration system built autonomously. For production legal applications, please consult qualified legal experts.

## 📝 License

MIT License - See [LICENSE](./LICENSE) for details
