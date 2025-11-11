# LovAI – Norwegian Legal AI Frontend

LovAI is a production-ready frontend for the Lovdata Retrieval-Augmented Generation (RAG) backend. It helps
legal professionals explore Norwegian legislation through hybrid dense + lexical search, reranking, and
citation-backed answers.

## ✨ Highlights

- **Responsive React UI** built with Vite + TypeScript, optimized for fast local iteration and static hosting.
- **Hybrid RAG controls**: toggle dense/lexical hybrid retrieval, reranking, citation enforcement, and top-k.
- **Rich result experience**: grounded answer view, supporting evidence, citation list with deep links, and
  live latency metrics.
- **Resilient networking**: aborts in-flight requests on new submissions, surfaces backend errors, and normalizes
  metrics coming from FastAPI.

## 🧱 Repository layout

```
frontend/       # Vite + React application
└─ src/
   ├─ components/   # Form, results, citation UI building blocks
   ├─ hooks/        # API integrations using TanStack Query
   ├─ lib/          # Typed client for the Lovdata RAG API
   └─ styles.css    # Global styling
```

Backend code for the Lovdata RAG stack lives in the separate `lovdata_rag` Python package and FastAPI server as
mentioned in the project brief. Point this frontend at that backend via `VITE_API_BASE_URL`.

## 🚀 Getting started

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Configure environment

Create a `.env` file inside `frontend/` to point at the running FastAPI instance:

```
VITE_API_BASE_URL=http://localhost:8000
```

The backend should expose at least `POST /search` with the payload described in the project brief.

### 3. Run locally

```bash
npm run dev
```

Open the printed URL (defaults to [http://localhost:5173](http://localhost:5173)).

### 4. Build for production

```bash
npm run build
```

The static site is emitted to `frontend/dist/` and can be served by any CDN or static host. To preview the
production build locally:

```bash
npm run preview
```

## 🧪 Testing the backend connection

The UI calls `POST /search` with the following JSON shape:

```json
{
  "query": "ansvar for styret",
  "top_k": 10,
  "use_hybrid": true,
  "with_citations": true,
  "rerank": true
}
```

Responses are expected to contain at least `answer`, with optional `citations`, `hits`, and `metrics` objects.
Any snake_case latency values are normalized automatically. Backend error payloads bubble up to the UI so that
operational issues surface quickly.

## 📦 Deployment notes

- Ship the built assets (`frontend/dist`) behind a CDN or static host (e.g., Vercel, Netlify, CloudFront).
- Configure the backend origin URL via `VITE_API_BASE_URL` at deploy time.
- For Docker-based hosting, add a Node build stage to compile the frontend and serve the static bundle with
  Nginx or another HTTP server.

## 📝 License

MIT License – see [LICENSE](./LICENSE) if present in the parent repository.
