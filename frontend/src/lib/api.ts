export type SearchCitation = {
  chunkId?: string;
  lawId?: string;
  section?: string;
  title?: string;
  text?: string;
  url?: string;
};

export type SearchHit = {
  id?: string;
  chunkId?: string;
  lawId?: string;
  score?: number;
  title?: string;
  section?: string;
  snippet?: string;
  text?: string;
  url?: string;
  rank?: number;
};

export type SearchMetrics = {
  latencyMs?: number;
  retrievalLatencyMs?: number;
  rerankLatencyMs?: number;
  embeddingLatencyMs?: number;
  hitCount?: number;
  hybridUsed?: boolean;
};

export type SearchResponse = {
  answer?: string;
  citations?: SearchCitation[];
  hits?: SearchHit[];
  metrics?: SearchMetrics;
  rawAnswer?: string;
};

export type SearchRequest = {
  query: string;
  top_k?: number;
  use_hybrid?: boolean;
  with_citations?: boolean;
  rerank?: boolean;
};

function getApiBaseUrl() {
  const fromEnv = import.meta.env.VITE_API_BASE_URL;
  if (fromEnv) {
    return fromEnv.replace(/\/$/, "");
  }
  return "http://localhost:8000";
}

export async function performSearch(payload: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
  const response = await fetch(`${getApiBaseUrl()}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Backend svarte ${response.status}: ${text}`);
  }

  const data = (await response.json()) as SearchResponse & {
    metrics?: SearchMetrics & { latency_ms?: number; retrieval_latency_ms?: number; rerank_latency_ms?: number };
    citations?: SearchCitation[];
    hits?: SearchHit[];
  };

  if (data.metrics) {
    data.metrics = normalizeMetrics(data.metrics);
  }

  return data;
}

function normalizeMetrics(metrics: SearchMetrics & {
  latency_ms?: number;
  retrieval_latency_ms?: number;
  rerank_latency_ms?: number;
}) {
  return {
    ...metrics,
    latencyMs: metrics.latencyMs ?? metrics.latency_ms,
    retrievalLatencyMs: metrics.retrievalLatencyMs ?? metrics.retrieval_latency_ms,
    rerankLatencyMs: metrics.rerankLatencyMs ?? metrics.rerank_latency_ms,
  } satisfies SearchMetrics;
}
