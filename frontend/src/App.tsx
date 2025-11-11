import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { SearchForm, SearchParams } from "./components/SearchForm";
import { ResultsPanel } from "./components/ResultsPanel";
import { useLovSearch } from "./hooks/useLovSearch";

const defaultParams: SearchParams = {
  query: "",
  topK: 5,
  useHybrid: true,
  withCitations: true,
  rerank: true,
};

function App() {
  const [params, setParams] = useState<SearchParams>(defaultParams);
  const { data, isPending, error, runSearch, lastFetchedAt } = useLovSearch();

  const handleSubmit = (values: SearchParams) => {
    setParams(values);
    if (values.query.trim().length === 0) return;
    runSearch(values);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <img src="/lovai.svg" alt="LovAI" className="app-logo" />
          <div>
            <h1>LovAI</h1>
            <p>Lovdata-backed legal research assistant for Norway</p>
          </div>
        </div>
        {data?.metrics?.latencyMs && (
          <div className="badge badge--neutral" aria-live="polite">
            <span>Latency</span>
            <strong>{Math.round(data.metrics.latencyMs)} ms</strong>
          </div>
        )}
        {lastFetchedAt && (
          <div className="badge badge--outline" aria-live="polite">
            <span>Last updated</span>
            <strong>{formatDistanceToNow(lastFetchedAt, { addSuffix: true })}</strong>
          </div>
        )}
      </header>

      <main className="app-main">
        <section className="panel">
          <SearchForm defaultValues={params} isLoading={isPending} onSubmit={handleSubmit} />
          {error && <p className="error-banner">{error.message}</p>}
        </section>

        <section className="panel panel--results" aria-live="polite">
          <ResultsPanel
            state={{ data, isLoading: isPending, hasAttempted: params.query.trim().length > 0 }}
          />
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Powered by a hybrid dense + lexical retrieval stack with reranking and citation enforcement.
        </p>
        <p>
          Backend URL: <code>{import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}</code>
        </p>
      </footer>
    </div>
  );
}

export default App;
