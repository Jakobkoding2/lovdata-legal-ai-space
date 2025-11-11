import { ReactNode } from "react";
import { CitationList } from "./CitationList";
import { SearchResponse } from "../lib/api";

type State = {
  data: SearchResponse | undefined;
  isLoading: boolean;
  hasAttempted: boolean;
};

type Props = {
  state: State;
};

export function ResultsPanel({ state }: Props) {
  if (!state.hasAttempted) {
    return (
      <EmptyState>
        <p>
          Start med å skrive et juridisk spørsmål. Systemet søker i Lovdata-korpuset og returnerer et
          begrunnet svar med sitater.
        </p>
      </EmptyState>
    );
  }

  if (state.isLoading) {
    return (
      <EmptyState>
        <div className="loader" aria-live="polite" aria-busy="true">
          Laster resultater...
        </div>
      </EmptyState>
    );
  }

  if (!state.data) {
    return (
      <EmptyState>
        <p>Ingen resultater tilgjengelig ennå. Prøv å justere søkeparametrene.</p>
      </EmptyState>
    );
  }

  const { answer, citations, hits, metrics } = state.data;

  return (
    <div className="results">
      <div className="results__answer">
        <h2>Svar</h2>
        {answer ? <p>{answer}</p> : <p>Backend returnerte ikke noe svar.</p>}
      </div>

      <div className="results__meta">
        {metrics && (
          <ul className="metrics">
            {metrics.latencyMs && (
              <li>
                <span>Ende-til-ende latenstid</span>
                <strong>{Math.round(metrics.latencyMs)} ms</strong>
              </li>
            )}
            {metrics.retrievalLatencyMs && (
              <li>
                <span>Retriever</span>
                <strong>{Math.round(metrics.retrievalLatencyMs)} ms</strong>
              </li>
            )}
            {metrics.rerankLatencyMs && (
              <li>
                <span>Reranker</span>
                <strong>{Math.round(metrics.rerankLatencyMs)} ms</strong>
              </li>
            )}
            {typeof metrics.hybridUsed === "boolean" && (
              <li>
                <span>Hybrid-søk</span>
                <strong>{metrics.hybridUsed ? "Aktivert" : "Deaktivert"}</strong>
              </li>
            )}
            {typeof metrics.hitCount === "number" && (
              <li>
                <span>Treff</span>
                <strong>{metrics.hitCount}</strong>
              </li>
            )}
          </ul>
        )}
      </div>

      <div className="results__hits">
        <h3>Grunnlag og sitater</h3>
        {hits && hits.length > 0 ? (
          <ul className="hit-list">
            {hits.map((hit) => (
              <li key={hit.id ?? `${hit.lawId}-${hit.chunkId ?? hit.rank}` } className="hit-card">
                <header className="hit-card__header">
                  <div>
                    <p className="hit-card__title">{hit.title ?? hit.section ?? "Ukjent kilde"}</p>
                    <p className="hit-card__meta">
                      {hit.lawId && <span>Lov: {hit.lawId}</span>}
                      {typeof hit.score === "number" && <span>Score: {hit.score.toFixed(3)}</span>}
                    </p>
                  </div>
                  {hit.url && (
                    <a className="link" href={hit.url} target="_blank" rel="noreferrer">
                      Åpne
                    </a>
                  )}
                </header>
                <p>{hit.snippet ?? hit.text ?? "Ingen utdrag tilgjengelig."}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p>Ingen underliggende utdrag ble returnert.</p>
        )}
      </div>

      <CitationList citations={citations ?? []} />
    </div>
  );
}

function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}
