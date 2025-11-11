import { SearchCitation } from "../lib/api";

type Props = {
  citations: SearchCitation[];
};

export function CitationList({ citations }: Props) {
  if (!citations.length) {
    return null;
  }

  return (
    <section className="citations">
      <h3>Sitater</h3>
      <ol>
        {citations.map((citation, index) => (
          <li key={citation.chunkId ?? citation.lawId ?? index}>
            <div className="citation-header">
              <span className="citation-index">[{index + 1}]</span>
              <div>
                <p className="citation-title">{citation.title ?? citation.section ?? "Ukjent tittel"}</p>
                <p className="citation-meta">
                  {citation.lawId && <span>Lov: {citation.lawId}</span>}
                  {citation.section && <span>Seksjon: {citation.section}</span>}
                </p>
              </div>
            </div>
            {citation.text && <p className="citation-text">{citation.text}</p>}
            {citation.url && (
              <a className="link" href={citation.url} target="_blank" rel="noreferrer">
                Åpne kilde
              </a>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
