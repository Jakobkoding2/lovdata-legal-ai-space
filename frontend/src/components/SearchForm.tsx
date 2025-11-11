import { FormEvent, useState } from "react";
import clsx from "clsx";

export type SearchParams = {
  query: string;
  topK: number;
  useHybrid: boolean;
  withCitations: boolean;
  rerank: boolean;
};

type Props = {
  defaultValues: SearchParams;
  isLoading: boolean;
  onSubmit: (values: SearchParams) => void;
};

export function SearchForm({ defaultValues, onSubmit, isLoading }: Props) {
  const [values, setValues] = useState<SearchParams>(defaultValues);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit(values);
  };

  const handleInputChange = <K extends keyof SearchParams>(key: K, value: SearchParams[K]) => {
    setValues((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <label className="field">
        <span className="field__label">Juridisk spørsmål</span>
        <textarea
          className="textarea"
          placeholder="F.eks. Hvilket ansvar har styret i et aksjeselskap?"
          required
          value={values.query}
          onChange={(event) => handleInputChange("query", event.target.value)}
          rows={3}
        />
      </label>

      <div className="field-grid">
        <label className="field">
          <span className="field__label">Antall treff (top-k)</span>
          <input
            type="range"
            min={3}
            max={15}
            value={values.topK}
            onChange={(event) => handleInputChange("topK", Number(event.target.value))}
          />
          <span className="range-value">{values.topK}</span>
        </label>
        <label className="field field--checkbox">
          <input
            type="checkbox"
            checked={values.useHybrid}
            onChange={(event) => handleInputChange("useHybrid", event.target.checked)}
          />
          <span>Hybrid-søk</span>
        </label>
        <label className="field field--checkbox">
          <input
            type="checkbox"
            checked={values.withCitations}
            onChange={(event) => handleInputChange("withCitations", event.target.checked)}
          />
          <span>Krev sitater</span>
        </label>
        <label className="field field--checkbox">
          <input
            type="checkbox"
            checked={values.rerank}
            onChange={(event) => handleInputChange("rerank", event.target.checked)}
          />
          <span>Aktiver reranking</span>
        </label>
      </div>

      <div className="actions">
        <button className={clsx("button", { "button--loading": isLoading })} type="submit" disabled={isLoading}>
          {isLoading ? "Søker..." : "Søk i Lovdata"}
        </button>
      </div>
    </form>
  );
}
