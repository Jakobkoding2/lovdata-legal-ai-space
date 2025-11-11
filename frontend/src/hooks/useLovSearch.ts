import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { performSearch, SearchRequest, SearchResponse } from "../lib/api";
import { SearchParams } from "../components/SearchForm";

export function useLovSearch() {
  const abortController = useRef<AbortController | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<Date | null>(null);

  const mutation = useMutation<SearchResponse, Error, SearchParams>({
    mutationFn: async (params) => {
      if (abortController.current) {
        abortController.current.abort();
      }
      const controller = new AbortController();
      abortController.current = controller;
      const payload: SearchRequest = {
        query: params.query,
        top_k: params.topK,
        use_hybrid: params.useHybrid,
        with_citations: params.withCitations,
        rerank: params.rerank,
      };
      const result = await performSearch(payload, controller.signal);
      setLastFetchedAt(new Date());
      return result;
    },
  });

  return {
    runSearch: mutation.mutate,
    data: mutation.data,
    error: mutation.error,
    isPending: mutation.isPending,
    lastFetchedAt,
  };
}
