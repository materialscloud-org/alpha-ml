import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { McloudSpinner, McResults } from "mc-react-library";

import { moleculeInput, predictPolarizabilities } from "../api";
import Results from "./Results";

/**
 * Once a molecule is loaded: what it is, and the polarizabilities of
 * `structure` (computed by the backend, again for each molecule and model)
 * with their results, where the model is chosen. `fileName` names the structure.
 * Nothing before.
 */
export default function AlphaML({ structure, fileName, className = "" }) {
  const [model, setModel] = useState("ccsd");
  const molecule = useMemo(
    () => structure && moleculeInput(structure),
    [structure],
  );

  // Cached by molecule (its atoms, not the file) and model: going back to
  // either shows the result at once. A request no longer wanted is cancelled.
  // The query's data: { structure, fileName, model, data }, the result of
  // that input (the previous one stays, dimmed, while the next is computed).
  const query = useQuery({
    queryKey: ["predict", model, molecule],
    queryFn: async ({ signal }) => ({
      structure,
      fileName,
      model,
      data: await predictPolarizabilities(molecule, model, { signal }),
    }),
    enabled: !!molecule,
    placeholderData: keepPreviousData,
  });
  const computing = query.isFetching;
  const error = query.error?.message;
  // A cached result: named after the file just loaded
  const result = useMemo(
    () =>
      query.data &&
      (query.isPlaceholderData ? query.data : { ...query.data, fileName }),
    [query.data, query.isPlaceholderData, fileName],
  );

  if (!structure) return null;

  return (
    <div className={className}>
      <div className="mx-auto max-w-6xl px-6 py-4 sm:px-10">
        {error && (
          <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        )}

        {/* The spinner over the (dimmed) previous result, without moving it */}
        <div className={result ? "relative" : "relative min-h-32"}>
          <McResults busy={computing} resultKey={result}>
            {result && (
              <Results
                structure={result.structure}
                result={result.data}
                model={result.model}
                fileName={result.fileName}
                selectedModel={model}
                onSelectModel={setModel}
              />
            )}
          </McResults>
          {computing && (
            <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center pt-40">
              <McloudSpinner
                className="w-48"
                aria-label={`Predicting the polarizability of ${fileName}`}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
