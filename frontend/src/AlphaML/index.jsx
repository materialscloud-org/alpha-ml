import { useEffect, useRef, useState } from "react";
import { hillFormula } from "matsci-parse";
import { fetchShare, formatChemicalFormula, McloudSpinner } from "mc-react-library";

import { predictPolarizabilities } from "../api";
import { MODELS } from "../utils";
import Results from "./Results";
import { resultFromShare } from "./share";

// Rough time of a prediction on the backend: measured on the example
// molecules (13 to 60 atoms: 1.7 to 7 s), about linear in the atoms.
const estimatedSeconds = (natoms) => Math.max(1, Math.round(0.5 + 0.12 * natoms));
const SLOW_ATOMS = 200; // ~25 s and more

/**
 * Once a molecule is loaded: what it is, the choice of model, the Calculate
 * button, the polarizabilities of `structure` (computed by the backend) and
 * their results. `fileName` names the structure. Nothing before.
 *
 * A molecule opened from a share link (`shareId`, in the kv-store at `kvUrl`)
 * shows the result the link holds, if any, without computing it again.
 * `onResult({ model, data, fromLink })` (or null) reports the result shown,
 * for the share button.
 */
export default function AlphaML({ structure, fileName, shareId, kvUrl, onResult, className = "" }) {
  const [model, setModel] = useState("ccsd");
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState(null);
  // { structure, fileName, model, data, fromLink }: the result belongs to that input
  const [result, setResult] = useState(null);
  const request = useRef(null);
  const calculating = useRef(false); // a link's result must not replace a calculation

  // A new molecule: the earlier result and error belong to the previous one,
  // and a calculation still running is cancelled (adjusting state while
  // rendering; the effect below aborts the request)
  const [current, setCurrent] = useState(structure);
  if (structure !== current) {
    setCurrent(structure);
    setResult(null);
    setError(null);
    setComputing(false);
  }

  // A newer request cancels the one in flight.
  const calculate = () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;

    setResult(null);
    setError(null);
    setComputing(true);
    calculating.current = true;
    predictPolarizabilities(structure, model, { signal: controller.signal })
      .then((data) => setResult({ structure, fileName, model, data }))
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message ?? String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setComputing(false);
      });
  };

  // Also when the molecule changes or the page goes
  useEffect(
    () => () => {
      request.current?.abort();
      calculating.current = false;
    },
    [structure],
  );

  // A share link's result (fetched again: the upload passes on the structure only)
  useEffect(() => {
    if (!structure || !shareId) return;
    let live = true;
    fetchShare(shareId, kvUrl)
      .then((doc) => {
        const shared = resultFromShare(doc, structure);
        if (!live || !shared || calculating.current) return;
        setModel(shared.model);
        setResult((current) => current ?? { structure, fileName, ...shared, fromLink: true });
      })
      .catch(() => {}); // the structure loaded; without the result, Calculate as usual
    return () => {
      live = false;
    };
  }, [structure, fileName, shareId, kvUrl]);

  useEffect(() => {
    onResult?.(result && { model: result.model, data: result.data, fromLink: !!result.fromLink });
  }, [result, onResult]);

  if (!structure) return null;
  const natoms = structure.sites.length;

  return (
    <div className={className}>
      <div className="mx-auto max-w-4xl px-6 sm:px-10">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
          <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="font-semibold text-slate-900">Loaded molecule:</span>
            <span className="text-base font-semibold text-slate-900">
              {formatChemicalFormula(hillFormula(structure))}
            </span>
            <span className="text-slate-500">
              {natoms} atoms{fileName && <>, from {fileName}</>}
            </span>
          </div>
          {/* matsci-parse sets moleculeBox when it made the cell itself */}
          {!structure.moleculeBox && (
            <p className="mb-3 text-xs text-amber-700">
              Your file has a cell: AlphaML does not use it. The atoms of the
              file are treated as one molecule, without periodic images.
            </p>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2">
              <span>AlphaML model</span>
              <select
                className="rounded border border-slate-300 bg-white px-2 py-1"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {MODELS.map(({ value, label }) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="rounded bg-[#337ab7] px-4 py-1.5 font-medium text-white hover:bg-[#286090] disabled:cursor-not-allowed disabled:opacity-50"
              disabled={computing}
              onClick={calculate}
            >
              Calculate
            </button>
            <span className="text-slate-500">
              estimated ~{estimatedSeconds(natoms)} s
              {natoms > SLOW_ATOMS && (
                <strong className="ml-1 text-amber-700">This may take a while!</strong>
              )}
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            CCSD, or the DFT models; Δ: the CCSD correction to the DFT one.
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-6 py-4 sm:px-10">
        {computing && (
          <div className="flex justify-center py-12">
            <McloudSpinner className="w-32" aria-label={`Predicting the polarizability of ${fileName}`} />
          </div>
        )}

        {error && (
          <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        )}

        {result?.fromLink && (
          <p className="mb-4 rounded-lg bg-sky-50 px-3 py-2 text-sm text-sky-800">
            These results come with the share link: they were calculated when
            it was made, not now. Press Calculate to compute them again.
          </p>
        )}

        {result && (
          <Results
            structure={result.structure}
            result={result.data}
            model={result.model}
            fileName={result.fileName}
          />
        )}
      </div>
    </div>
  );
}
