import { useMemo, useState } from "react";
import { createMatrix, eigh } from "matsci-parse/matrix";
import { McCopyAccordion } from "mc-react-library";

import { modelLabel } from "../utils";
import { FORMATS } from "./downloads";
import MoleculeBox from "./MoleculeBox";
import MoleculeViewer from "./MoleculeViewer";

const fmt = (x) => x.toFixed(4);
const AXES = ["x", "y", "z"];
const BOHR3_TO_ANGSTROM3 = 0.148184711; // a0³ in Å³

/** Isotropic polarizability and principal values of a 3x3 tensor (row-major). */
function analyze(tensor) {
  // The model's tensors are symmetric up to round-off
  const symmetric = tensor.map((_, k) => {
    const [i, j] = [Math.floor(k / 3), k % 3];
    return (tensor[3 * i + j] + tensor[3 * j + i]) / 2;
  });
  const { values } = eigh(createMatrix(3, 3, symmetric));
  const [a, b, c] = values;
  return {
    isotropic: (tensor[0] + tensor[4] + tensor[8]) / 3,
    // Δα = sqrt(((α1-α2)² + (α2-α3)² + (α3-α1)²) / 2)
    anisotropy: Math.sqrt(((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) / 2),
    principal: Array.from(values),
  };
}

/** A 3x3 tensor (row-major) as a matrix in brackets, with the axes. */
function TensorMatrix({ tensor }) {
  const bracket = "row-span-3 w-2 self-stretch border-y-2 border-slate-800";
  return (
    <div className="flex items-center gap-3">
      <span className="font-serif text-xl pt-5 italic text-slate-800">α =</span>
      {/* Columns: axis, "[", the three components, "]" */}
      <div className="grid grid-cols-[auto_auto_repeat(3,auto)_auto] items-center gap-x-3 gap-y-1 text-sm tabular-nums">
        <span />
        <span />
        {AXES.map((a) => (
          <span key={a} className="text-center text-xs text-slate-400">
            {a}
          </span>
        ))}
        <span />
        {AXES.map((a, i) => [
          <span key={`label-${a}`} className="text-xs text-slate-400">
            {a}
          </span>,
          i === 0 && (
            <span key="open" className={`${bracket} rounded-l-md border-l-2`} />
          ),
          ...AXES.map((_, j) => (
            <span
              key={`${a}${j}`}
              className={`text-right ${i === j ? "font-semibold text-slate-900" : "text-slate-600"}`}
            >
              {fmt(tensor[3 * i + j])}
            </span>
          )),
          i === 0 && (
            <span
              key="close"
              className={`${bracket} rounded-r-md border-r-2`}
            />
          ),
        ])}
      </div>
    </div>
  );
}

function Stat({ label, value, note }) {
  return (
    <div className="px-4">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="font-semibold tabular-nums text-slate-900">{value}</div>
      {note && (
        <div className="text-xs tabular-nums text-slate-500">{note}</div>
      )}
    </div>
  );
}

function AtomTable({ structure, atomic }) {
  return (
    <div className="max-h-96 overflow-auto rounded shadow border border-slate-200">
      <table className="w-full text-sm tabular-nums">
        <thead className="sticky top-0 bg-slate-100">
          <tr>
            {[
              "#",
              "Element",
              <>
                α<sub>xx</sub>
              </>,
              <>
                α<sub>yy</sub>
              </>,
              <>
                α<sub>zz</sub>
              </>,
              <>
                α<sub>xy</sub>
              </>,
              <>
                α<sub>xz</sub>
              </>,
              <>
                α<sub>yz</sub>
              </>,
              <>
                α<sub>iso</sub>
              </>,
            ].map((h) => (
              <th
                key={h}
                className="px-2 py-1 text-right font-medium first:text-left"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {atomic.map((t, i) => (
            <tr key={i} className="odd:bg-white even:bg-slate-50">
              {/* index is left aligned */}
              <td className="px-2 text-left">{i + 1}</td>
              {/* rest is right aligned using numercis */}
              <td className="px-2 text-right">
                {structure.sites[i].species.symbol}
              </td>
              {[0, 4, 8, 1, 2, 5].map((k) => (
                <td key={k} className="px-2 text-right">
                  {fmt(t[k])}
                </td>
              ))}
              <td className="px-2 text-right">
                {fmt((t[0] + t[4] + t[8]) / 3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The polarizabilities `result` of `structure` by `model`, with the choice of
 * the next model (`selectedModel`, `onSelectModel`) over the viewer.
 */
export default function Results({
  structure,
  result,
  model,
  fileName,
  selectedModel,
  onSelectModel,
}) {
  // All downloads closed at first; one open at a time
  const [open, setOpen] = useState(null);
  const [ellipsoids, setEllipsoids] = useState(true);
  const [labels, setLabels] = useState(false);
  const delta = model.startsWith("delta");
  const molecular = result.molecular_polarizability;
  const { isotropic, anisotropy, principal } = useMemo(
    () => analyze(molecular),
    [molecular],
  );
  const downloads = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(FORMATS).map(([key, { write }]) => [
          key,
          write(structure, result, model),
        ]),
      ),
    [structure, result, model],
  );
  const baseName = (fileName ?? "molecule").replace(/\.[^.]+$/, "");

  return (
    <div className="space-y-6 pt-4">
      <section>
        <h2 className="mb-1 text-lg font-semibold text-slate-900">Results</h2>
        <div className="relative h-[420px] w-full rounded border border-slate-200">
          <MoleculeBox
            structure={structure}
            fileName={fileName}
            model={selectedModel}
            onModelChange={onSelectModel}
            ellipsoids={ellipsoids}
            onEllipsoidsChange={setEllipsoids}
            labels={labels}
            onLabelsChange={setLabels}
          />
          <MoleculeViewer
            structure={structure}
            tensors={result.atomic_polarizabilities}
            showEllipsoids={ellipsoids}
            scale={delta ? 0.6 : 0.25}
            showAxes={delta}
            showLabels={labels}
          />
          <p className="pointer-events-none absolute bottom-2 left-2 z-10 mr-2 rounded bg-white/80 px-1.5 py-0.5 text-xs text-slate-500">
            Semi-axes ∝ √|principal value|
            {delta &&
              "; axes red where the correction is positive, black where negative"}
            . drag to rotate, scroll to zoom.
          </p>
        </div>
      </section>

      <section>
        <h2 className="mb-1 text-lg font-semibold text-slate-900">
          Molecular polarizability ({modelLabel(model)})
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          Atomic units (bohr³), in the Cartesian frame of the input file;
          computed in {result.compute_time.toFixed(2)} s.
        </p>
        <div className="flex flex-wrap items-center gap-x-10">
          <TensorMatrix tensor={molecular} />
          <div className="flex flex-wrap items-center gap-3">
            <Stat
              label="Isotropic α (trace / 3)"
              value={fmt(isotropic)}
              note={`${(isotropic * BOHR3_TO_ANGSTROM3).toFixed(3)} Å³`}
            />
            <Stat label="Anisotropy Δα" value={fmt(anisotropy)} />
            <Stat
              label="Principal values α₁ ≤ α₂ ≤ α₃"
              value={principal.map((x) => x.toFixed(2)).join("  ")}
            />
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-1 text-lg font-semibold text-slate-900">
          Atom-centred contributions
        </h2>
        <p className="mb-3 text-xs text-slate-500">
          The decomposition of the molecular polarizability into atomic
          environments (they add up to the tensor above).
        </p>
        <AtomTable
          structure={structure}
          atomic={result.atomic_polarizabilities}
        />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-slate-900">Download</h2>
        {Object.entries(FORMATS).map(([key, { label, extension }]) => (
          <McCopyAccordion
            key={key}
            title={label}
            text={downloads[key]}
            filename={`${baseName}_${model}_polarizability.${extension}`}
            open={open === key}
            onToggle={() => setOpen(open === key ? null : key)}
          />
        ))}
      </section>
    </div>
  );
}
