import { hillFormula } from "matsci-parse";
import { formatChemicalFormula } from "mc-react-library";

import { MODELS } from "../utils";

function Checkbox({ checked, onChange, children }) {
  return (
    <label className="flex items-center gap-1.5">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {children}
    </label>
  );
}

/**
 * Over the viewer's top-left corner: the molecule (formula, atoms, file), the
 * choice of model and the viewer's options (ellipsoids, atom labels).
 */
export default function MoleculeBox({
  structure,
  fileName,
  model,
  onModelChange,
  ellipsoids,
  onEllipsoidsChange,
  labels,
  onLabelsChange,
}) {
  return (
    <div className="absolute z-10 max-w-xs space-y-1.5  shadow  bg-slate-500/5 px-4.5 py-1.5 text-sm text-slate-800">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-base font-semibold text-slate-900">
          Formula: {formatChemicalFormula(hillFormula(structure))}
        </span>
        <span className="text-xs text-slate-700 italic">
          {structure.sites.length} atoms{fileName && <>, {fileName}</>}
        </span>
      </div>
      {/* matsci-parse sets moleculeBox when it made the cell itself */}
      {/* {!structure.moleculeBox && (
        <p className="text-xs text-amber-700">
          Your file has a cell: AlphaML does not use it. The atoms are treated
          as one molecule, without periodic images.
        </p>
      )} */}
      <label className="flex items-center gap-2">
        Selected Model
        <select
          className="rounded border border-slate-300 bg-white px-1 py-0.5"
          title="CCSD, or the DFT models; Δ: the CCSD correction to the DFT one"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {MODELS.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <Checkbox checked={ellipsoids} onChange={onEllipsoidsChange}>
        Polarizability ellipsoids
      </Checkbox>
      <Checkbox checked={labels} onChange={onLabelsChange}>
        Atom labels
      </Checkbox>
    </div>
  );
}
