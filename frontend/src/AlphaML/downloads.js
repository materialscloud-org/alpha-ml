// The downloads: the molecule as matsci-parse writes it, with the
// polarizabilities (atomic units) added to the text in each format's own way.

import { cartesian, toCIF, toPDB, toXYZ } from "matsci-parse";

import { modelLabel } from "../utils";

// The six independent components
const COMPONENTS = [
  ["xx", 0],
  ["yy", 4],
  ["zz", 8],
  ["xy", 1],
  ["xz", 2],
  ["yz", 5],
];

const CITATION =
  "David M. Wilkins, Andrea Grisafi, Yang Yang, Ka Un Lao, Robert A. DiStasio and Michele Ceriotti: Accurate molecular polarizabilities by coupled-cluster theory and machine learning Proc. Natl. Acad. Sci. 116, 3401 (2019).";

const six = (tensor) => COMPONENTS.map(([, k]) => tensor[k]);
const fixed = (tensor) => six(tensor).map((x) => x.toFixed(6));

/** Extended XYZ, as the legacy download: six more columns per atom, the
 * molecular polarizability, the model and the citation in the comment line.
 * Without the Lattice of toXYZ: a molecule has no cell (AlphaML never used
 * one). */
function xyz(structure, result, model) {
  const [count, header, ...atoms] = toXYZ(structure).split("\n");
  const columns = COMPONENTS.map(([name]) => `:a${name}:R:1`).join("");
  const comment =
    ` ${model}_polarizability="${fixed(result.molecular_polarizability).join(" ")}"` +
    ` Comments="Polarizabilities (atomic units) predicted by AlphaML with the ${modelLabel(model)} model.` +
    ` Please cite: ${CITATION}` +
    ` The elements given are axx,ayy,azz,axy,axz,ayz; the per-atom terms add up to the molecular polarizability."`;

  return [
    count,
    header
      .replace(/Lattice="[^"]*"\s*/, "")
      .replace(/(Properties=\S+)/, `$1${columns}`) + comment,
    ...atoms.map(
      (line, i) =>
        `${line} ${fixed(result.atomic_polarizabilities[i]).join(" ")}`,
    ),
  ].join("\n");
}

/** PDB, as the legacy download: each atom's polarizability in an ANISOU
 * record (x 1000, truncated to integers), which PDB viewers draw as
 * ellipsoids; the molecular one in a REMARK. */
function pdb(structure, result, model) {
  let atom = 0;
  const lines = toPDB(structure)
    .split("\n")
    .flatMap((line) => {
      if (!/^(ATOM  |HETATM)/.test(line)) return [line];
      const tensor = result.atomic_polarizabilities[atom++];
      const u = six(tensor).map((x) =>
        String(Math.trunc(1000 * x)).padStart(7),
      );
      // Columns 7-28 (serial to insertion code) and 77-78 (element) as ATOM
      const anisou =
        `ANISOU${line.slice(6, 28)}${u.join("")}`.padEnd(76) +
        line.slice(76, 78);
      return [line, anisou];
    });
  const remarks = [
    `REMARK   1 AlphaML ${modelLabel(model)} polarizability (a.u.), xx yy zz xy xz yz:`,
    `REMARK   1 ${fixed(result.molecular_polarizability).join(" ")}`,
    "REMARK   1 ANISOU: the atom-centred polarizabilities x 1000 (not displacements).",
  ];
  const header = lines.findIndex((line) => !line.startsWith("HEADER"));
  lines.splice(header, 0, ...remarks);
  return lines.join("\n");
}

/** The Cartesian positions of the atoms, in the file's frame (a molecule
 * read without a cell: its box's origin added back). */
const filePositions = (structure) =>
  structure.sites.map((site) =>
    Array.from(
      cartesian(structure.lattice, site),
      (x, k) => x + (structure.moleculeBox?.origin[k] ?? 0),
    ),
  );

/** CIF of a molecule: matsci-parse's CIF (a crystal's: cell and fractional
 * positions) without the cell and symmetry, the atoms at their Cartesian
 * positions (_atom_site_Cartn_*, which matsci-parse reads back as a molecule
 * in a box). Then a loop of the atom-centred polarizabilities, keyed by the
 * atom labels, and the molecular one (custom tags, which CIF readers that do
 * not know them ignore). */
function cif(structure, result, model) {
  const positions = filePositions(structure);
  const labels = [];
  let atom = -1; // the atom-site rows, once its header is passed
  const lines = toCIF(structure)
    .replace(/\s+$/, "")
    .split("\n")
    .filter((line) => !/^_(symmetry|cell)_/.test(line))
    .map((line) => {
      if (line.startsWith("_atom_site_fract_")) {
        if (line.endsWith("_z")) atom = 0;
        return line.replace("_fract_", "_Cartn_");
      }
      if (atom < 0 || !line.trim()) return line;
      const [label, symbol] = line.trim().split(/\s+/);
      labels.push(label);
      const xyz = positions[atom++].map((x) => x.toFixed(6));
      return [label, symbol, ...xyz].join(" ");
    })
    // The blank lines left where the cell and symmetry were
    .filter((line, i, all) => line.trim() || all[i - 1]?.trim());

  return [
    ...lines,
    "",
    `# AlphaML ${modelLabel(model)} polarizabilities (atomic units). ${CITATION}`,
    `_alphaml_model ${model}`,
    ...COMPONENTS.map(
      ([name, k]) =>
        `_alphaml_polarizability_${name} ${result.molecular_polarizability[k].toFixed(6)}`,
    ),
    "",
    "loop_",
    "_alphaml_atom_polarizability_label",
    ...COMPONENTS.map(([name]) => `_alphaml_atom_polarizability_${name}`),
    ...labels.map(
      (label, i) =>
        `${label} ${fixed(result.atomic_polarizabilities[i]).join(" ")}`,
    ),
    "",
  ].join("\n");
}

/** The results as JSON, for scripts. */
function json(structure, result, model) {
  return JSON.stringify(
    {
      model,
      units: "atomic units (bohr^3)",
      symbols: structure.sites.map((site) => site.species.symbol),
      atomic_polarizabilities: result.atomic_polarizabilities,
      molecular_polarizability: result.molecular_polarizability,
      citation: CITATION,
    },
    null,
    1,
  );
}

/** The download formats: label, file extension and writer. */
export const FORMATS = {
  xyz: { label: "Extended XYZ", extension: "xyz", write: xyz },
  pdb: {
    label: "PDB (polarizabilities as ANISOU)",
    extension: "pdb",
    write: pdb,
  },
  cif: { label: "CIF", extension: "cif", write: cif },
  json: { label: "JSON", extension: "json", write: json },
};
