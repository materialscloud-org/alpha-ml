// Shared constants of the app (the limits of the legacy AlphaML app).

export const MAX_NUMBER_OF_ATOMS = 2000;
export const ALLOWED_ELEMENTS = ["H", "C", "N", "O", "S", "Cl"];

export const PAPER_DOI = "10.1073/pnas.1816132116";

// The models of the backend, with the labels of the legacy app.
export const MODELS = [
  { value: "ccsd", label: "CCSD" },
  { value: "b3lyp", label: "B3LYP" },
  { value: "scan0", label: "SCAN0" },
  { value: "delta-b3lyp-ccsd", label: "Δ(CCSD-B3LYP)" },
  { value: "delta-scan0-ccsd", label: "Δ(CCSD-SCAN0)" },
];

export const modelLabel = (value) =>
  MODELS.find((m) => m.value === value)?.label ?? value;
