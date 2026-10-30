import { isDisordered } from "matsci-parse";
import { McStructureUpload } from "mc-react-library";

import { ALLOWED_ELEMENTS, MAX_NUMBER_OF_ATOMS } from "../utils";
import { examples } from "./examples";

// The checks of the legacy app (webservice/seekpath_web_module.py), with its
// messages; the backend repeats them.
function checkMolecule(structure) {
  if (isDisordered(structure)) {
    throw new Error(
      "This structure contains partial occupancies. AlphaML only supports ordered molecules.",
    );
  }
  const natoms = structure.sites.length;
  if (natoms > MAX_NUMBER_OF_ATOMS) {
    throw new Error(
      `Sorry, this online app is limited to ${MAX_NUMBER_OF_ATOMS} atoms in the input cell, while your structure has ${natoms} atoms.`,
    );
  }
  const species = [...new Set(structure.sites.map((s) => s.species.symbol))];
  const other = species.filter((s) => !ALLOWED_ELEMENTS.includes(s));
  if (other.length > 0) {
    throw new Error(
      `Sorry, AlphaML is limited to the predictions of polarizabilities of molecules containing ${ALLOWED_ELEMENTS.join(", ")}. The atomic species you provided are ${species.join(", ")}.`,
    );
  }
}

export default function MoleculeUpload(props) {
  return (
    <McStructureUpload
      historyKey="alphaml.parsedStructures"
      examples={examples}
      validate={checkMolecule}
      label="Upload a molecule"
      title="Drop your molecule here"
      examplesTitle="Otherwise, pick an example molecule"
      accept=".xyz,.extxyz,.pdb,.cif"
      hint={`XYZ, PDB and CIF files of molecules with ${ALLOWED_ELEMENTS.join(", ")} (up to ${MAX_NUMBER_OF_ATOMS} atoms)`}
      {...props}
    />
  );
}
