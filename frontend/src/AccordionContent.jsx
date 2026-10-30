// Content of the information accordions of the landing page (ported from the
// legacy Flask app: webservice/templates/visualizer_select_base.html).

import { useState } from "react";
import { DoiBadge, McInfoAccordion } from "mc-react-library";

import { ALLOWED_ELEMENTS, MAX_NUMBER_OF_ATOMS, PAPER_DOI } from "./utils";

/** "About AlphaML" accordion. */
function AboutContent() {
  return (
    <div className="space-y-3">
      <p>
        AlphaML was developed by the Ceriotti group at EPFL and the DiStasio
        group at Cornell. It uses a machine-learning framework based on
        comparisons of local atomic environments to predict the polarizabilities
        of molecules. The model was trained on the 7211 compounds of the QM7b
        dataset, containing up to 7 heavy atoms (C, N, O, S, Cl) as well as
        hydrogen. For these molecules, the polarizability was calculated with
        B3LYP-DFT, SCAN0-DFT and CCSD.
      </p>
      <p>
        A model trained on the 5400 most diverse members of the QM7b set
        predicts the CCSD polarizability of the remaining molecules with an
        error of 0.04 a.u., an improvement of an order of magnitude over the
        prediction of DFT, and the polarizability of a showcase set of more
        complex molecules with an error of 0.24 a.u., comparable in accuracy to
        the DFT prediction.
      </p>
    </div>
  );
}

/** "What AlphaML does" accordion. */
function WhatAlphaMLDoesContent() {
  return (
    <div className="space-y-3">
      <p>
        AlphaML takes as an input the{" "}
        <strong>atomic coordinates of a molecule</strong> and predicts its{" "}
        <strong>dipole polarizability</strong>, as well as a decomposition of
        this polarizability into atom-centred environments. You can choose the
        reference the model was trained on: CCSD, B3LYP or SCAN0, or the Δ
        models, which predict the difference between CCSD and DFT.
      </p>
      <ul className="list-disc pl-10">
        <li>
          Molecules with {ALLOWED_ELEMENTS.join(", ")} only, up to{" "}
          {MAX_NUMBER_OF_ATOMS} atoms.
        </li>
        <li>
          AlphaML has so far been tested for DFT-optimized configurations of
          such molecules.
        </li>
        <li>
          Upload XYZ (plain or extended), PDB or CIF files. A cell in the file
          is not used: the atoms are treated as one molecule, without periodic
          images.
        </li>
        <li>
          The results can be downloaded as extended XYZ, with the per-atom
          polarizability as extra columns.
        </li>
      </ul>
    </div>
  );
}

/** "How to cite" accordion. */
function HowToCiteContent() {
  return (
    <div>
      If you use this tool, please cite: David M. Wilkins, Andrea Grisafi, Yang
      Yang, Ka Un Lao, Robert A. DiStasio Jr. and Michele Ceriotti,{" "}
      <i>
        Accurate molecular polarizabilities by coupled-cluster theory and
        machine learning
      </i>
      , Proc. Natl. Acad. Sci. 116, 3401 (2019){" "}
      <span className="inline-block align-middle">
        <DoiBadge doi={PAPER_DOI} label="PNAS" color="#a2e5b7" />
      </span>
      .
    </div>
  );
}

/** "Version history" accordion. */
function VersionHistoryContent() {
  return (
    <ul className="list-disc space-y-1 pl-10">
      <li>
        2026: new web app (React frontend, Python 3 backend); the predictions
        are those of version 1.1.
      </li>
      <li>
        02.03.2019: version 1.1 Beta. Inclusion of radial scaling to improve the
        accuracy of predictions.
      </li>
      <li>
        13.08.2018: version 1.0 Beta. Implementation of the AlphaML framework as
        described in Wilkins et al.
      </li>
    </ul>
  );
}

const ACCORDIONS = [
  { title: "About AlphaML", content: <AboutContent /> },
  { title: "What AlphaML does", content: <WhatAlphaMLDoesContent /> },
  { title: "How to cite", content: <HowToCiteContent /> },
  { title: "Version history", content: <VersionHistoryContent /> },
];

/** The information accordions at the top of the page, one open at a time. */
export default function InfoAccordions() {
  const [open, setOpen] = useState(null);
  return ACCORDIONS.map(({ title, content }, i) => (
    <McInfoAccordion
      key={title}
      title={title}
      open={open === i}
      onToggle={() => setOpen(open === i ? null : i)}
    >
      {content}
    </McInfoAccordion>
  ));
}
