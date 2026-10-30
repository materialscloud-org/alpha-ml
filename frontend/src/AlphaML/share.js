// Share links with the result. mc-react-library's structure links hold only
// the input, and apps recompute on load; AlphaML predictions take seconds to
// minutes on the server, so its links also carry the result, under
// `alphaml`. Other apps open them as plain structure links.

import { toStructureShare } from "mc-react-library";

import { MODELS } from "../utils";

/** `{ structure, fileName, result }` -> share document. `result`:
 * `{ model, data }` as computed by the backend. */
export const toAlphaMLShare = ({ structure, fileName, result }) => ({
  ...toStructureShare({ structure, fileName }),
  ...(result && {
    alphaml: {
      model: result.model,
      atomic_polarizabilities: result.data.atomic_polarizabilities,
      molecular_polarizability: result.data.molecular_polarizability,
      compute_time: result.data.compute_time,
    },
  }),
});

const isTensor = (t) => Array.isArray(t) && t.length === 9 && t.every(Number.isFinite);

/** The result in a share document, if there is one for `structure`:
 * `{ model, data }`, or null. Anyone can write a share document, so it is
 * checked against the molecule before it is shown. */
export function resultFromShare(doc, structure) {
  const r = doc?.alphaml;
  if (
    !r ||
    !MODELS.some((m) => m.value === r.model) ||
    !isTensor(r.molecular_polarizability) ||
    !Array.isArray(r.atomic_polarizabilities) ||
    r.atomic_polarizabilities.length !== structure.sites.length ||
    !r.atomic_polarizabilities.every(isTensor)
  ) {
    return null;
  }
  return {
    model: r.model,
    data: {
      atomic_polarizabilities: r.atomic_polarizabilities,
      molecular_polarizability: r.molecular_polarizability,
      compute_time: Number.isFinite(r.compute_time) ? r.compute_time : 0,
    },
  };
}
