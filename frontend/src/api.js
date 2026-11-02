// Client for the AlphaML backend.
//
// Contract (POST /api/predict), see backend/alphaml_api/schemas.py:
//
// Request body:
//   {
//     symbols: string[],               // one per atom
//     positions: [x, y, z][],          // Cartesian, Å
//     model: "ccsd" | "b3lyp" | "scan0" | "delta-b3lyp-ccsd" | "delta-scan0-ccsd",
//   }
//
// Response body (200):
//   {
//     atomic_polarizabilities: number[9][],  // per atom, row-major 3x3, a.u.
//     molecular_polarizability: number[9],   // their sum
//     compute_time: number,                  // seconds
//   }
//
// Molecules: only the atoms are sent (moleculeInput). The cell matsci-parse reads (a box
// around the molecule, e.g. Lattice= of an extended XYZ file) is not used by
// the model, which takes no periodic images.
//
// Errors: non-2xx with FastAPI's { detail: string | [{ msg, ... }] }.

import { cartesian } from "matsci-parse";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function errorMessage(detail, status) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((d) => d.msg ?? String(d)).join("; ");
  }
  return `Request failed (${status})`;
}

/** The atoms of `structure` as the backend takes them: `{ symbols, positions }`
 * (plain arrays, also the cache key of the prediction). */
export const moleculeInput = (structure) => ({
  symbols: structure.sites.map((site) => site.species.symbol),
  positions: structure.sites.map((site) =>
    Array.from(cartesian(structure.lattice, site)),
  ),
});

export async function predictPolarizabilities(molecule, model, { signal } = {}) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...molecule, model }),
    signal,
  });

  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(errorMessage(detail, res.status));
  }

  return res.json();
}
