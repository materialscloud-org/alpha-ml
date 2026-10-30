# AlphaML app

Web app that predicts the dipole polarizability of molecules (H, C, N, O, S,
Cl) with symmetry-adapted Gaussian process regression trained on QM7b, and its
decomposition into atom-centred contributions. Five models: CCSD, B3LYP,
SCAN0, and the Δ(CCSD-B3LYP) and Δ(CCSD-SCAN0) corrections.

Developed by the Ceriotti group (EPFL) and the DiStasio group (Cornell). Please
cite: D. M. Wilkins, A. Grisafi, Y. Yang, K. U. Lao, R. A. DiStasio Jr. and
M. Ceriotti, *Accurate molecular polarizabilities by coupled-cluster theory and
machine learning*, Proc. Natl. Acad. Sci. 116, 3401 (2019),
[doi:10.1073/pnas.1816132116](https://doi.org/10.1073/pnas.1816132116).

- `frontend/`: React (Vite + Tailwind) single-page app. Molecules are parsed in
  the browser with [matsci-parse](https://www.npmjs.com/package/matsci-parse);
  the results (tensors, principal values, per-atom table, extended XYZ
  download) are made there too. `App.jsx` is the page; the tool is in
  `src/AlphaML/`.
- `backend/`: FastAPI service with one compute endpoint, `POST /api/predict`
  (atoms in, polarizability tensors out). It also serves the built frontend.
  See [backend/README.md](backend/README.md) for the API, the model and how it
  was checked against the legacy implementation.

## The model data

The trained model (training-set power spectra, sparsification matrices and the
weights of each model, 411 MB) is not in git: it is `data.tar.gz` on the
Materials Cloud S3 server. Unpack it in `data/` (gitignored):

```bash
mkdir data && tar xzf data.tar.gz -C data --strip-components=1
```

The backend uses the model as fixed tensors built from it once (about 90 MB,
in `tensors/`, gitignored):

```bash
cd backend && uv run python scripts/build_tensors.py ../data ../tensors
```

See [backend/README.md](backend/README.md). The Docker image is built from
`data/` but holds only the tensors.

## Running it

With Docker (from this folder, with `data/` unpacked):

```bash
docker compose up --build   # http://localhost:8000 (ALPHAML_PORT=8080 for another port)
```

or `docker build -t alphaml . && docker run -p 8000:8000 alphaml`. The
`Dockerfile` builds the frontend and the backend into one image, with the
model data.

For development, run the two separately:

```bash
cd backend && uv sync && ALPHAML_DATA_DIR=../tensors uv run uvicorn alphaml_api.main:app --reload   # :8000
cd frontend && pnpm install && pnpm dev   # :5173, proxies /api to :8000
```

Tests: `cd backend && uv run pytest` (API tests), and with
`ALPHAML_DATA_DIR=../tensors` also the comparison with the legacy predictions.
Settings (threads, memory): [backend/README.md](backend/README.md).

## Input files

Upload XYZ (plain or extended), PDB or CIF. Only the atoms are sent to the
backend: molecules are not periodic, and a cell in the file is not used (the
app says so when there is one). Files without a cell (plain XYZ, PDB without
CRYST1) are read by matsci-parse (0.18.18) into a padded box, which the app
does not show.

## History

Until version 1.1 Beta (2019) this was a Python 2 Flask app on the Materials
Cloud `tools-barebone` image (Apache/Passenger), with the prediction in a
Celery task (Redis), Cython kernels and a JSmol viewer. That implementation is
in the git history and in the monorepo at `apps/contributed/legacy/alpha-ml`.
The model and its data are unchanged: the new backend is a numpy port that
reproduces the legacy predictions of the six example molecules with every
model to 4e-9 (relative to the largest tensor entry; see `backend/README.md`).
