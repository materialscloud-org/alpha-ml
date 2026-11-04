# AlphaML app

The trained model can be found on Materials Cloud S3 s3:mc-contributed-app-data/alpha-ml/data.tar.gz it is `data.tar.gz`
MD5
`224f5b70e0a29b3071d3a532572ead5b`:

```bash
rclone copy s3:mc-contributed-app-data/alpha-ml/data.tar.gz .
```

The relevant Mateirals Cloud Archive entry is [rmp9p-1k216](https://archive.materialscloud.org/records/rmp9p-1k216)
(Yang et al., 2019)

```bash
mkdir data && tar xzf data.tar.gz -C data --strip-components=1
```

The backend uses the model as fixed tensors that have been built form the training data `tensors/`.

If you need to rebuild these from the data directory:

```bash
cd backend && uv run python scripts/build_tensors.py ../data ../tensors
```

### development

```bash
cd backend && uv sync && ALPHAML_DATA_DIR=../tensors uv run uvicorn alphaml_api.main:app --reload   # :8000
cd frontend && pnpm install && pnpm dev   # :5173, proxies /api to :8000
```

### docker

```bash
docker compose up --build
# goto http://localhost:8000
```

### Publishing to Materials Cloud harbor

```bash
TAG=$(date +%Y.%m.%d-%H%M%S)
IMAGE=harbor.materialscloud.cscs.ch/contributed/tools-alphaml

docker build -t $IMAGE:latest .
docker push $IMAGE:latest
docker tag $IMAGE:latest $IMAGE:$TAG
docker push $IMAGE:$TAG
```

### History

Until version 1.1 Beta (2026) this was a Python 2 Flask app on the Materials
Cloud `tools-barebone` image (Apache/Passenger), with the prediction in a
Celery task (Redis), Cython kernels and a JSmol viewer. That implementation is
in the git history. The model and its data are unchanged: the new backend is a numpy port that reproduces the old behaviour.

## ABOUT

Web app that predicts the dipole polarizability of molecules (H, C, N, O, S,
Cl) with symmetry-adapted Gaussian process regression trained on QM7b, and its
decomposition into atom-centred contributions. Five models: CCSD, B3LYP,
SCAN0, and the Δ(CCSD-B3LYP) and Δ(CCSD-SCAN0) corrections.

Developed by the Ceriotti group (EPFL) and the DiStasio group (Cornell). Please
cite: D. M. Wilkins, A. Grisafi, Y. Yang, K. U. Lao, R. A. DiStasio Jr. and
M. Ceriotti, _Accurate molecular polarizabilities by coupled-cluster theory and
machine learning_, Proc. Natl. Acad. Sci. 116, 3401 (2019),
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
