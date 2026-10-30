# AlphaML: React frontend + FastAPI backend in one image.
# Needs the training data unpacked in data/ (data.tar.gz, from the Materials
# Cloud S3 server; not in git). The image holds only the model's tensors built
# from it (backend/scripts/build_tensors.py, ~90 MB instead of 411 MB):
#   mkdir data && tar xzf data.tar.gz -C data --strip-components=1
#   docker compose up --build        # http://localhost:8000
# or: docker build -t alphaml . && docker run -p 8000:8000 alphaml

FROM node:24-slim AS frontend
RUN npm install -g pnpm@11.5.3
WORKDIR /src/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim AS backend
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/.python-version backend/README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/alphaml_api ./alphaml_api
RUN uv sync --frozen --no-dev

# The model's tensors, from the training data (which stays in this stage)
FROM backend AS tensors
COPY backend/scripts ./scripts
COPY data /data
RUN .venv/bin/python scripts/build_tensors.py /data /tensors

FROM backend
COPY --from=frontend /src/frontend/dist /app/frontend/dist
COPY --from=tensors /tensors /app/tensors

# The settings: backend/alphaml_api/config.py (ALPHAML_THREADS, the chunk
# sizes, ...). One BLAS thread: no faster with more, for these matrix sizes.
ENV FRONTEND_DIST=/app/frontend/dist ALPHAML_DATA_DIR=/app/tensors OMP_NUM_THREADS=1
EXPOSE 8000
# The tensors are memory-mapped, so several workers share them.
CMD ["/app/backend/.venv/bin/uvicorn", "alphaml_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
