# ---------- Stage 1: build the package and train the served model ----------
FROM python:3.12-slim AS trainer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
RUN pip install -r requirements.txt && pip install --no-deps .
COPY configs/ configs/
# Downloads BRFSS 2015 (SHA-256 verified) and trains the single served model.
RUN diabetes-xai --data-dir /build/data --results-dir /build/results --artifacts-dir /build/artifacts \
    train configs/serve.yaml

# ---------- Stage 2: slim runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/artifacts/brfss_serve LOG_LEVEL=INFO PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
RUN pip install -r requirements.txt && pip install --no-deps . && rm -rf src
COPY --from=trainer --chown=app:app /build/artifacts/brfss_serve /app/artifacts/brfss_serve
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn diabetes_xai.api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
