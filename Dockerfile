FROM python:3.11-slim

# libgomp: OpenMP runtime used by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

# Model artifact is mounted (or baked in by CI after `churn train`).
ENV CHURN_MODEL_DIR=/app/artifacts/model \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
