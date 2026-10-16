FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
LABEL org.opencontainers.image.title="mathematician-knowledge-analysis" \
      org.opencontainers.image.description="Mathematician knowledge analysis with ReAct and GraphRAG"
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    MATHKB_ENV=prod
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.1,<3' && \
      pip install --no-cache-dir -r requirements-ml.txt; \
    fi
COPY pyproject.toml alembic.ini ./
COPY migrations/ ./migrations/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/seed/ ./data/seed/
COPY --from=frontend /build/frontend/dist/ ./frontend/dist/
RUN mkdir -p /app/.index /app/reports /app/data/seed/uploads && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 \
  CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); assert r['status']=='ok'"
CMD ["uvicorn", "mathkb.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
