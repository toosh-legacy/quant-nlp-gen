# Volatility Radar — Hugging Face Space (Docker SDK).
#
# Two stages: Node builds the React dashboard, then a slim Python image serves it with the
# FastAPI app. The container is read-only by design: it serves the reports that the daily
# GitHub Actions refresh pushes into this Space, and never fetches or trains anything.

FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

# Spaces run the container as uid 1000; give that user the app directory.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY --chown=user volradar/ ./volradar/
COPY --chown=user data/reports/ ./data/reports/
COPY --chown=user --from=web /web/dist ./frontend/dist

USER user
EXPOSE 7860
CMD ["python", "-m", "volradar.cli", "serve"]
