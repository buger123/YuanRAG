# syntax=docker/dockerfile:1.7

# ---- Frontend build stage ----
FROM node:20-alpine AS frontend
WORKDIR /build/src/frontend
COPY src/frontend/package.json src/frontend/package-lock.json* ./
RUN npm ci || npm install
COPY src/frontend/ ./
RUN npm run build

# ---- Python runtime stage ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for Docling/Tantivy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cache layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Frontend build artifact
COPY --from=frontend /build/src/frontend/dist ./src/frontend/dist

# Default data dir inside container
ENV RAG_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket, sys; s=socket.socket(); \
    sys.exit(0 if s.connect_ex(('127.0.0.1', 8765)) == 0 else 1)"

CMD ["python", "-m", "src.main"]
