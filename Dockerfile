# Stage 1: dependency builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: production image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal32 \
        curl \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN addgroup --system greenscore && adduser --system --ingroup greenscore greenscore

WORKDIR /app

COPY --chown=greenscore:greenscore . .

RUN mkdir -p models && chown greenscore:greenscore models

RUN mkdir -p /etc/supervisor/conf.d
COPY docker/supervisord.conf /etc/supervisor/conf.d/greenscore.conf

# Streamlit (8501) + FastAPI (8000)
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health && \
        curl --fail http://localhost:8000/health || exit 1

# supervisord runs as root, spawns child processes as greenscore (set in supervisord.conf)
USER root
ENTRYPOINT ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/greenscore.conf"]
